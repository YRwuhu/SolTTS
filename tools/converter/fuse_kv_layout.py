#!/usr/bin/env python3
"""Store the KV cache in attention-native layout and delete the per-layer fix-up ops.

The exported decoder stores `past_*`/`present_*` as `[S, 1, 512]` and then, inside
every one of the 24 layers, reshapes and transposes the *whole cache* before
attention can read it:

    K:  present_k [S,1,512] -> Reshape[-1,16,32] -> Unsqueeze[0]
                            -> Transpose[0,2,3,1] -> [1,16,32,S]
    V:  present_v [S,1,512] -> Reshape[-1,16,32] -> Transpose[1,0,2]
                            -> Unsqueeze[0] -> [1,16,S,32]

Both chains touch every cached token, so they scale with sequence length, and
they exist only to undo an export-time layout choice. ORT's kernel profile puts
`Transpose`+`Concat`+`Reshape`+`Unsqueeze` at 38.7% of a decode step; these 48
chains are the part of that which can be removed outright.

The rewrite stores the cache the way attention consumes it:

    past/present K: [1,16,32,S]      Concat on the last axis
    past/present V: [1,16,S,32]      Concat on axis 2

Per layer that removes 6 layout nodes that touch the whole cache (3 for K, 3 for
V) and replaces them with 2 small Reshapes on the *new token only*, whose cost
does not grow with `S`.

The external KV interface therefore changes shape. A caller that fed
`[S,1,512]` must now feed `[1,16,32,S]` for K and `[1,16,S,32]` for V, and the
`present_*` outputs come back in the same layout.

Usage:
    python fuse_kv_layout.py in.onnx out.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

HEADS = 16
HEAD_DIM = 32
LAYERS = 24
SEQ_IN = "past_sequence_len"
SEQ_OUT = "present_sequence_len"


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    graph = model.graph
    nodes = list(graph.node)

    def consumer_of(name: str, op: str):
        found = [n for n in nodes if n.op_type == op and name in n.input]
        if len(found) != 1:
            raise SystemExit(f"{name}: expected one {op} consumer, found {len(found)}")
        return found[0]

    def producer_of(name: str):
        for n in nodes:
            if name in n.output:
                return n
        raise SystemExit(f"{name}: no producer")

    dead = set()
    added = []

    for layer in range(LAYERS):
        pairs = (
            (f"present_k_layer_{layer}", "k", 3, [1, HEADS, HEAD_DIM, 1]),
            (f"present_v_layer_{layer}", "v", 2, [1, HEADS, 1, HEAD_DIM]),
        )
        for present, kind, concat_axis, token_shape in pairs:
            concat = producer_of(present)
            # The stage decoder concatenates onto the cache; the first stage emits
            # the first cache entry directly, so there is no Concat to retarget.
            joins_cache = concat.op_type == "Concat"
            if not joins_cache and concat.op_type != "Slice":
                raise SystemExit(f"{present}: producer is {concat.op_type}, expected Concat or Slice")

            reshape = consumer_of(present, "Reshape")
            if kind == "k":
                unsq = consumer_of(reshape.output[0], "Unsqueeze")
                trans = consumer_of(unsq.output[0], "Transpose")
                perm = list(next(a.ints for a in trans.attribute if a.name == "perm"))
                if perm != [0, 2, 3, 1]:
                    raise SystemExit(f"{present}: K perm {perm}, expected [0,2,3,1]")
                tail = trans.output[0]
                dead.update((id(reshape), id(unsq), id(trans)))
            else:
                trans = consumer_of(reshape.output[0], "Transpose")
                perm = list(next(a.ints for a in trans.attribute if a.name == "perm"))
                if perm != [1, 0, 2]:
                    raise SystemExit(f"{present}: V perm {perm}, expected [1,0,2]")
                unsq = consumer_of(trans.output[0], "Unsqueeze")
                tail = unsq.output[0]
                dead.update((id(reshape), id(trans), id(unsq)))

            if not joins_cache:
                # The first stage emits the whole initial cache in one piece and
                # runs once per sentence, so its layout chain is cheap and stays.
                # Only the *names* move: the graph output `present` becomes the
                # native-layout tensor that attention already reads, and the flat
                # tensor the chain consumed gets a private name.
                _rename(nodes, present, f"{present}__flat")
                _rename(nodes, tail, present)
                # The chain still produces `present`, so it must stay.
                dead.difference_update((id(reshape), id(trans)))
                if kind == "k":
                    dead.difference_update((id(unsq),))
                else:
                    dead.difference_update((id(unsq),))
                continue

            # Attention reads the cache directly now.
            for c in nodes:
                if c.op_type == "Concat" and present in c.output:
                    continue
                for i, inp in enumerate(c.input):
                    if inp == tail:
                        c.input[i] = present

            if joins_cache:
                # The new token is Concat'd into the cache along the sequence axis.
                shape_name = f"kv_{kind}_token_shape_{layer}"
                graph.initializer.append(numpy_helper.from_array(np.array(token_shape, dtype=np.int64), shape_name))
                token_name = f"kv_{kind}_token_{layer}"
                added.append(
                    helper.make_node("Reshape", [concat.input[1], shape_name], [token_name], name=token_name + "_node")
                )
                concat.input[1] = token_name
                for a in list(concat.attribute):
                    if a.name == "axis":
                        concat.attribute.remove(a)
                concat.attribute.append(helper.make_attribute("axis", concat_axis))

    keep = [n for n in nodes if id(n) not in dead]
    _topological_append(keep, added)
    keep = _toposort(keep)
    graph.ClearField("node")
    graph.node.extend(keep)

    for io in list(graph.input):
        if io.name.startswith("past_k_"):
            _set_dims(io, [1, HEADS, HEAD_DIM, SEQ_IN])
        elif io.name.startswith("past_v_"):
            _set_dims(io, [1, HEADS, SEQ_IN, HEAD_DIM])
    for io in list(graph.output):
        if io.name.startswith("present_k_"):
            _set_dims(io, [1, HEADS, HEAD_DIM, SEQ_OUT])
        elif io.name.startswith("present_v_"):
            _set_dims(io, [1, HEADS, SEQ_OUT, HEAD_DIM])

    pruned = _prune(graph)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, dst)

    layout_ops = sum(1 for n in graph.node if n.op_type in ("Transpose", "Reshape", "Unsqueeze", "Concat"))
    print(f"{src} -> {dst}")
    print(f"  layers rewritten: {LAYERS}, dead nodes pruned: {pruned}")
    print(f"  K cache: [1,{HEADS},{HEAD_DIM},S]   V cache: [1,{HEADS},S,{HEAD_DIM}]")
    print(f"  layout ops remaining: {layout_ops}")


def _toposort(nodes):
    """Stable topological sort; graphs edited by rename can fall out of order."""
    produced = {}
    for n in nodes:
        for o in n.output:
            produced[o] = n
    out, state = [], {}
    def visit(n):
        key = id(n)
        if state.get(key) == 2:
            return
        if state.get(key) == 1:
            raise SystemExit(f"cycle at {n.name}")
        state[key] = 1
        for i in n.input:
            if i in produced:
                visit(produced[i])
        state[key] = 2
        out.append(n)
    for n in nodes:
        visit(n)
    return out


def _rename(nodes, old, new):
    """Rename a value everywhere it is produced or consumed."""
    for n in nodes:
        for i, o in enumerate(n.output):
            if o == old:
                n.output[i] = new
        for i, i_name in enumerate(n.input):
            if i_name == old:
                n.input[i] = new


def _topological_append(nodes, added):
    """Insert each added node right after its last producer, keeping the graph sorted."""
    for new in added:
        last = -1
        for i, n in enumerate(nodes):
            if any(o in n.output for o in new.input if o):
                last = i
        nodes.insert(last + 1, new)


def _set_dims(io, dims):
    tt = io.type.tensor_type
    del tt.shape.dim[:]
    for d in dims:
        dim = tt.shape.dim.add()
        if isinstance(d, int):
            dim.dim_value = d
        else:
            dim.dim_param = d


def _prune(graph) -> int:
    before = len(graph.node)
    while True:
        used = {i for n in graph.node for i in n.input if i}
        used |= {o.name for o in graph.output}
        keep = [n for n in graph.node if any(o in used for o in n.output)]
        if len(keep) == len(graph.node):
            break
        graph.ClearField("node")
        graph.node.extend(keep)

    used = {i for n in graph.node for i in n.input if i}
    used |= {o.name for o in graph.output}
    keep_init = [i for i in graph.initializer if i.name in used]
    graph.ClearField("initializer")
    graph.initializer.extend(keep_init)
    return before - len(graph.node)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
