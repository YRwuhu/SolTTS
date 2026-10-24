#!/usr/bin/env python3
"""Delete layout ops that are numerically the identity on this graph.

The test is a real run: for every `Transpose`, the input and output tensors are
compared element for element. A permutation that only touches size-1 axes leaves
the data untouched, and in the decoder that is the 24 per-layer `Transpose_6`
nodes that undo the preceding reshape.

Comparing shapes is *not* enough: a `(2,1,0)` transpose of a tensor whose axes
happen to have equal extents keeps the same shape but reorders values. An earlier
version of this script used the shape test and shipped exactly that bug; the
numeric check here caught it.

Usage:
    python drop_noop_layout.py in.onnx out.onnx [--iy L] [--kv S]
"""
import argparse
import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def _elem_types(model):
    try:
        inf = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return {}
    return {
        v.name: v.type.tensor_type.elem_type
        for v in list(inf.graph.value_info) + list(inf.graph.output) + list(inf.graph.input)
    }


def _feed(sess, iy, kv):
    rng = np.random.default_rng(0)
    feed = {}
    for inp in sess.get_inputs():
        if inp.name == "iy":
            feed[inp.name] = np.arange(iy, dtype=np.int64)[None]
        elif inp.name == "iy_emb":
            feed[inp.name] = (rng.standard_normal((1, iy - 1, 512)) * 0.2).astype(np.float32)
        elif inp.name.startswith("past_k_"):
            feed[inp.name] = (rng.standard_normal((1, 16, 32, kv)) * 0.2).astype(np.float32)
        elif inp.name.startswith("past_v_"):
            feed[inp.name] = (rng.standard_normal((1, 16, kv, 32)) * 0.2).astype(np.float32)
        else:
            raise SystemExit(f"unexpected graph input {inp.name}")
    return feed


def identity_transposes(model, iy, kv):
    """Names of Transpose nodes whose output equals their input, by value."""
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    typed = _elem_types(probe)
    wanted = []
    for n in probe.graph.node:
        if n.op_type == "Transpose":
            wanted += [n.input[0], n.output[0]]
    known = {o.name for o in probe.graph.output}
    for name in sorted(set(wanted)):
        if name not in known:
            probe.graph.output.append(
                helper.make_tensor_value_info(name, typed.get(name, TensorProto.FLOAT), None)
            )
    path = os.path.join(tempfile.mkdtemp(prefix="noop_"), "probe.onnx")
    onnx.save(probe, path)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, _feed(sess, iy, kv))
    names = [o.name for o in sess.get_outputs()]
    values = {n: np.asarray(outs[names.index(n)]) for n in set(wanted) if n in names}

    found = []
    for n in model.graph.node:
        if n.op_type != "Transpose":
            continue
        a, b = values.get(n.input[0]), values.get(n.output[0])
        if a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b):
            found.append(n.name)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--iy", type=int, default=2)
    ap.add_argument("--kv", type=int, default=40)
    args = ap.parse_args()

    model = onnx.load(args.src)
    graph = model.graph

    names = set(identity_transposes(model, args.iy, args.kv))
    dropped = [n for n in graph.node if n.op_type == "Transpose" and n.name in names]

    if not dropped:
        print(f"{args.src}: no identity transposes, nothing to do")
        return

    drop_ids = {id(n) for n in dropped}
    out_names = {n.output[0]: n.input[0] for n in dropped}

    nodes = [n for n in graph.node if id(n) not in drop_ids]
    # Rewire consumers of a dropped node to its input; repeat so chains collapse.
    changed = True
    while changed:
        changed = False
        for n in nodes:
            for i, name in enumerate(n.input):
                if name in out_names:
                    n.input[i] = out_names[name]
                    changed = True
    graph.ClearField("node")
    graph.node.extend(nodes)

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, args.dst)
    print(f"{args.src} -> {args.dst}")
    print(f"  dropped {len(dropped)} identity Transpose node(s): {len(graph.node)} nodes remain")


if __name__ == "__main__":
    main()
