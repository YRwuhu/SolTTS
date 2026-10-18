#!/usr/bin/env python3
"""
roberta_noloop.py — replaces the Loop node in roberta.onnx that causes ~278 ms of CPU control flow
with an equivalent vectorized subgraph (a CumSum + comparison + Gather implementation of repeat_interleave).

Background (read-only analysis conclusions):
  - Original /Loop semantics = torch.repeat_interleave(hidden, repeats, dim=0)
    · trip count = seq_len-2 (Shape(Slice(hidden,1:-1))[0]), cond always true
    · each body iteration takes the i-th repeat value r_i and appends row i of hidden r_i times
  - repeats is first normalized by /Equal+/Expand+/Where (broadcast to [M] when len==1) → /Where_output_0
  - hidden content region = /Slice_output_0 ([M,1024] float16)

Replacement implementation (all operators have DirectML kernels; Loop/SplitToSequence do not):
    cum   = CumSum(/Where_output_0, axis=0)          # [M] int64 prefix sums (inclusive)
    last  = Gather(Shape(/Where_output_0), [0])      # [1] int64 total = Σr
    ones  = ConstantOfShape(last, 1)                 # [total] int64
    ridx  = CumSum(ones, axis=0) - 1                 # [total] int64  0..total-1
    mask  = LessOrEqual(cum[1,M], ridx[total,1])     # [total,M] bool  broadcast
    idx   = ReduceSum(Cast(mask,int64), axes=[1])    # [total,1] int64
    phone_features = Gather(/Slice_output_0, idx)    # [total,1024] float16

Correctness argument: output row k belongs to segment i (cum[i-1] ≤ k < cum[i]).
idx[k] = #{ j : cum[j] ≤ k }; since cum is non-decreasing, this count is exactly i when k falls in segment i
(r_i>0), and empty segments (r_i=0) are naturally never hit by any k (idx skips that value), so the result
matches repeat_interleave row by row. Manually verified with cases such as [2,3,1], [2,0,3], [0,3].

Usage:
    python roberta_noloop.py <input roberta.onnx> <output roberta_noloop.onnx>
Reads the original model only (does not load the 599 MB of weights; keeps external data references) and rewrites only the graph structure.
"""

import sys
import os
import onnx
from onnx import helper, TensorProto


def build_replacement():
    """Builds the nodes and value_info for the replacement subgraph."""
    new_nodes = []
    new_vis = []

    def add_node(op, inputs, outputs, name, **attrs):
        n = helper.make_node(op, inputs, outputs, name=name, **attrs)
        new_nodes.append(n)
        return n

    # ---- Constants ----
    # CumSum axis: this project's model family consistently uses an int32 scalar (see /ar_audio_position/CumSum in stage_decoder)
    ax0 = helper.make_tensor('/noloop/axis0', TensorProto.INT32, [], [0])
    # Scalar constants
    one = helper.make_tensor('/noloop/one', TensorProto.INT64, [], [1])
    axes0 = helper.make_tensor('/noloop/axes0', TensorProto.INT64, [1], [0])
    axes1 = helper.make_tensor('/noloop/axes1', TensorProto.INT64, [1], [1])
    unsq_axes1 = helper.make_tensor('/noloop/unsq_axes1', TensorProto.INT64, [1], [1])
    unsq_axes0 = helper.make_tensor('/noloop/unsq_axes0', TensorProto.INT64, [1], [0])

    # ---- Computation ----
    # cum = CumSum(repeats_norm, axis=0)   [M] int64 prefix sums (inclusive)
    add_node('CumSum', ['/Where_output_0', '/noloop/axis0'], ['/noloop/cum'], '/noloop/CumSum')

    # total = ReduceSum(repeats_norm, axes=[0], keepdims=1)  [1] int64 = Σr
    # Note: do not use Shape(repeats) (that is the length M, not the sum of repeat counts — measured to cause out-of-bounds indexing)
    add_node('ReduceSum', ['/Where_output_0', '/noloop/axes0'], ['/noloop/total'], '/noloop/Total', keepdims=1)

    # ones = ConstantOfShape(total, value=1) [total] int64
    add_node('ConstantOfShape', ['/noloop/total'], ['/noloop/ones'], '/noloop/ConstantOfShape',
             value=helper.make_tensor('v', TensorProto.INT64, [1], [1]))

    # onesum = CumSum(ones, axis=0)          [total] = 1..total
    add_node('CumSum', ['/noloop/ones', '/noloop/axis0'], ['/noloop/onesum'], '/noloop/CumSumOnes')

    # ridx = onesum - 1                      [total] = 0..total-1
    add_node('Sub', ['/noloop/onesum', '/noloop/one'], ['/noloop/ridx'], '/noloop/Sub')

    # cum2  = Unsqueeze(cum,   axes=[0])     [1, M]
    add_node('Unsqueeze', ['/noloop/cum', '/noloop/unsq_axes0'], ['/noloop/cum2'], '/noloop/UnsqueezeCum')

    # ridx2 = Unsqueeze(ridx,  axes=[1])     [total, 1]
    add_node('Unsqueeze', ['/noloop/ridx', '/noloop/unsq_axes1'], ['/noloop/ridx2'], '/noloop/UnsqueezeRidx')

    # mask = LessOrEqual(cum2, ridx2)  →  [total, M], element [k,j] = (cum[j] <= k)
    # idx[k] = Σ_j mask[k,j] = #{j : cum[j] <= k} = segment index i (segments with r_i=0 are skipped automatically)
    add_node('LessOrEqual', ['/noloop/cum2', '/noloop/ridx2'], ['/noloop/mask'], '/noloop/LessOrEqual')

    # maskl = Cast(mask, int64)              [total, M]
    add_node('Cast', ['/noloop/mask'], ['/noloop/maskl'], '/noloop/Cast', to=TensorProto.INT64)

    # idx = ReduceSum(maskl, axes=[1], keepdims=0)  [total] int64
    # keepdims=0 ensures the Gather output is [total,1024], matching the original model's /ConcatFromSequence result
    add_node('ReduceSum', ['/noloop/maskl', '/noloop/axes1'], ['/noloop/idx'], '/noloop/ReduceSum', keepdims=0)

    # phone_features = Gather(hidden_content, idx, axis=0)  [total,1024] float16
    add_node('Gather', ['/Slice_output_0', '/noloop/idx'], ['phone_features'], '/noloop/GatherOut', axis=0)

    extra_inits = [ax0, one, axes0, axes1, unsq_axes1, unsq_axes0]

    # ---- value_info (shape hints; dynamic dims left empty) ----
    def vi(name, dtype, shape):
        new_vis.append(helper.make_tensor_value_info(name, dtype, shape))

    vi('/noloop/cum', TensorProto.INT64, [None])
    vi('/noloop/total', TensorProto.INT64, [1])
    vi('/noloop/ones', TensorProto.INT64, [None])
    vi('/noloop/onesum', TensorProto.INT64, [None])
    vi('/noloop/ridx', TensorProto.INT64, [None])
    vi('/noloop/mask', TensorProto.BOOL, [None, None])
    vi('/noloop/maskl', TensorProto.INT64, [None, None])
    vi('/noloop/idx', TensorProto.INT64, [None])

    return new_nodes, extra_inits, new_vis


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]

    print(f'[1/6] Loading original model (without external weights): {src}')
    model = onnx.load(src, load_external_data=False)
    graph = model.graph

    # ---------- Precondition assertions: structure must match the investigation ----------
    print('[2/6] Verifying target structure...')
    node_by_name = {n.name: n for n in graph.node}

    required = ['/Where', '/Slice', '/SplitToSequence', '/SplitToSequence_1',
                '/SequenceEmpty', '/Loop', '/ConcatFromSequence']
    for name in required:
        if name not in node_by_name:
            raise SystemExit(f'Precondition failed: missing node {name} (model structure differs from expected; aborted)')

    loop = node_by_name['/Loop']
    if list(loop.input) != ['/Unsqueeze_output_0', '/Constant_8_output_0', '/SequenceEmpty_output_0']:
        raise SystemExit(f'Precondition failed: unexpected /Loop inputs: {list(loop.input)}')
    if list(loop.output) != ['/Loop_output_0']:
        raise SystemExit(f'Precondition failed: unexpected /Loop outputs: {list(loop.output)}')
    if len(loop.attribute) != 1 or loop.attribute[0].name != 'body':
        raise SystemExit('Precondition failed: /Loop attribute is not a single body')

    con = node_by_name['/ConcatFromSequence']
    if list(con.input) != ['/Loop_output_0']:
        raise SystemExit(f'Precondition failed: unexpected /ConcatFromSequence inputs: {list(con.input)}')

    where = node_by_name['/Where']
    if list(where.input) != ['/Equal_output_0', '/Expand_output_0', 'repeats']:
        raise SystemExit(f'Precondition failed: unexpected /Where inputs: {list(where.input)}')

    slice_node = node_by_name['/Slice']
    if list(slice_node.output) != ['/Slice_output_0']:
        raise SystemExit('Precondition failed: unexpected /Slice output name')

    # Confirm /Where_output_0 and /Slice_output_0 are float16 / int64 as expected
    # (indirectly via the producer output type: /Slice input comes from a LayerNorm output)
    print('    Structure validation passed.')

    # ---------- Build replacement nodes ----------
    print('[3/6] Building vectorized replacement subgraph...')
    new_nodes, extra_inits, new_vis = build_replacement()

    # ---------- Remove control-flow nodes ----------
    print('[4/6] Removing Loop/SplitToSequence/ConcatFromSequence/SequenceEmpty...')
    to_remove = {'/Loop', '/SplitToSequence', '/SplitToSequence_1', '/ConcatFromSequence',
                 '/SequenceEmpty'}
    kept_nodes = [n for n in graph.node if n.name not in to_remove]
    # Append new nodes
    kept_nodes.extend(new_nodes)
    del graph.node[:]
    graph.node.extend(kept_nodes)

    # Append initializers
    existing_init_names = {i.name for i in graph.initializer}
    for t in extra_inits:
        if t.name in existing_init_names:
            raise SystemExit(f'Precondition failed: initializer name conflict {t.name}')
        graph.initializer.append(t)

    # Append value_info
    for v in new_vis:
        graph.value_info.append(v)

    # ---------- Dead-code elimination (prune backwards from consumers) ----------
    print('[5/6] Eliminating dead code...')
    keep = set(o.name for o in graph.output)
    changed = True
    # Iterate in reverse until stable
    while changed:
        changed = False
        for n in reversed(graph.node):
            if any(o in keep for o in n.output):
                for i in n.input:
                    if i and i not in keep:
                        keep.add(i)
                        changed = True
    removed = []
    final_nodes = []
    for n in graph.node:
        if any(o in keep for o in n.output):
            final_nodes.append(n)
        else:
            removed.append(n.name)
    del graph.node[:]
    graph.node.extend(final_nodes)
    removed_inits = [i for i in graph.initializer
                     if i.name not in keep
                     and not any(i.name == inp for n in graph.node for inp in n.input)]
    if removed_inits:
        init_names = {i.name for i in removed_inits}
        graph.initializer[:] = [i for i in graph.initializer if i.name not in init_names]
    print(f'    Removed {len(removed)} nodes and {len(removed_inits)} initializers')
    if removed:
        print(f'    Examples: {removed[:8]}')

    # ---------- Validate and save ----------
    print('[6/6] Validating and saving...')
    onnx.checker.check_model(model, full_check=False)
    print('    onnx.checker passed')

    onnx.save(model, dst, save_as_external_data=False)
    size = os.path.getsize(dst)
    print(f'    Saved: {dst} ({size/1e6:.1f} MB)')
    print(f'    Node count: {len(graph.node)} (original {len(kept_nodes) - len(new_nodes) + len(to_remove)})')


if __name__ == '__main__':
    main()
