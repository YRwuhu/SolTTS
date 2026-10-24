#!/usr/bin/env python3
"""Emit an all-position variant of the stage decoder for multi-token verification.

The shipped decoder computes logits for one position only: a `Gather(axis=1, -1)`
sits between the final LayerNorm and `ar_predict_layer`, so every downstream node
(sampler included) sees a single token.

Removing that one Gather lets `ar_predict_layer` run on all `iy` positions and
produces `logits [1, iy_len, 1025]` — exactly what a lookahead / multi-token
verification pass needs. Nothing else changes: the transformer body already
processes the whole sequence; only the tail was restricted to the last row.

The sampler chain (GatherElements / TopK / RandomNormalLike / ArgMax /
ScatterElements) is removed so sampling can be done reproducibly by the caller.

    iy, iy_emb, past_*  ->  logits [1,S,1025], y_emb, present_*

Usage:
    python export_multiposition_body.py in.onnx out.onnx
"""
import sys

import onnx
from onnx import TensorProto, helper

LAST_POS_GATHER = "/Gather_output_0"
LOGITS = "/ar_predict_layer/MatMul_output_0"
FULL_SEQ = "/transformer_encoder/layers.23/norm2/LayerNormalization_output_0"


def build(src: str, dst: str) -> None:
    model = onnx.load(src, load_external_data=False)
    graph = model.graph

    # 1. Drop the last-position Gather so the predict layer sees [1, S, 512].
    nodes = []
    dropped = 0
    for n in graph.node:
        if n.output and n.output[0] == LAST_POS_GATHER:
            dropped += 1
            continue
        nodes.append(n)

    # 2. Rewire every consumer of the dropped tensor to the full-sequence output.
    rewired = 0
    for n in nodes:
        for i, inp in enumerate(n.input):
            if inp == LAST_POS_GATHER:
                n.input[i] = FULL_SEQ
                rewired += 1

    # 3. Keep only the nodes that feed logits, y_emb and the 48 caches,
    #    which drops the entire sampling tail.
    prod = {o: idx for idx, n in enumerate(nodes) for o in n.output}
    roots = [LOGITS, "y_emb"] + [f"present_k_layer_{i}" for i in range(24)] + \
            [f"present_v_layer_{i}" for i in range(24)]
    needed = set()
    stack = [r for r in roots if r in prod]
    while stack:
        name = stack.pop()
        idx = prod[name]
        if idx in needed:
            continue
        needed.add(idx)
        for inp in nodes[idx].input:
            if inp in prod:
                stack.append(inp)
    kept = [n for idx, n in enumerate(nodes) if idx in needed]

    graph.ClearField("node")
    graph.node.extend(kept)
    graph.ClearField("output")
    graph.output.append(helper.make_tensor_value_info(LOGITS, TensorProto.FLOAT, [1, "iy_len", 1025]))
    graph.output.append(helper.make_tensor_value_info("y_emb", TensorProto.FLOAT, [1, "y_emb_len", 512]))
    for i in range(24):
        for p in ("k", "v"):
            graph.output.append(
                helper.make_tensor_value_info(f"present_{p}_layer_{i}", TensorProto.FLOAT, ["present_len", 1, 512]))

    # Every downstream value_info carries the old single-position rank. Rather than
    # guess which entries are stale, drop them all: value_info is annotation only,
    # and the checker re-infers shapes from the nodes.
    graph.ClearField("value_info")

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, dst)
    print(f"{src} -> {dst}")
    print(f"  dropped {dropped} last-position Gather, rewired {rewired} consumer(s)")
    print(f"  nodes {len(graph.node)}/{len(nodes)} after sampler removal")
    print(f"  outputs: logits[1,S,1025] + y_emb + 48 present_*")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    build(sys.argv[1], sys.argv[2])
