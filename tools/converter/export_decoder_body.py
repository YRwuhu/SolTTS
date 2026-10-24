#!/usr/bin/env python3
"""Split the stage decoder into a GPU-friendly body and let the caller sample.

The exported decoder contains the whole sampling chain (GatherElements,
ScatterElements, TopK, Softmax, RandomNormalLike, ArgMax) inside the ONNX
graph. On a CPU provider that costs nothing extra. On the ggml plugin every
host-side op forces a *flush*: the ggml graph is submitted and every live
device value is read back to the host. The sampler touches the graph between
the transformer and the outputs, so one decode step becomes four flushes,
and the flush overhead (~20-30 ms) dwarfs the GPU compute (~5 ms).

This script cuts the graph at the logits and emits a body that contains only
transformer work:

    iy, iy_emb, past_*  ->  y_emb, logits, present_*

Sampling moves to the caller, which already reads the token back anyway, so
the body has no host-side op after the last device op and one decode step
becomes a single flush.

The body drops `y` (Concat(iy, token)) and `stop_condition_tensor`; both are
pure sampling bookkeeping the caller now owns.

Usage:
    python export_decoder_body.py in.onnx out.onnx
"""
import sys

import onnx
from onnx import TensorProto, helper

LOGITS = "/ar_predict_layer/MatMul_output_0"
BODY_OUTPUTS = ["y_emb", LOGITS]


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    graph = model.graph

    producer = {}
    for i, n in enumerate(graph.node):
        for o in n.output:
            producer[o] = i

    roots = BODY_OUTPUTS + [f"present_k_layer_{i}" for i in range(24)] + [f"present_v_layer_{i}" for i in range(24)]

    needed = set()
    stack = [r for r in roots if r in producer]
    while stack:
        name = stack.pop()
        i = producer[name]
        if i in needed:
            continue
        needed.add(i)
        for inp in graph.node[i].input:
            if inp in producer:
                stack.append(inp)

    nodes = [n for i, n in enumerate(graph.node) if i in needed]
    total = len(graph.node)

    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.ClearField("output")
    graph.output.append(helper.make_tensor_value_info("y_emb", TensorProto.FLOAT, [1, "y_emb_len", 512]))
    graph.output.append(helper.make_tensor_value_info(LOGITS, TensorProto.FLOAT, [1, 1025]))
    for i in range(24):
        for p in ("k", "v"):
            graph.output.append(
                helper.make_tensor_value_info(f"present_{p}_layer_{i}", TensorProto.FLOAT, ["present_len", 1, 512])
            )

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, dst)
    print(f"{src} -> {dst}: {len(nodes)}/{total} nodes")
    print(f"outputs: {len(graph.output)} (y_emb, logits, 48 present_*)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
