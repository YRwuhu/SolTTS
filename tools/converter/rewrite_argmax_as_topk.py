#!/usr/bin/env python3
"""Rewrite ArgMax into TopK(k=1) + Squeeze so the ggml plugin can claim the whole decoder.

Why: the ggml plugin implements TopK on its host interpreter but has no ArgMax,
and its GetCapability is all-or-nothing. Two ArgMax nodes in the probability
sampler were the only reason the 1300-node stage decoder was rejected outright
("2 of 1300 nodes use ops this provider lacks: ArgMax x2"). The first stage
decoder fails the same way ("ArgMax x1").

The rewrite is exact. For `axis` with keepdims=0:

    ArgMax(x, axis) == Squeeze(TopK(x, k=1, axis).indices, axes=[axis])

TopK returns the top-k values in descending order with ties broken by the lower
index, which is what ArgMax does when select_last_index=0. For keepdims=1 the
TopK indices already have the input rank with the axis reduced to 1, so no
Squeeze is needed.

The TopK itself runs in the provider's host interpreter, i.e. one readback of
the logits per decode step. That readback already happens for `y`, so it costs
nothing extra.

Usage:
    python rewrite_argmax_as_topk.py in.onnx out.onnx
"""
import sys

import onnx
from onnx import helper, numpy_helper, TensorProto


def rewrite(path_in: str, path_out: str) -> int:
    model = onnx.load(path_in)
    graph = model.graph

    new_nodes = []
    new_inits = []
    counter = 0
    rewrites = 0

    for node in graph.node:
        if node.op_type != "ArgMax":
            new_nodes.append(node)
            continue

        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        axis = int(attrs.get("axis", 0))
        keepdims = int(attrs.get("keepdims", 1))
        if int(attrs.get("select_last_index", 0)) != 0:
            raise SystemExit(f"{node.name}: select_last_index=1 is not equivalent to TopK")

        tag = f"argmax_as_topk_{counter}"
        counter += 1

        k_name = f"{tag}_k"
        new_inits.append(numpy_helper.from_array(_i64([1]), k_name))

        values = f"{tag}_values"
        indices = f"{tag}_indices"
        new_nodes.append(
            helper.make_node(
                "TopK",
                [node.input[0], k_name],
                [values, indices],
                name=f"{tag}_topk",
                axis=axis,
                largest=1,
                sorted=1,
            )
        )

        if keepdims:
            new_nodes.append(
                helper.make_node("Identity", [indices], [node.output[0]], name=f"{tag}_identity")
            )
        else:
            axes_name = f"{tag}_axes"
            new_inits.append(numpy_helper.from_array(_i64([axis]), axes_name))
            new_nodes.append(
                helper.make_node(
                    "Squeeze", [indices, axes_name], [node.output[0]], name=f"{tag}_squeeze"
                )
            )
        rewrites += 1

    if rewrites == 0:
        print(f"{path_in}: no ArgMax, nothing to do")
        return 0

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)

    # TopK's second output is int64, matching ArgMax's output type.
    for out in graph.output:
        pass

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path_out)
    print(f"{path_in} -> {path_out}: rewrote {rewrites} ArgMax node(s)")
    return rewrites


def _i64(values):
    import numpy as np

    return np.asarray(values, dtype=np.int64)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    rewrite(sys.argv[1], sys.argv[2])
