#!/usr/bin/env python3
"""
把 attention 层里 Q/K/V 的三次 `Slice` 合并为一次 `Split`。

背景
----
解码器每层这样切 Q/K/V：

    q = Slice(proj, starts=0,   ends=512,  axes=2)
    k = Slice(proj, starts=512, ends=1024, axes=2)
    v = Slice(proj, starts=1024,ends=1536, axes=2)

三次 Slice 各自物化一份张量并做一次边界计算。`Split` 一次切三段，
少两次算子调度、少两次 shape 推断，且下游能更早拿到全部三份。

这是纯结构重写：数值完全等价（同一条轴上的连续切分），不涉及量化或精度。

用法
----
  python tools/converter/fuse_qkv_split.py \
      --input  resources/models_v1/chenxing/t2s_stage_decoder.onnx \
      --output resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def const_value(graph, name):
    """取 initializer 的标量/小数组值；不是 initializer 则返回 None。"""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def topo_sort(nodes):
    by_output = {out: n for n in nodes for out in n.output}
    visited, ordered = set(), []

    def visit(node, stack):
        key = id(node)
        if key in visited:
            return
        if key in stack:
            raise ValueError(f"cycle at {node.name}")
        stack.add(key)
        for inp in node.input:
            p = by_output.get(inp)
            if p is not None:
                visit(p, stack)
        stack.discard(key)
        visited.add(key)
        ordered.append(node)

    for n in nodes:
        visit(n, set())
    return ordered


def fuse(graph):
    """寻找同一输入上的连续 3 段 Slice，替换为 Split。返回 (替换组数, 跳过数)。"""
    producers = {out: n for n in graph.node for out in n.output}
    consumers = defaultdict(list)
    for n in graph.node:
        for inp in n.input:
            consumers[inp].append(n)

    # 按 (数据输入, axis) 分组候选 Slice
    groups = defaultdict(list)
    skipped = 0
    for n in graph.node:
        if n.op_type != "Slice" or len(n.input) < 3:
            skipped += 1
            continue
        data = n.input[0]
        starts = const_value(graph, n.input[1])
        ends = const_value(graph, n.input[2])
        axes = const_value(graph, n.input[3]) if len(n.input) > 3 else np.array([0])
        if starts is None or ends is None or axes is None:
            skipped += 1
            continue
        if starts.size != 1 or ends.size != 1 or axes.size != 1:
            skipped += 1
            continue
        groups[(data, int(axes.reshape(-1)[0]))].append((int(starts.reshape(-1)[0]),
                                                         int(ends.reshape(-1)[0]), n))

    replacements = 0
    remove = set()
    additions = []

    for (data, axis), items in groups.items():
        if len(items) != 3:
            continue
        items.sort(key=lambda x: x[0])
        # 必须是连续且等长的三段
        s0, e0, _ = items[0]
        s1, e1, _ = items[1]
        s2, e2, _ = items[2]
        if s0 != 0 or s1 != e0 or s2 != e1:
            continue
        if not (e0 - s0 == e1 - s1 == e2 - s2):
            continue
        # 每个 Slice 必须只有这一个消费者，且输出不被图输出直接引用
        ok = True
        for _, _, node in items:
            if len(consumers.get(node.output[0], [])) != 1:
                ok = False
                break
        if not ok:
            continue

        split_sizes = [e0 - s0, e1 - s1, e2 - s2]
        base = items[0][2].name
        split_node = helper.make_node(
            "Split",
            inputs=[data],
            outputs=[items[0][2].output[0], items[1][2].output[0], items[2][2].output[0]],
            name=base + "_qkv_split",
            axis=axis,
        )
        # Split 的 sizes 作为第二个输入（opset 13+）
        sizes_name = base + "_qkv_sizes"
        graph.initializer.append(
            numpy_helper.from_array(np.array(split_sizes, dtype=np.int64), sizes_name)
        )
        split_node.input.append(sizes_name)

        remove.update(node.name for _, _, node in items)
        additions.append(split_node)
        replacements += 1

    if replacements == 0:
        return 0, skipped

    kept = [n for n in graph.node if n.name not in remove]
    kept.extend(additions)
    ordered = topo_sort(kept)
    del graph.node[:]
    graph.node.extend(ordered)
    return replacements, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"[qkv] loading {args.input}")
    model = onnx.load(args.input, load_external_data=False)
    before = len(model.graph.node)

    replaced, skipped = fuse(model.graph)
    print(f"[qkv] nodes {before} -> {len(model.graph.node)}  (split groups {replaced}, slices skipped {skipped})")

    if replaced == 0:
        print("[qkv] nothing fused")
        return 1

    print("[qkv] checking ...")
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as e:
        print(f"[qkv] checker FAILED: {e}")
        return 1
    print("[qkv] checker OK")
    onnx.save(model, args.output, save_as_external_data=False)
    print(f"[qkv] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
