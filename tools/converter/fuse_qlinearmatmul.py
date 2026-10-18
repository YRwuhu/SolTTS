#!/usr/bin/env python3
"""
[REJECTED - kept for the record]

Attempt: fuse `MatMulInteger + Cast + Mul(scale)` into a single `QLinearMatMul`.

Why it does not work
--------------------
`QLinearMatMul` takes 8 inputs and produces an **int8/uint8** output. The stock
dynamic-quantization export instead produces an **int32** accumulator that is only
scaled to float afterwards. On real decoder activations that int32 value spans
roughly +/-20000, so forcing it through an int8 output saturates almost everything
(measured: max abs error 2.67 on outputs whose own range is about +/-2.7).

Making it work would require re-quantizing with a per-layer output scale chosen from
calibration data, which changes numerics rather than preserving them. Not pursued.

The structural rewrite that *does* work is fuse_qkv_split.py, which is bit-exact.

把动态量化解码器中的 `MatMulInteger + Cast + Mul(scale)` 三节点链
替换为单节点 `QLinearMatMul`，减少中间张量与算子调度。

原理
----
ORT 的动态量化导出会把每个线性层展开成：

    a_quant, a_scale, a_zero = DynamicQuantizeLinear(a)
    b_quant, b_scale, b_zero = <静态权重>
    y_int32 = MatMulInteger(a_quant, b_quant, a_zero, b_zero)
    y_float = Cast(y_int32, float) * (a_scale * b_scale)

`QLinearMatMul` 用一次调用完成同样的数学：输入是 (a_quant, a_scale, a_zero,
b_quant, b_scale, b_zero)，输出直接是 float。省掉 int32 中间张量的物化、
一次 Cast、一次 Mul，以及对应的内存分配。

注意
----
- 这是**数值等价重写**，不是重新量化。权重和激活的 scale/zero_point 全部沿用原图，
  因此输出应当逐位或极接近（QLinearMatMul 内部同样是 int32 累加）。
- 只处理模式完全匹配的链；任何不匹配的保持原样。
- 输出模型仍需通过 onnx.checker 和逐元素数值对比。

用法
----
  python tools/converter/fuse_qlinearmatmul.py \
      --input  resources/models_v1/chenxing/t2s_stage_decoder.onnx \
      --output resources/models_v1/chenxing/t2s_stage_decoder_qmm.onnx
"""

import argparse
import sys
from collections import defaultdict

import onnx
from onnx import helper, TensorProto


def build_consumer_map(graph):
    consumers = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    return consumers


def fuse(graph) -> tuple[int, int]:
    """原地重写 graph。返回 (替换数, 跳过数)。"""
    producers = {out: n for n in graph.node for out in n.output}
    consumers = build_consumer_map(graph)
    nodes_by_name = {n.name: n for n in graph.node}

    replaced = 0
    skipped = 0
    remove = set()
    additions = []

    for mmi in [n for n in graph.node if n.op_type == "MatMulInteger"]:
        # 模式 1：a 侧来自 DynamicQuantizeLinear（激活），b 侧是静态量化权重
        if len(mmi.input) != 4:
            skipped += 1
            continue
        a_q, b_q, a_zp, b_zp = mmi.input

        dq = producers.get(a_q)
        if dq is None or dq.op_type != "DynamicQuantizeLinear":
            skipped += 1
            continue
        if len(dq.output) < 3:
            skipped += 1
            continue
        a_scale = dq.output[1]
        # dq.output[2] 是 a 的 zero point，应当与 mmi 的第 3 个输入一致
        if dq.output[2] != a_zp:
            skipped += 1
            continue

        # b 的 scale / zero point 必须是 initializer（静态权重）
        if b_q not in {i.name for i in graph.initializer}:
            skipped += 1
            continue
        b_scale_init = next((i for i in graph.initializer if i.name == b_zp), None)
        if b_scale_init is None:
            # 有时 b_zero_point 是 initializer 而 scale 另有其名；按命名约定找
            b_scale_name = b_q.replace("_quantized", "_scale")
            b_scale_init = next((i for i in graph.initializer if i.name == b_scale_name), None)
            if b_scale_init is None:
                skipped += 1
                continue
            b_scale = b_scale_name
        else:
            # b_zp 本身是 zero point；scale 用命名约定
            b_scale = b_q.replace("_quantized", "_scale")
            if not any(i.name == b_scale for i in graph.initializer):
                skipped += 1
                continue

        # mmi 的下游必须是 Cast -> Mul
        mmi_outs = mmi.output
        if len(mmi_outs) != 1:
            skipped += 1
            continue
        cast = None
        for c in consumers.get(mmi_outs[0], []):
            if c.op_type == "Cast":
                cast = c
                break
        if cast is None or len(cast.output) != 1:
            skipped += 1
            continue
        mul = None
        for c in consumers.get(cast.output[0], []):
            if c.op_type == "Mul":
                mul = c
                break
        if mul is None or len(mul.output) != 1:
            skipped += 1
            continue

        # Mul 的另一个输入应是 a_scale * b_scale（可能是中间 Mul 的输出）
        other = [x for x in mul.input if x != cast.output[0]]
        if len(other) != 1:
            skipped += 1
            continue
        scale_chain = other[0]
        scale_mul = producers.get(scale_chain)
        if scale_mul is not None and scale_mul.op_type == "Mul":
            # scale_mul 的两个输入应是 a_scale 与 b_scale
            if set(scale_mul.input) != {a_scale, b_scale}:
                skipped += 1
                continue
            remove.add(scale_mul.name)
        elif scale_chain not in (a_scale, b_scale):
            skipped += 1
            continue

        out_name = mul.output[0]
        qmm_out = out_name  # 复用原输出名，避免下游改名

        qmm = helper.make_node(
            "QLinearMatMul",
            inputs=[a_q, a_scale, a_zp, b_q, b_scale, b_zp],
            outputs=[qmm_out],
            name=mmi.name + "_qmm",
        )

        remove.update({mmi.name, cast.name, mul.name})
        additions.append(qmm)
        replaced += 1

    if replaced == 0:
        return 0, skipped

    kept = [n for n in graph.node if n.name not in remove]
    kept.extend(additions)

    # 新节点被追加到末尾，但下游可能依赖它们；按依赖关系重新拓扑排序。
    ordered = topo_sort(kept)
    del graph.node[:]
    graph.node.extend(ordered)
    return replaced, skipped


def topo_sort(nodes):
    """对节点列表做稳定拓扑排序；输入来自 initializer/graph input 的视为已就绪。"""
    by_output = {out: n for n in nodes for out in n.output}
    ready = set()
    visited = set()
    ordered = []

    def visit(node, stack):
        key = id(node)
        if key in visited:
            return
        if key in stack:
            raise ValueError(f"cycle detected at node {node.name}")
        stack.add(key)
        for inp in node.input:
            producer = by_output.get(inp)
            if producer is not None:
                visit(producer, stack)
        stack.discard(key)
        visited.add(key)
        ordered.append(node)

    for n in nodes:
        visit(n, set())
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"[qmm] loading {args.input}")
    model = onnx.load(args.input, load_external_data=False)
    before = len(model.graph.node)

    replaced, skipped = fuse(model.graph)
    print(f"[qmm] nodes {before} -> {len(model.graph.node)}  (fused {replaced}, skipped {skipped})")

    if replaced == 0:
        print("[qmm] nothing fused; refusing to write an identical model")
        return 1

    print("[qmm] checking model ...")
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as e:
        print(f"[qmm] checker FAILED: {e}")
        return 1
    print("[qmm] checker OK")

    onnx.save(model, args.output, save_as_external_data=False)
    print(f"[qmm] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
