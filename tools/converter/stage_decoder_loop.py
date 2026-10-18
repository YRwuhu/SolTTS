#!/usr/bin/env python3
"""
Wraps t2s_stage_decoder.onnx into a Loop model that completes the entire autoregressive decode in a single Run.

Principle
---------
The original graph is one forward pass: inputs (iy, iy_emb, 48 past KV), outputs (y, y_emb, stop, 48 present KV).
The C# side calls Run once per generated token, feeding the present KV back as the next past — about 95 calls total.

This script places all original nodes unchanged inside an ONNX `Loop` body subgraph:
  - carried state = the original 50 inputs
  - next state = the original 50 outputs (y/iy_emb/present KV)
  - loop condition = NOT(stop_condition_tensor)
      (the original output semantics are "should stop", while the ONNX Loop body's first output means "continue")

The 410 initializers stay in the outer graph and are implicitly referenced by the body from the outer scope (not copied, avoiding ~75 MB of redundancy).

Output
------
t2s_stage_decoder_loop.onnx with the interface:
  inputs:  max_steps (int64 scalar), cond_init (bool scalar), and the original 50 inputs
  outputs: 50 final states (y, y_emb, 48 KV)

Usage
-----
  python tools/converter/stage_decoder_loop.py \
      --input  resources/models_v1/chenxing/t2s_stage_decoder.onnx \
      --output resources/models_v1/chenxing/t2s_stage_decoder_loop.onnx
"""

import argparse
import os
import sys

import onnx
from onnx import helper, TensorProto


def build_loop_model(model: onnx.ModelProto, max_steps: int = 500) -> onnx.ModelProto:
    g = model.graph
    inputs = list(g.input)
    outputs = list(g.output)

    if len(inputs) != 50:
        raise ValueError(f"expected 50 inputs, got {len(inputs)}: {[i.name for i in inputs]}")
    if len(outputs) != 51:
        raise ValueError(f"expected 51 outputs, got {len(outputs)}: {[o.name for o in outputs]}")

    if inputs[0].name != "iy" or inputs[1].name != "iy_emb":
        raise ValueError("unexpected input order; expected iy, iy_emb first")
    if outputs[0].name != "y" or outputs[1].name != "y_emb":
        raise ValueError("unexpected output order; expected y, y_emb first")
    if outputs[2].name != "stop_condition_tensor":
        raise ValueError("expected stop_condition_tensor at output index 2")

    # body needs: iter (int64 scalar) + cond (bool scalar) + 50 carried values
    body_inputs = [
        helper.make_tensor_value_info("iter_num", TensorProto.INT64, []),
        helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
    ]
    body_inputs += inputs

    # State mapping: next-step value for each carried input.
    # iy -> y, iy_emb -> y_emb, past_{k,v}_layer_N -> present_{k,v}_layer_N
    # stop_condition_tensor is not state; it only generates the loop condition.
    state_map = {"iy": "y", "iy_emb": "y_emb"}
    for layer in range(24):
        state_map[f"past_k_layer_{layer}"] = f"present_k_layer_{layer}"
        state_map[f"past_v_layer_{layer}"] = f"present_v_layer_{layer}"

    missing = [i.name for i in inputs if i.name not in state_map]
    if missing:
        raise ValueError(f"unmapped inputs: {missing}")
    state_outputs = [state_map[i.name] for i in inputs]
    out_by_name = {o.name: o for o in outputs}
    for name in state_outputs:
        if name not in out_by_name:
            raise ValueError(f"missing state output: {name}")

    # body outputs = 1 loop condition + 50 new states (same order as carried inputs)
    body_outputs = [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])]
    for name in state_outputs:
        o = out_by_name[name]
        body_outputs.append(
            helper.make_tensor_value_info(
                o.name, o.type.tensor_type.elem_type,
                [d.dim_value if d.dim_value else d.dim_param
                 for d in o.type.tensor_type.shape.dim],
            )
        )

    not_node = helper.make_node(
        "Not",
        inputs=["stop_condition_tensor"],
        outputs=["cond_out"],
        name="/LoopBody/NotStop",
    )

    # Keep all original nodes, preserving order; append Not last (it depends only on original outputs)
    body_nodes = list(g.node) + [not_node]

    body_graph = helper.make_graph(
        nodes=body_nodes,
        name="stage_decoder_body",
        inputs=body_inputs,
        outputs=body_outputs,
        initializer=[],  # Left empty: provided by the outer scope
        value_info=list(g.value_info),
    )

    # Outer Loop: trip count + initial condition + 50 initial states.
    # trip/cond must be real graph inputs (not baked-in initializers),
    # otherwise the streaming path cannot batch calls by chunk size.
    trip = helper.make_tensor(
        "loop_max_steps_default", TensorProto.INT64, [], [max_steps]
    )
    cond_init = helper.make_tensor(
        "loop_cond_init_default", TensorProto.BOOL, [], [True]
    )

    loop_inputs = ["loop_max_steps", "loop_cond_init"] + [i.name for i in inputs]
    # ONNX Loop semantics: the body's output 0 is the loop condition (not a Loop output);
    # the remaining 50 are the new state values corresponding one-to-one with the carried inputs.
    loop_outputs = [name + "_out" for name in state_outputs]

    loop_node = helper.make_node(
        "Loop",
        inputs=loop_inputs,
        outputs=loop_outputs,
        body=body_graph,
        name="/StageDecoderLoop",
    )

    # State types/shapes come from the body's actual outputs (y/y_emb/present_*),
    # not the input declarations — for example, iy[1,iy_length] produces [1,iy_length+1].
    outer_outputs = []
    for name in state_outputs:
        o = out_by_name[name]
        outer_outputs.append(
            helper.make_tensor_value_info(
                o.name + "_out",
                o.type.tensor_type.elem_type,
                [d.dim_value if d.dim_value else d.dim_param
                 for d in o.type.tensor_type.shape.dim],
            )
        )

    new_graph = helper.make_graph(
        nodes=[loop_node],
        name=model.graph.name + "_loop",
        inputs=[helper.make_tensor_value_info("loop_max_steps", TensorProto.INT64, []),
                helper.make_tensor_value_info("loop_cond_init", TensorProto.BOOL, [])],
        outputs=outer_outputs,
        initializer=list(g.initializer),
        value_info=[],
    )

    # Keep the original loop inputs (carried state) as outer inputs
    new_graph.input.extend(inputs)

    new_model = helper.make_model(
        new_graph,
        producer_name="sol_stage_decoder_loop",
        opset_imports=model.opset_import,
    )
    new_model.ir_version = model.ir_version

    return new_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    print(f"[loop] loading {args.input}")
    model = onnx.load(args.input, load_external_data=False)
    print(f"[loop] nodes={len(model.graph.node)} initializers={len(model.graph.initializer)}")

    new_model = build_loop_model(model, args.max_steps)

    print("[loop] running checker (full_check=True) ...")
    try:
        onnx.checker.check_model(new_model, full_check=True)
        print("[loop] checker OK")
    except Exception as e:
        print(f"[loop] checker FAILED: {e}")
        return 1

    onnx.save(new_model, args.output, save_as_external_data=False)
    size = os.path.getsize(args.output)
    print(f"[loop] wrote {args.output} ({size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
