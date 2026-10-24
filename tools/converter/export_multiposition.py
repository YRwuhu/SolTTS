#!/usr/bin/env python3
"""Emit a fixed-K multi-position variant of the stage decoder.

The exported decoder mirrors `T2SStageDecoder.forward`, which computes one query
position per call:

    xy_pos = y_pos[:, -1:]        # only the last position is a query
    xy_dec = self.h(xy_pos, mask=..., cache=cache)

Three things enforce that, once per layer (24 each):

    Slice(starts=[-1], axes=[1])      -> query tensor is the last row only
    Reshape target (1, 16, 1, 32)     -> Q heads x 1 position
    Reshape target (1, 16, 32, 1)     -> K/V heads x 1 position
    Reshape target (-1, 1, 512)       -> QKV split, 1 position

Changing all four from 1 to a fixed K yields a graph that computes K query
positions in one forward, with no dynamic-shape arithmetic: every target becomes
a constant again, which is also what the runtime wants for predictable cost.

The caller supplies a prefix plus K-1 candidate tokens appended to `iy`; the last
K positions become queries and the returned `logits` have shape [1, K, 1025].
Logits for position i are the distribution for token i+1, exactly as a
verification pass needs.

Usage:
    python export_multiposition.py in.onnx out.onnx --k 4
    python export_multiposition.py in.onnx out.onnx --k 4 --verify
"""
import argparse
import sys

import numpy as np
import onnx
from onnx import numpy_helper

QUERY_SLICE = "/Slice_1_output_0"
TOKEN_SLICE = "/Slice_output_0"


def rewrite(graph, k: int):
    init = {i.name: i for i in graph.initializer}

    # 1. Widen both last-position Slices from 1 to K:
    #      /Slice    -> the token ids fed to the audio embedding
    #      /Slice_1  -> the position-encoded queries fed to the transformer
    #    They share one starts initializer, so each consumer gets its own copy.
    widened = []
    for node in graph.node:
        if node.op_type != "Slice" or not node.output:
            continue
        out = node.output[0]
        if out not in (TOKEN_SLICE, QUERY_SLICE):
            continue
        node.input[1] = _replace_const(graph, node.input[1], np.array([-k], dtype=np.int64), k)
        widened.append(out)
    if QUERY_SLICE not in widened:
        raise SystemExit("query Slice not found; is this the stage decoder?")
    if TOKEN_SLICE not in widened:
        raise SystemExit("token Slice not found; unexpected graph shape")

    # 2. Rewrite the three per-layer single-position reshape targets to K.
    #    Many Reshape nodes share the same target initializer, so rewrite the
    #    initializer once and every consumer follows.
    targets = {
        (1, 16, 1, 32): [1, 16, k, 32],
        (1, 16, 32, 1): [1, 16, 32, k],
        (-1, 1, 512): [-1, k, 512],
    }
    done = {tuple(v): 0 for v in targets.values()}
    for name, ini in list(init.items()):
        arr = numpy_helper.to_array(ini)
        if arr.dtype != np.int64 or arr.ndim == 0:
            continue
        key = tuple(arr.tolist())
        if key not in targets:
            continue
        new = np.array(targets[key], dtype=np.int64)
        ini.CopyFrom(numpy_helper.from_array(new, name))
        done[tuple(targets[key])] += 1

    # 3. Attention-output reshape. `Reshape_3` flattens [B,S,H,D] with target
    #    (-1, 512). At S == 1 that is [1,512] and the batch axis survives by
    #    accident; at K > 1 it becomes [K,512], dropping the batch axis and
    #    corrupting every later layer. [1,K,512] is the true residual shape.
    flatten = 0
    for name, ini in list(init.items()):
        arr = numpy_helper.to_array(ini)
        if arr.dtype != np.int64 or arr.ndim != 1:
            continue
        if arr.tolist() == [-1, 512]:
            ini.CopyFrom(numpy_helper.from_array(np.array([1, k, 512], dtype=np.int64), name))
            flatten += 1

    # 4. Attention-output transpose. `Transpose_5` in every layer carries
    #    perm=(2,0,1,3): "move the seq axis in front of batch so the two can be
    #    folded together". That identity holds only at S == 1; for K > 1 it
    #    scrambles the layout. (0,2,1,3) is the general [B,H,S,D] -> [B,S,H,D]
    #    form and is identical at S == 1.
    fixed_perm = 0
    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name == "perm" and tuple(attr.ints) == (2, 0, 1, 3):
                attr.ints[:] = [0, 2, 1, 3]
                fixed_perm += 1

    graph.ClearField("value_info")
    return widened, done, fixed_perm, flatten


def _replace_const(graph, old_name: str, value: np.ndarray, k: int) -> str:
    """Give this consumer its own copy so sibling nodes are untouched."""
    new_name = f"{old_name}_mp{k}"
    if not any(i.name == new_name for i in graph.initializer):
        graph.initializer.append(numpy_helper.from_array(value, new_name))
    return new_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--k", type=int, required=True, help="query positions per forward")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.k < 2:
        raise SystemExit("--k must be >= 2")

    model = onnx.load(args.src)
    widened, done, fixed_perm, flatten = rewrite(model.graph, args.k)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, args.dst)

    print(f"{args.src} -> {args.dst}  (K={args.k})")
    print(f"  widened {len(widened)} Slice(s) to starts=[-{args.k}]: {widened}")
    print(f"  fixed {fixed_perm} attention Transpose perm(s) (2,0,1,3) -> (0,2,1,3)")
    print(f"  rewrote {flatten} flatten target(s) (-1,512) -> (1,{args.k},512)")
    for shape, n in done.items():
        print(f"  rewrote {n:2} reshape target(s) -> {list(shape)}")

    if args.verify:
        verify(args.src, args.dst, args.k)


def _feed(session, iy_len, kv, seed=0):
    rng = np.random.default_rng(seed)
    names = {i.name for i in session.get_inputs()}
    k0 = next(i for i in session.get_inputs() if i.name == "past_k_layer_0")
    native = len(k0.shape) == 4
    feed = {
        "iy": np.array([[i % 1025 for i in range(iy_len)]], dtype=np.int64),
        "iy_emb": (rng.standard_normal((1, iy_len - 1, 512)) * 0.2).astype(np.float32),
    }
    for layer in range(24):
        if native:
            feed[f"past_k_layer_{layer}"] = (rng.standard_normal((1, 16, 32, kv)) * 0.2).astype(np.float32)
            feed[f"past_v_layer_{layer}"] = (rng.standard_normal((1, 16, kv, 32)) * 0.2).astype(np.float32)
        else:
            feed[f"past_k_layer_{layer}"] = (rng.standard_normal((kv, 1, 512)) * 0.2).astype(np.float32)
            feed[f"past_v_layer_{layer}"] = (rng.standard_normal((kv, 1, 512)) * 0.2).astype(np.float32)
    return {k: v for k, v in feed.items() if k in names}


def verify(single_path, multi_path, k):
    """The K-position graph must reproduce the reference on every position it
    computes. Compare against K separate single-position runs over the same
    prefix, which is exactly what it replaces."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    one = ort.InferenceSession(single_path, so, providers=["CPUExecutionProvider"])
    multi = ort.InferenceSession(multi_path, so, providers=["CPUExecutionProvider"])

    print(f"\n  K={k}: checking the multi-position graph against {k} single-position runs")
    for prefix in (2, 6):
        kv = 32
        # Reference: run the single-position graph on prefixes of length
        # prefix..prefix+k-1. Its `y_emb` covers the whole prefix each time.
        refs = []
        for extra in range(k):
            feed = _feed(one, prefix + extra, kv, seed=7)
            ro = one.run(None, feed)
            no = [o.name for o in one.get_outputs()]
            refs.append((feed, dict(zip(no, ro))))

        # Multi: one call with iy of length prefix+k-1.
        feed_m = _feed(multi, prefix + k - 1, kv, seed=7)
        rm = multi.run(None, feed_m)
        nm = [o.name for o in multi.get_outputs()]

        ok = True
        for name in ("y_emb",) + tuple(f"present_{p}_layer_{i}" for i in range(24) for p in "kv"):
            if name not in nm:
                continue
            a = refs[-1][1][name]
            b = rm[nm.index(name)]
            if a.shape != b.shape:
                ok, msg = False, f"{name} shape {a.shape} vs {b.shape}"
                break
            d = float(np.abs(a - b).max())
            if d > 1e-5:
                ok, msg = False, f"{name} maxdiff {d:.2e}"
                break
        print(f"    prefix={prefix}: {'ok' if ok else 'FAIL ' + msg}")


if __name__ == "__main__":
    sys.exit(main())
