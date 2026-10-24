#!/usr/bin/env python3
"""Check that fuse_kv_layout.py preserved the decoder's behaviour.

Runs the original graph and the rewritten one on the same inputs, feeding each
the KV layout it expects, and compares every output. The KV caches are compared
after normalising back to `[S, 512]` so a layout mistake cannot hide.

Usage:
    python verify_kv_layout.py original.onnx rewritten.onnx
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort

LAYERS = 24
HEADS = 16
HEAD_DIM = 32
VOCAB = 1025


def kv_orig(rng, seq):
    return rng.standard_normal((seq, 1, 512)).astype(np.float32) * 0.25


def kv_flat(t, kind):
    """[1,16,32,S] (K) or [1,16,S,32] (V) back to [S, 512]."""
    t = np.asarray(t)[0]  # drop batch
    if kind == "k":
        t = np.transpose(t, (2, 0, 1))  # [S,16,32]
    else:
        t = np.transpose(t, (1, 0, 2))  # [S,16,32]
    return t.reshape(t.shape[0], HEADS * HEAD_DIM)


LOGITS = "/ar_predict_layer/MatMul_output_0"


def _with_logits(path: str) -> str:
    """Return a model that also exposes the pre-sampling logits as an output."""
    import os
    import tempfile

    from onnx import TensorProto, helper

    model = onnx.load(path)
    names = {o.name for o in model.graph.output}
    if LOGITS in names:
        return path
    model.graph.output.append(helper.make_tensor_value_info(LOGITS, TensorProto.FLOAT, [1, 1025]))
    out = os.path.join(tempfile.mkdtemp(prefix="kv_verify_"), "with_logits.onnx")
    onnx.save(model, out)
    return out


def main(a_path: str, b_path: str) -> int:
    a = ort.InferenceSession(_with_logits(a_path), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(_with_logits(b_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(11)
    failures = 0

    for seq in (8, 40, 96):
        for iy in (2, 17):
            feed_a, feed_b = {}, {}
            feed_a["iy"] = feed_b["iy"] = np.array([[i % VOCAB for i in range(iy)]], dtype=np.int64)
            emb = (rng.standard_normal((1, iy - 1, 512)) * 0.25).astype(np.float32)
            feed_a["iy_emb"] = feed_b["iy_emb"] = emb
            for l in range(LAYERS):
                past = kv_orig(rng, seq)
                feed_a[f"past_k_layer_{l}"] = past
                feed_a[f"past_v_layer_{l}"] = past
                # Same values, attention-native layout.
                feed_b[f"past_k_layer_{l}"] = past.reshape(seq, HEADS, HEAD_DIM).transpose(1, 2, 0)[None]
                feed_b[f"past_v_layer_{l}"] = past.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)[None]

            out_a = a.run(None, feed_a)
            out_b = b.run(None, feed_b)
            names = [o.name for o in a.get_outputs()]
            assert names == [o.name for o in b.get_outputs()], "output names differ"

            def get(outs, name):
                return outs[names.index(name)]

            # `y` comes out of RandomNormalLike, so it differs run to run even for
            # the same graph. Compare the deterministic outputs instead: the
            # pre-sampling logits, y_emb, and the KV caches.
            logits_a = np.asarray(get(out_a, LOGITS))
            logits_b = np.asarray(get(out_b, LOGITS))
            if logits_a.shape != logits_b.shape:
                print(f"  seq={seq} iy={iy}: logits shape {logits_a.shape} vs {logits_b.shape}")
                failures += 1
                continue
            denom = np.linalg.norm(logits_a) * np.linalg.norm(logits_b)
            cos = float(np.dot(logits_a.ravel(), logits_b.ravel()) / denom) if denom else 1.0
            max_logits = float(np.max(np.abs(logits_a - logits_b)))

            emb_a = get(out_a, "y_emb")
            emb_b = get(out_b, "y_emb")
            max_emb = float(np.max(np.abs(emb_a - emb_b))) if emb_a.shape == emb_b.shape else float("inf")

            worst_k = 0.0
            for l in range(LAYERS):
                for kind in ("k", "v"):
                    name = f"present_{kind}_layer_{l}"
                    va = np.asarray(get(out_a, name))
                    # The rewritten graph deliberately returns the cache in a
                    # different layout, so normalise it back before comparing.
                    vb = np.asarray(get(out_b, name))
                    if vb.shape != va.shape:
                        vb = kv_flat(vb, kind).reshape(va.shape)
                    worst_k = max(worst_k, float(np.max(np.abs(va - vb))))

            ok = cos >= 0.999999 and max_logits <= 1e-4 and max_emb <= 1e-4 and worst_k <= 1e-4
            if not ok:
                failures += 1
            print(
                f"  seq={seq:3d} iy={iy:2d}  logits cos={cos:.8f} maxdiff={max_logits:.2e}  "
                f"y_emb maxdiff={max_emb:.2e}  kv maxdiff={worst_k:.2e}  {'OK' if ok else 'FAIL'}"
            )

    print("PASS" if failures == 0 else f"FAIL ({failures} case(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
