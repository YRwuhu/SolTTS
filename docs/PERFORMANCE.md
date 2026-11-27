# Performance notes: what helped, what didn't

Every number here was measured on a Ryzen 5 5500 (6C/12T) + Radeon RX 5500 XT.
Numbers are medians, not best-case picks. Negative results are kept on purpose, so
nobody re-runs them.

For the decoder deep-dive — kernel profile, GPU-offload analysis, and the reason
thread count makes no difference — see [DECODER_BASELINE.md](DECODER_BASELINE.md).

## What helped

| Change | Effect |
|---|---|
| Vocoder on GPU (ggml/Vulkan) | vits kernels ~5×; ~25% end-to-end non-streaming, ~2× streaming |
| Parallel sentence decoding (`ParallelSentenceDegree: 3`) | 3 sentences: ~3.5 s → ~2.4 s |
| Streaming cross-sentence overlap | Next sentence's decoder starts while the previous sentence's vocoder runs; ~10% off streaming total |
| `VocoderRetriggerFactor: 4.0` | Vocoder calls 56 → 42 per run; vocoder time −29%; streaming total ~−10% |
| BERT feature cache | Saves a ~280–390 ms forward pass on repeated text |
| Rust native audio (`sol_audio`) | Audio pipeline p99 27 µs → 4 µs; ring buffer 0.70 µs → 0.05 µs per chunk |
| `GraphOptimizationLevel: 2` | ORT fuses the decoder 1300 → 696 nodes (`DynamicQuantizeMatMul`, `SkipLayerNormalization`, `FusedMatMul`); token sequence identical over a full 95-step rollout |
| Native KV cache layout (`UseNativeKvLayout`, on by default) | Decoder layout ops 369 → 273 nodes; **1.07–1.13×** per decode step. Logits and rendered audio are **bit-identical**. Requires running `tools/converter/fuse_kv_layout.py` once per resource tree; falls back to the shipped models if the generated files are absent |
| Native KV layout generated for the **ATRI** models (v1.4.0) | Measured on this repo's shipped decoder: **1.16×** at (iy=48, kv=96), **1.23×** at (96,160), **1.36×** at (160,320). `verify_kv_layout.py` maxdiff **0.00e+00** on all six shape combos; `y_emb` and all 48 KV caches bit-identical. `fuse_kv_layout.py` also learned the simplified `Reshape→Transpose` K chain (no `Unsqueeze`) that the ATRI export uses |
| `VocoderRetriggerFactor` 1.5 → 4.0 (v1.4.0 default) | Streaming: 3-sentence total **3343 → 2539 ms (−24%)**, single-sentence TTFB **855 → 592 ms**, vocoder calls **48 → 34**, vocoder time **9823 → 6599 ms** |
| Output-index cache in the decode loop (v1.4.0) | Removes ~50 `First(r => r.Name == …)` string scans **per decode step** (~2500 per token sequence) by resolving output positions once per session and indexing directly |
| Single-flatten BERT features (v1.4.0) | Drops two per-request copies: `float[,] → 1-D` for the reference BERT and `1-D → 2-D → 1-D` for each sentence. Reference BERT is flattened once and cached |
| Emotion `(emotion, α) → (ge, ge_advanced)` cache (v1.4.0) | Switching back to a previously used emotion/α skips the `prompt_encoder` forward pass entirely; `emotion_bench` emotion-switch mean **1925 → 2019 ms** with 0 empty outputs (previously ~2/12 calls degraded to 0.00 s) |
| Semantic-signal wakeup instead of 5 ms polling (v1.4.0) | The streaming vocoder consumer no longer spins on `Task.Delay(5)`; removes up to 5 ms of jitter per chunk |
| Honest retry accounting in non-streaming T2S (v1.4.0) | `RunT2S` now honours `MaxRetries` (was hard-coded 10), separates first-token-EOS sampling noise (cap 12) from the formal budget, and keeps the longest attempt so a late bad draw cannot collapse a sentence to 0.00 s of audio |

## What did not help

| Attempt | Result |
|---|---|
| INT8-quantizing the decoders | Already INT8 in the shipped graphs (`MatMulInteger` + `DynamicQuantizeLinear`) |
| BERT on GPU | **1.7× slower** (294 → 489 ms). The plugin refuses the graph ("22 of 1089 nodes use ops this provider lacks: Gelu") |
| ggml `partial` offload of the decoder | 14.63 ms/step vs 15.71 ms on CPU — 7%, not worth the memory |
| Merging decode steps into one ONNX `Loop` | 1.02× / 0.98× — inside noise |
| Shortening the decoder's `iy` input | 109 → 2 tokens changes nothing (15.36 → 15.25 ms/step) |
| Incremental vocoding (decode only new tokens) | Not possible: the VITS `enc_p` attention and flow depend on the whole sequence |
| `QLinearMatMul` fusion of the quantized matmuls | Rejected: its output is int8, but the int32 accumulators span ±20000, so results saturate |
| Hand-written QKV `Split` fusion | Redundant: at GOL 2 ORT performs the same fusion, so both models optimize to 696 nodes. Kept as a script for GOL 0/1 only |
| CoreML on macOS | Slower than CPU for these models |
| Whole-graph ggml offload of the decoder | Reachable (rewrite 2 `ArgMax` → `TopK`, then the plugin claims 1301/1301 nodes and 147.6 MiB of weights), but **0.2–0.3×**: one step is 25–43 ms vs 8–12 ms on CPU |
| ORT thread / arena / spinning sweep | No variant clears the noise floor; `IntraOpNumThreads` 8 and 12 are actively worse |
| `weights=q8_0` on the offloaded decoder | Helps the flush (13.1 → 9.4 ms at kv=32) but still loses to CPU |
| Repairing `roberta_noloop.py` for the shipped model (v1.4.0) | **Not applicable**: the shipped `resources/shared/v1_extra/bert/roberta.onnx` already has no `Loop`/`SplitToSequence` nodes (the vectorized `CumSum`/`Where`/`Slice` form is present), so the script's preconditions fail by design. CPU inference measures **93.7 ms** median |
| ggml/Vulkan for BERT (re-measured v1.4.0) | Still **0.32×** (294.0 ms vs 93.7 ms CPU). Plugin reports "claiming nothing": 22 `Gelu` + 66 `CastLike` + 22 `Sum` nodes unsupported. Keep `UseBertGgml: false` |
| Inter-op threads 0 → 1 in `config.yaml` (v1.4.0) | Aligns the shipped config with the measured default (`1` is ~4% faster than `0` on the native-KV decoder); not a new experiment, just config hygiene |

## The honest summary

The decoder step is bound by **node count and layout**, not by the 72 MB weight set.
37% is `DynamicQuantizeMatMul`, 39% is `Transpose`/`Concat`/`Reshape`/`Unsqueeze` that
exist only to satisfy the exporter, and threads change nothing because there is no
bandwidth to saturate. Moving the graph to the GPU moves the same per-node work to a
place where the plugin must also rebuild the graph each run, so it is strictly slower.

The useful directions are folding the 48 per-layer KV `Reshape`+`Transpose` chains and
reducing the quantized-matmul count in the exporter — see
[DECODER_BASELINE.md](DECODER_BASELINE.md).