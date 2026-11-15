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

## The honest summary

The decoder step is bound by **node count and layout**, not by the 72 MB weight set.
37% is `DynamicQuantizeMatMul`, 39% is `Transpose`/`Concat`/`Reshape`/`Unsqueeze` that
exist only to satisfy the exporter, and threads change nothing because there is no
bandwidth to saturate. Moving the graph to the GPU moves the same per-node work to a
place where the plugin must also rebuild the graph each run, so it is strictly slower.

The useful directions are folding the 48 per-layer KV `Reshape`+`Transpose` chains and
reducing the quantized-matmul count in the exporter — see
[DECODER_BASELINE.md](DECODER_BASELINE.md).