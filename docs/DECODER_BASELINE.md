# Decoder baseline: what limits a decode step, and what actually helped

This document records the reproducible baseline for the V1 semantic decoder, the
measurements behind the "GPU should be 3–5× faster" hypothesis in the v2 plan, and
what that investigation produced. Every number here comes from a tool in this
repository; the commands to reproduce each table are included.

Hardware: Ryzen 5 5500 (6C/12T), Radeon RX 5500 XT 4 GB, Vulkan 1.3, Windows 10.0.26100.
Software: .NET 10.0.401, ONNX Runtime 1.30.0 (CPU EP), onnxruntime-ggml 0.2.0 (Vulkan).

## Summary

The v2 plan assumed the decoder step is limited by streaming 72 MB of INT8 weights
from system memory, and that keeping the graph and its weights resident on the GPU
would therefore turn ~15.7 ms into 3–5 ms. The measurements contradict the premise
and, with it, the expected payoff:

1. **The graph was never split because of the sampling operators.** The plugin has
   host implementations for `TopK`, `RandomNormalLike`, `ScatterElements` and
   `GatherElements`. The *only* thing rejecting the whole graph is **`ArgMax`**
   (2 nodes in the stage decoder, 1 in the first stage decoder) — a missing host op,
   not a missing GPU kernel.
2. **With `ArgMax` rewritten to `TopK(k=1)` the graph is claimed whole: 1301/1301
   nodes, 147.6 MiB of resident weights.** It is still **2–3× slower than CPU**
   (25–43 ms/step vs 10–15 ms/step).
3. **The GPU kernels themselves are fast.** A flush of the body-only graph runs in
   **3.8 ms**. Everything above that is per-step overhead inside the plugin:
   graph rebuild, buffer allocation, and 3.1–9.1 MiB of KV read-back.
4. **That overhead is dominated by tensor-count, not bandwidth.** Moving the same
   15.7 MB as 1 tensor costs 12.3 ms; as 48 tensors it costs 26.2 ms. The decoder
   crosses that boundary with 48 KV tensors every step by construction.
5. **The CPU bottleneck is not weight streaming either.** ORT's own kernel profile
   puts 37.1% of a step in `DynamicQuantizeMatMul` and 38.7% in layout ops
   (`Transpose`/`Concat`/`Reshape`/`Unsqueeze`). Both scale with node count, not with
   the 72 MB weight set. Adding threads does nothing (3/6/8/12 threads all land
   within the noise), which fits a per-node-overhead bound.
6. **The one change that worked does not involve the GPU at all.** Storing the KV
   cache in attention-native layout removes 48 per-layer `Reshape`+`Transpose`
   chains that each touched the whole cache: **1.10–1.23× on the CPU path**, with
   bit-identical logits and bit-identical rendered audio. It is implemented, on by
   default, and falls back when the generated models are absent.

The practical conclusion: the decoder's cost is **operator and layout overhead, not
memory bandwidth**. Moving it to the GPU moves the same overhead somewhere that also
rebuilds its graph every run, so it loses. Graph-level surgery on the CPU path
(fewer layout nodes) attacks the largest removable block directly.

## Reproducing

```bash
dotnet build SolTTS.sln -c Release

# Generate the native-KV-layout decoders (needed once per resources tree)
python tools/converter/fuse_kv_layout.py \
    resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx \
    resources/models_v1/chenxing/t2s_stage_decoder_kvlayout.onnx
python tools/converter/fuse_kv_layout.py \
    resources/models_v1/chenxing/t2s_first_stage_decoder.onnx \
    resources/models_v1/chenxing/t2s_first_stage_decoder_kvlayout.onnx
python tools/converter/verify_kv_layout.py \
    resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx \
    resources/models_v1/chenxing/t2s_stage_decoder_kvlayout.onnx

# CPU / GPU single-step latency at several KV lengths
dotnet run --project tools/decoder_bench -c Release -- \
    --model resources/models_v1/chenxing/t2s_stage_decoder_qkv.onnx --steps 60

# ORT kernel hotspots (writes onnxruntime_profile_*.json to the cwd)
dotnet run --project tools/decoder_profile -c Release -- --kv 96 --iy 2 --steps 20

# What the ggml plugin claims, and what it names as unsupported
dotnet run --project tools/ggml_probe -c Release -- \
    resources/models_v1/chenxing/t2s_stage_decoder.onnx 1

# Per-tensor transfer cost at the KV boundary
dotnet run --project tools/xfer_probe -c Release
```

The converter scripts are in `tools/converter/`: `fuse_kv_layout.py` (wired into
the engine), `verify_kv_layout.py`, `rewrite_argmax_as_topk.py` and
`export_decoder_body.py` (the latter two are experiment artefacts for the GPU
path, kept so the negative result is reproducible).

## Baseline: CPU single-step latency

`tools/decoder_bench`, 60 steps after 8 warmup, p50/p95 in ms. The production
decoder is `t2s_stage_decoder_qkv.onnx` (GraphOptimizationLevel 2, 696 nodes).

| KV length | p50 | p95 |
|---|---|---|
| 32 | 8.05 | 9.73 |
| 96 | 9.45 | 10.99 |
| 160 | 10.72 | 13.24 |

The 15.7 ms/step figure in the README is the full end-to-end loop average at the
lengths a real sentence reaches (KV 100–300, `iy` up to ~40). At `kv=160` with the
longer `iy` a real step uses, this tool reproduces ~11–13 ms; the difference from
15.7 is the `iy` length and the KV tail, not a regression.

Thread count does not matter, which is itself a strong signal:

| `IntraOpNumThreads` | kv=32 p50 | kv=96 p50 | kv=160 p50 |
|---|---|---|---|
| auto | 9.72 | 12.35 | 13.76 |
| 3 | 8.66 | 10.71 | 12.37 |
| 6 | 9.65 | 11.23 | 12.49 |
| 8 | 9.62 | 10.80 | 11.65 |
| 12 | 9.13 | 10.88 | 12.11 |

Run-to-run spread is ±1.5 ms, so these are all the same number. A bandwidth-bound
kernel would improve with threads; a node-dispatch-bound graph does not.

## Where a CPU step actually goes

`tools/decoder_profile2`, ORT profiling, `kv=96`, `iy=2`, 20 steps. Percentages are
of the kernel time ORT attributes; the profiler inflates absolute time (it serialises
kernels), so read the shares, not the totals.

| Kernel | share | invocations/step |
|---|---|---|
| `DynamicQuantizeMatMul` | 37.1% | 97 |
| `Transpose` | 14.3% | 121 |
| `Concat` | 10.5% | 51 |
| `Reshape` | 8.7% | 122 |
| `Unsqueeze` | 5.2% | 75 |
| `FusedMatMul` | 4.6% | 24 |
| `MatMul` | 3.8% | 24 |
| `SkipLayerNormalization` | 2.6% | 24 |
| `Split` | 2.5% | 24 |
| `LayerNormalization` | 2.2% | 24 |
| `Softmax` | 1.9% | 25 |
| `Add` | 1.9% | 25 |
| `Relu` | 1.8% | 24 |

The 97 `DynamicQuantizeMatMul` per step are the 97 `MatMulInteger` nodes the
exporter emitted (QuantizeLinear of the activation fused with the integer matmul).
They are 37.1% of the step. The layout ops (`Transpose` + `Concat` + `Reshape` +
`Unsqueeze`) are **38.7%** and exist almost entirely to satisfy the exporter's rank
gymnastics, not to compute anything.

That is the real target: 76% of a step is quantization kernels plus layout, and
neither changes if the data lives on the GPU — the plugin has to run the same 97
quantized matmuls (it dequantizes them at upload) and the same reshapes.

## The GPU claim experiment

### What actually blocked the graph

`tools/ggml_probe` on the shipped decoder, GraphOptimizationLevel 0, 1 and 2:

```
2 of 1300 nodes use ops this provider lacks: ArgMax x2
```

The plugin's `op_supported` consults its host interpreter's op list. `TopK`,
`RandomNormalLike`, `ScatterElements` and `GatherElements` are all on it
(`host/eval_index.rs`, `host/eval_random.rs`). **`ArgMax` is not implemented anywhere
in the plugin**, in any form. So the v2 plan's stage 3 ("implement TopK,
RandomNormalLike, ScatterElements") targets three operators that already work and
misses the one that does not.

### Rewriting ArgMax to TopK

`tools/converter/rewrite_argmax_as_topk.py` replaces

```
ArgMax(x, axis, keepdims=0)  ->  Squeeze(TopK(x, k=1, axis).indices, axes=[axis])
```

TopK returns indices in descending value order with ties broken by lower index,
which is exactly `ArgMax(select_last_index=0)`. Verified numerically on the real
graph with a fixed random seed: `y` and `stop_condition_tensor` are **exactly equal**,
`y_emb` and all 48 `present_*` have **max absolute difference 0.0**.

After the rewrite the plugin reports:

```
claimed 1301 of 1301 nodes
compiled ... weights=296 weight_bytes=147.6 MiB weights_f16=97 weight_f16_bytes=145.0 MiB
```

The graph is resident. This is the state the plan wanted to reach in stage 3.

### What it costs

`tools/decoder_bench`, full-claim session, same inputs as the CPU baseline:

| KV | CPU p50 | GPU full-claim p50 | ratio |
|---|---|---|---|
| 32 | 8.05 | 25.26 | 0.32× |
| 96 | 9.45 | 42.88 | 0.22× |
| 160 | 10.72 | 35.42 | 0.30× |

`partial=1` is only mildly better (20–32 ms) and now makes no sense as a strategy:
the split is not what costs here.

### Why, precisely

With the plugin's debug log on, one step of the rewritten graph reports:

```
flush reason="GatherElements on host"  ggml_nodes=1024  readback=3.1 MiB  alloc_ms=4.66 compute_ms=16.20 read_ms=7.15
flush reason="ScatterElements on host" ggml_nodes=5     readback=8 B
flush reason="Less on host"            ggml_nodes=1     readback=4 B
flush reason="TopK on host"            ggml_nodes=3     readback=4.0 KiB
run host_ops=25 device_ops=840 flushes=4 uploads=57 upload=3.0 MiB total_ms=21.09
```

Four flushes per step, because the sampler is host-side and forces a sync between
the transformer output and the next host op. Each flush:

- rebuilds a ggml graph from the program IR (`emit_ms` 18–35 ms in the log),
- runs `ggml_gallocr_alloc_graph` (4–8 ms),
- reads back every live device value (5–7 ms for 3.1 MiB),
- calls `ggml_reset` and `ggml_new_graph_custom`.

The compute itself is **3.8 ms** per flush once warm. The overhead is ~4–5× the
compute, every step.

Removing the sampler from the graph (`tools/converter/export_decoder_body.py`)
gets to **one** flush per step. It does not make the GPU path faster overall,
because the 48 `present_*` KV tensors still have to come back to the host each step
and the 48 `past_*` tensors go up again — and each tensor pays a fixed per-tensor
cost:

`tools/xfer_probe`, pure copy, 15.7 MB each way:

| Shape | p50 |
|---|---|
| one `[48,160,1,512]` tensor | 12.33 ms |
| 48 `[160,1,512]` tensors | 26.24 ms |

That is ~0.8 GB/s effective, and the decoder's fixed 48-tensor KV boundary can be
neither merged nor avoided without plugin changes (`sticky` cannot help: every
`present_*` differs each step, so each is marked volatile and re-uploaded).

Best case with everything working — body-only graph, one flush, `q8_0` weights:

| KV | CPU p50 | GPU p50 |
|---|---|---|
| 160 | 9.2–10.0 | 7.5–10.0 |

That is parity within noise, not 3–5×.

**A correction worth recording.** Measuring the rewritten graph with
`fuse_kv_layout.py` at CPU-native KV shapes first appeared to show the GPU at
8.6–15.8 ms — near parity. That reading was wrong: the probe had not been told
the new input rank, so the session silently ran the CPU EP. With the KV layout
rewrite and the ArgMax rewrite combined and an actually-claimed session
(`claimed 1157 of 1157`), the GPU numbers are 21.7–47.0 ms against a CPU
7.99–14.65 ms. Whenever a rewrite changes graph input ranks, verify the claim
before trusting a timing.

## Why the Loop model does not rescue this

`t2s_stage_decoder_loop.onnx` was measured earlier at 1.02×/0.98×. The reason it can
not help the GPU path either: the plugin's `GetCapability` only walks subgraphs to
check support (`nested_supported`), the `Loop` body is executed by the plugin's
**host interpreter**, and `Loop` itself is on the host list. Every iteration would
be host-driven, so the KV cache still crosses to the host each iteration — the same
boundary, with more Python-shaped glue.

## The KV layout rewrite: the one change that worked

`tools/converter/fuse_kv_layout.py` stores the cache the way attention consumes it
instead of `[S,1,512]`, removing 48 chains of `Reshape`+`Transpose`+`Unsqueeze`
that each touch the entire cache:

| Model | nodes | `Transpose` | `Unsqueeze` | layout ops total |
|---|---|---|---|---|
| shipped (`_qkv`) | 1252 | 121 | 75 | 369 |
| native KV layout | 1156 | 73 | 27 | 273 |

`tools/kv_layout_ab` runs both models in one process in alternating rounds, so
machine noise hits both equally (25 steps/round, 7 rounds; the reported number is
the median of per-round medians, with the range in parentheses):

| Position in a sentence | flat p50 | native p50 | speedup |
|---|---|---|---|
| `iy=2, kv=32` (start) | 10.10 (9.0–11.5) | 8.91 (7.8–12.1) | 1.13× |
| `iy=16, kv=96` (early) | 9.82 (9.5–9.9) | 8.68 (8.3–8.8) | 1.13× |
| `iy=48, kv=160` (mid) | 12.60 (12.2–14.4) | 11.22 (10.8–12.6) | 1.12× |
| `iy=96, kv=320` (late) | 15.95 (15.8–19.8) | 14.94 (14.3–16.1) | 1.07× |

So **1.07–1.13×**, and it is the only change in this investigation that improved
anything. Verified numerically exact: logits, `y_emb` and all 48 caches have max
absolute difference **0.0** across `seq ∈ {8,40,96}` and `iy ∈ {2,17}`
(`tools/converter/verify_kv_layout.py`).

The rewrite changes the KV interface — `past_k_*` is `[1,16,32,S]`, `past_v_*` is
`[1,16,S,32]` — but **not the caller**: the decode loop already feeds each step's
`present_*` straight back as the next `past_*` (`InferenceEngineV1.cs`), so both
sides use the same layout and no conversion is needed. The first stage decoder is
rewritten the same way, which is what makes the hand-off work: it emits its whole
initial cache once per sentence in native layout, and the loop passes it to the
stage decoder unchanged.

This is wired up and on by default:

- `UseNativeKvLayout: true` in `TTSConfig` selects
  `t2s_stage_decoder_kvlayout.onnx` / `t2s_first_stage_decoder_kvlayout.onnx`
  when present, and falls back to the shipped models when they are not.
- Both files are generated by `tools/converter/fuse_kv_layout.py` and are
  gitignored with the rest of `resources/`, so a release needs to run the
  converter once.

End-to-end verification with every random source seeded (`RandomNormalLike` in
both decoders and in VITS):

| Check | Result |
|---|---|
| 12-step autoregressive roll-out, original vs native | token sequences identical |
| Full render, original vs native | audio **bit-identical** (cosine 1.0, max abs delta 0, same sample count) |
| `audio_cmp` A-vs-A control | bit-identical, so the harness itself is deterministic |

The control matters: an unseeded A-vs-A run gives cosine 0.86, because
`RandomNormalLike` in the decoders and in VITS makes every render different.
Any audio comparison in this project needs those seeds pinned first.

## What this means for the v2 plan

| v2 assumption | measured reality |
|---|---|
| Graph is rejected by TopK/RandomNormalLike/ScatterElements | Rejected only by `ArgMax` (2 + 1 nodes); the other three are supported |
| Whole-graph claim needs 5–8 days of new GPU kernels | Claim reached in one converter script; the kernels are not the bottleneck |
| 72 MB/step weight streaming is the limit | 37.1% quantization kernels + 38.7% layout ops; threads make no difference |
| Whole-graph residency gives 3–5× | Gives 0.2–0.3×; GPU compute is 3.8 ms but per-step plugin overhead is 12–20 ms |
| Device allocator + IOBinding removes the copy cost | Would remove part of it, but the 48-tensor boundary cost (26 ms for 15.7 MB) and per-run graph rebuild remain |

The device-allocator work in stage 2 is real and would help the vocoder and any
future GPU-resident model, but it cannot make this decoder beat the CPU: even at
zero transfer cost the plugin still spends 18–35 ms rebuilding the graph per run.

## Stage-1 runtime options: nothing to keep

`tools/runtime_ab` runs every session option from the plan's stage 1 against the
same graph and inputs. Each variant is measured in 10 interleaved rounds of 30
steps; the number is the median of per-round medians so that scheduling noise
does not decide the winner.

| Variant | median ms | vs baseline |
|---|---|---|
| baseline | 10.46 | 1.00× |
| `IntraOpNumThreads=3` | 10.40 | 1.01× |
| `IntraOpNumThreads=6` | 10.60 | 0.99× |
| `no memory pattern` | 11.29 | 0.93× |
| `no cpu arena` | 10.70 | 0.98× |
| `allow_spinning=0` | 12.38 | 0.85× |
| `allow_spinning=1` | 10.41 | 1.01× |

Nothing clears the noise floor, and the two variants that clear it are *worse*.
The plan expected 5–15% from this stage; the measured number is 0%. That is
consistent with the kernel profile: the graph is bound by per-node dispatch, so
threads and arena flags have nothing to act on. `IntraOpNumThreads=8` and `=12`
are actively harmful (0.85× and 0.71× in the 5-round run), which is why the
shipped default of letting ORT decide is the right one.

## Recommendations, in order of expected value

1. **Layout-op elimination on the CPU path.** 38.7% of a step is
   Transpose/Concat/Reshape/Unsqueeze. A converter pass that folds the per-layer
   rank gymnastics (especially the 24 QKV `Concat` + `Transpose` + `Reshape` chains)
   attacks the largest removable block. This needs no plugin work and no GPU.
2. **Reduce `DynamicQuantizeMatMul` count (37.1%).** 97 quantized matmuls per step
   for 24 transformer layers is ~4 per layer; some of that is the exporter
   quantizing weights that are already static. Static-scale quantization
   (`QLinearMatMul`-style with a constant activation scale, or keeping the 97
   weight paths as `MatMul` + `DynamicQuantizeLinear` only where needed) would halve
   the count. The earlier `QLinearMatMul` attempt failed for a real reason (int8
   output saturates the ±20000 accumulators) — the fix is a wide accumulator, not
   abandoning the direction.
3. **Only then reconsider GPU.** Even with the KV rewrite, the plugin's fixed
   12–20 ms per-run overhead makes GPU pointless for this decoder regardless of how
   good its kernels get. The GPU is the right home for the
   vocoder (250 MB, one call, already 5×) and not for a 95-step decoder whose
   per-step state crosses a 48-tensor boundary.

## Negative results to keep

- Whole-graph ggml claim of the decoder: reachable (ArgMax→TopK), **0.2–0.3×**.
- `partial=1` on the decoder: 20–32 ms, worse than CPU.
- `weights=q8_0`: helps the flush (9.4 vs 13.1 ms at kv=32) but not enough to win.
- Thread sweeps 3–12: no effect.
- `sticky` for KV: cannot apply; the cache is volatile by construction.

## Remaining headroom, measured

After the KV layout rewrite, the per-step cost decomposes cleanly. The decoder is
now within about 10% of its own memory floor:

| Component | Cost | How it was measured |
|---|---|---|
| 72.5 MiB of INT8 weights | 7.4–7.9 ms | step with KV length 1 |
| KV stream at `kv=320` | +6.1 ms | `tools/kv_traffic` |
| Everything else (dispatch, layout) | ~1–2 ms | `kv=1` step vs the bandwidth arithmetic |

`72.5 MiB / 7.4 ms` is 9.8 GB/s, which is what a single core can pull. So:

- **Weight bytes** are the floor. Halving them should halve the step, and INT4
  does exactly that on paper — but ORT's `MatMulNBits` CPU kernel runs 9× slower
  than the INT8 path on these single-token matvecs (97 matmuls, 100–112 ms/step
  vs 9–14 ms). Weight-only INT4 is a dead end on this runtime.
- **KV bytes** grow with the sentence and are 40% of a late step. Removing them
  needs device-resident caches, which is the GPU track that already measured 2–3×
  slower for other reasons.
- **Forwards per token** is the only lever with headroom larger than 1.5×.

### The multi-token forward, costed

A step that computes S positions instead of 1 costs almost nothing extra, because
the weights are read once either way (`tools/iy_sweep`, native-KV graph with the
last-position Slice removed):

| `iy` | p50 | vs `iy=2` |
|---|---|---|
| 2 | 9.33 ms | 1.00× |
| 4 | 9.71 ms | 1.04× |
| 8 | 10.56 ms | 1.13× |
| 16 | 12.76 ms | 1.37× |
| 32 | 16.84 ms | 1.81× |

Break-even for verifying K draft tokens per forward, as a function of per-draft
acceptance `a` (a value above 1.00 beats today's per-token cost):

| K | cost | a=0.3 | a=0.5 | a=0.7 | a=0.9 |
|---|---|---|---|---|---|
| 2 | 1.00× | 1.30 | 1.50 | 1.70 | 1.90 |
| 4 | 1.04× | 1.36 | 1.80 | 2.44 | 3.31 |
| 8 | 1.13× | 1.26 | 1.76 | 2.78 | 5.04 |
| 16 | 1.37× | 1.04 | 1.46 | 2.43 | 5.95 |
| 32 | 1.81× | 0.79 | 1.10 | 1.84 | 5.34 |

At `a=0.7` a 4-token verify is worth ~2.4×; at `a=0.5` it is 1.8×. Semantic tokens
in TTS are highly correlated with their neighbours, so acceptance in that range
is plausible, but it is unmeasured — and it is the single number that decides
whether this track is worth building.

What it would take, concretely:

1. **A body-only multi-position graph** (`tools/converter/` has no such script
   yet; `export_decoder_body.py` is the closest starting point). The blocker is
   that the exported graph's sampling tail assumes a single position: after
   `Gather(-1)` the `Gather/ArgMax/Equal` chain infers rank 2 where the original
   declares rank 1. Cutting the whole sampler and emitting `logits [1,S,1025]`
   sidesteps it; sampling then moves to C#.
2. **A draft source.** With no second model, the drafts have to come from the same
   forward (Jacobi/lookahead), which means iterating to a fixed point rather than
   one pass — a different cost curve from the table above.
3. **A seeded sampler in C#**, because the graph's `RandomNormalLike` makes
   verification non-deterministic.

None of that is implemented. The measurements above are the go/no-go input.

## Test-harness caveat, learned the hard way

`audio_regression` fails on the unmodified baseline too (its serial-vs-serial
"noise baseline" is 0% because it compares a run with itself). Do not use it as a
gate. Use `tools/audio_cmp` with every `RandomNormalLike` seeded — in both
decoders *and* in VITS — and always run an A-vs-A control first: unseeded, the
same model twice gives cosine 0.86, so an unseeded A-vs-B comparison proves
nothing.

Two rewrite bugs were caught by exactly this kind of check and are worth
remembering: replacing `ArgMax` with `Squeeze(TopK(...))` needed the squeeze axis
pinned, and dropping "no-op" transposes by *shape* equality is wrong — a
`(2,1,0)` transpose of a tensor with equal extents keeps its shape and reorders
the data. `drop_noop_layout.py` now compares values, not shapes.

## "Move the decoder to the GPU": what was actually tested

An earlier version of this document concluded "the GPU does not help". That was
too broad. Three different GPU paths exist and only two had been measured; this
section closes the gap and separates "this plugin is slow" from "a GPU cannot
help".

### The numbers, side by side

Decoder step, INT8 model, same synthetic inputs, p50 in ms:

| KV length | CPU (ORT 1.30) | ggml plugin (Vulkan) | DirectML (ORT 1.23) |
|---|---|---|---|
| 1 | 7.27 | — | 23.96 |
| 32 | 7.32 | 21.7 | 42.42 |
| 96 | 9.97 | 33.7 | 47.24 |
| 160 | 11.58 | 30.6 | 50.41 |
| 320 | 14.37 | 46.9 | 60.53 |

`tools/dml_probe` (DirectML) and `tools/decoder_bench --provider ggml`.

**Both GPU providers are 2–4× slower, and the reason is visible in the KV=1 row.**
With no KV to transfer, DirectML still costs 24 ms against the CPU's 7.3 ms. That
is a fixed per-step cost — kernel launches, descriptor setup, and the fact that a
1-token forward is 97 tiny matvecs of 512–2048 elements, which is exactly the
shape a GPU is worst at. A GPU wins on large batched GEMMs; this workload has
none. It is not a porting problem that a better plugin would fix.

For completeness, ggml's *compute* time was only 3.8 ms per flush — the kernels
are fine. The 20–45 ms is graph rebuild, buffer allocation, and 3.1–9.1 MiB of
readback per step. DirectML does not rebuild graphs and is still 24 ms, which
says the ceiling for this shape on this GPU is not much better.

### What would change the answer

The GPU becomes the right place only if the work per step becomes *batched*: for
example verifying K draft tokens in one forward (the multi-token track) turns 97
matvecs into 97 small GEMMs, which is what a GPU is for, and the 72 MB of weights
would then stay in VRAM instead of crossing PCIe. That is a different programme
with a different cost model, and it is not implemented.

### Rust rewrite: measured, and the ceiling is ~1.5×

The appealing idea is that ORT's INT8 path leaves bandwidth on the table. A
hand-written AVX2 kernel was measured against it (`_work/rust_probe`, 97 matvecs
of the decoder's exact shapes, 72.5 MiB of int8 weights, results consumed):

| Implementation | p50 | effective bandwidth |
|---|---|---|
| ORT INT8, full decoder step | 7.4 ms | 9.8 GB/s |
| Rust AVX2 matvecs, 1 thread | 4.89 ms | 14.5 GB/s |
| Rust AVX2 matvecs, 12 threads | 8.05 ms | 8.8 GB/s |

So a hand-written kernel *is* ~1.5× faster at the matvec part, and — worth
noting — **more threads make it worse**. The 4.9 ms is already near one core's
DRAM limit; rayon's fan-out costs more than the extra channels deliver. That also
retroactively explains why `IntraOpNumThreads` never helped this decoder.

But the matvecs are only 66% of an ORT step. The other 2.5 ms is QKV splitting,
48 LayerNorms, 24 softmaxes, KV concatenation, 97 activation quantisations, RoPE,
and the sampler — all of which a Rust rewrite would have to reimplement:

| Scenario | Speedup |
|---|---|
| Rust matvecs, everything else free (unreachable) | 1.51× |
| Rust matvecs + 2.0 ms of other work | 1.07× |
| Rust matvecs + 1.0 ms of very tight other work | 1.25× |

A realistic Rust decoder lands between **1.05× and 1.3×** for a full rewrite of
the decoder plus its integration, plus permanent maintenance of a second
inference path. That is not a good trade against the 1.1× the KV layout rewrite
already delivered for a converter script.

Rust does make sense for something narrower: replacing the *quantised matvec
kernel inside the existing pipeline* rather than the whole decoder. That keeps
ORT's graph, shape handling and sampler, and swaps only the inner loop — but ORT
does not expose a seam for that, so it would mean a custom op, which is an
open-ended project of its own.

## Bottom line on "is there more to win"

| Track | Ceiling | Status |
|---|---|---|
| KV layout (native cache) | 1.07–1.16× | **shipped** |
| Multi-token forward | 1.8–2.4× at 50–70% acceptance | unmeasured acceptance; the only track with real headroom |
| Rust decoder rewrite | 1.05–1.3× realistic | measured; not worth it |
| GPU (ggml or DirectML) | 0.3–0.5× as-is | measured; needs batching first |
| INT4 weights | 0.1× | measured; ORT kernel is 9× slower |
| Threads / arena / spinning | 1.00× | measured |
| Static activation quantization | 1.05× and numerically wrong | measured |
| Delete identity transposes | 1.00–1.02× | measured, in the noise |

The honest answer to "are GPU and Rust really dead ends": **for this workload
shape, yes — and for a reason that is now measured rather than assumed.** The
decode step is 97 single-token matvecs, which is a memory-bound, launch-bound,
non-batched shape. GPUs lose on launch overhead, INT4 loses on kernel quality,
Rust wins on the inner loop but cannot make up for the 34% of the step it does
not own. The one thing that changes the shape of the problem is doing several
tokens per forward, which would also be what finally makes a GPU worth using.

## Can architecture changes make the GPU competitive? Yes — and here is the accounting

The earlier "GPU loses 2–4x" result is about the plugin as it is, not about the
architecture. A staged plan (fixed shapes → in-graph KV update → IOBinding →
persistent command buffers) maps almost one-to-one onto the measured overheads.
This section costs each stage against those numbers instead of against intuition.

### The measurement that makes the plan viable

Cutting the sampler out of the graph (§ "The GPU claim experiment") and also
dropping the 48 `present_*` readbacks changes the picture completely. Same graph,
same kernels, only the output set differs:

| Body variant | KV=32 | KV=96 | KV=160 | KV=320 |
|---|---|---|---|---|
| 50 outputs (reads back 30.4 MiB of KV) | 24.7 ms | 37.4 ms | 34.0 ms | 43.2 ms |
| 2 outputs (reads back 324 KiB) | 10.7 ms | 16.3 ms | 7.8 ms | 8.9 ms |

The 30.4 MiB of KV readback costs **14–34 ms per step**. Everything else is small
by comparison. That single number is what a device-resident KV cache buys, and it
is why the ordering in such a plan matters.

### Where the remaining 8.9 ms goes (2-output body, native KV)

| Component | ms | Removed by |
|---|---|---|
| ggml kernel execution | 4.45 | nothing — this is the real compute |
| ggml graph emit (rebuilt every `Run`) | 1.25 | stage 4: cached graph / persistent command buffer |
| `ggml_gallocr_alloc_graph` | 1.45 | stage 3/4: pre-allocated persistent buffers |
| KV upload (49 sticky hits, 640 B) | 0.32 | stage 3: resident buffers |
| ORT `ReadInputs`/`WriteOutputs` | 0.08 | stage 3: IOBinding |
| ORT kernel-context + session dispatch | 1.42 | stage 3: IOBinding (partly) |

So after stages 2–4 the floor is the **4.45 ms of actual GPU compute** plus
whatever dispatch cost cannot be removed. Against a CPU step of 7.3–14.4 ms that
is a real win — but only if the KV never crosses the bus.

### Stage-by-stage verdict

**Stage 1 (audit)** — already done in this document and `tools/`; nothing left to
discover. The dynamic-dimension sites are the 48 `past_*` inputs, the
`iy`/`iy_emb` pair, and `y`/`y_emb`; the per-step copy paths are the 50 graph
outputs and the `present_* → past_*` hand-off in `InferenceEngineV1`.

**Stage 2 (fixed shapes + in-graph KV update)** — *necessary but not sufficient
on its own*. Verified feasible here:

- `ScatterElements(cache[1,16,32,MAX], pos, token, axis=3)` expresses the update,
  and a fixed-shape cache survives a 6-step feedback loop unchanged in shape
  (`E:/AI/_work/synth/state_loop.onnx` demo).
- Concat-based update is already `Concat(past, new_token)` in the native-KV graph,
  so replacing it with a positional write is a local edit, not a re-export.
- **But** a fixed `MAX=512` cache is 48 MiB per hand-off, versus ~30 MiB today at
  `kv=320`. Fixed shapes alone make the copy *bigger*. Their value is that they
  let stage 3 keep one buffer per layer forever.

**Stage 3 (IOBinding + resident buffers)** — this is where the 14–34 ms lives.
The plugin currently cannot support it: `create_allocator` returns null,
`create_sync_stream` returns null, and `default_memory_device` returns null, so
ORT has no device memory to bind to and every input/output is host memory
(`ep/provider.rs`). Making this real means implementing a Vulkan allocator and a
sync stream, then teaching the runtime to hand device pointers straight to ggml
instead of copying into `HostTensor`. That is the single largest piece of work in
the whole plan, and it is also the only piece that buys a multiple rather than a
percentage.

A cheaper intermediate was worth testing: keep host memory, but **merge the 48 KV
tensors into one `[48, …]` tensor** at the graph boundary. `tools/xfer_probe` had
shown 15.7 MB as 1 tensor = 12.3 ms versus 48 tensors = 26.2 ms — but that
measurement is of the *ggml plugin's* per-tensor host/device path. On ORT's own
tensor path the result reverses:

| KV | 1 packed tensor | 48 separate tensors | ratio |
|---|---|---|---|
| 96 | 1.03 ms | 0.72 ms | 0.70× |
| 160 | 2.77 ms | 2.53 ms | 0.92× |
| 320 | 7.16 ms | 6.46 ms | 0.90× |

(`tools/kv_pack_probe`, synthetic graphs with matching byte counts.) Packing is
*slower*, so it is not a shortcut around the allocator work. The 48-tensor penalty
is specific to the plugin's per-tensor upload/readback path; ORT's own graph I/O
already amortises it.

**Stage 4 (persistent command buffers)** — worth up to 2.7 ms/step (emit + alloc).
Real, but smaller than stage 3, and it only pays off after stage 3 has removed the
bus traffic; otherwise it optimises 12% of a 30 ms step.

### Realistic ceiling, in order

| Configuration | Est. step | vs CPU |
|---|---|---|
| Today (plugin, full graph, host KV) | 24–43 ms | 0.3–0.5× |
| + sampler moved out (1 flush) | 18–33 ms | 0.5× |
| + no `present_*` readback | 7.8–8.9 ms | 1.0–1.6× |
| + persistent buffers (stage 3/4) | ~5–6 ms | 1.3–2.5× |
| + q8_0 weights (halves resident weight bytes; measured 13.1 → 9.4 ms on a full-graph flush) | ~4–6 ms | 1.3–3× |

The last two rows are estimates built from measured components, not measurements
of a built system. But the first three are measured, and they show the GPU track
crossing from "clearly worse" to "clearly better" **at the point where the KV
stops crossing the bus** — not at the point where the kernels get faster.

### What this does not fix

- A 1-token forward is still 97 tiny matvecs. GPU kernel execution was 3.8–4.5 ms
  against a CPU floor of ~7.4 ms, so even a perfect zero-overhead wrapper caps out
  around 1.6–2×. Raising that cap needs more work per launch, which is the
  multi-token track.
- The 4.45 ms of GPU compute is itself launch-bound. Batching K tokens per forward
  would amortise the same launches over K× the work and is the only change that
  moves the kernel-time floor, not just the wrapper.
- Concurrency: one device buffer set per engine instance means one in-flight
  decode per GPU; the `TtsEnginePool` would need per-instance device state.

### Recommendation

If the GPU track is pursued, do it in this order, each with a measurement gate:

1. **Sampler out of the graph** (already have `tools/converter/export_decoder_body.py`
   and the body graphs used above). Gate: one flush per step, logits still exact.
2. **Measured baseline with 2 outputs**: 7.8–8.9 ms/step at `MAX=512` shapes.
   Gate: this is the number any further work must beat.
3. **Plugin allocator + IOBinding (stage 3)** — where the remaining 14–34 ms
   actually is, and the largest and riskiest change. Tensor packing is *not* a
   shortcut: measured slower on ORT's tensor path (see above).
4. Stage 4 last: it is 2.7 ms, and pointless before stage 3.

Do not start with stage 4 (persistent command buffers) even though it is the most
familiar GPU optimisation: on today's numbers it optimises the smallest term.
