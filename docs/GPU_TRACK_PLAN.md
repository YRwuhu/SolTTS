# GPU track: implementation plan, costed against measurements

This plan supersedes the "GPU loses 2–4×" verdict in `docs/DECODER_BASELINE.md` for
the *architecture* question. It does not contradict its measurements: the plugin as
shipped is 2–4× slower on this decoder, and the reasons are enumerated below. What
changes the verdict is moving the KV cache off the bus, and — separately — giving the
GPU more than one token of work per launch.

Every number below is marked **measured** (reproduced on this machine, command given)
or **estimated** (arithmetic on measured components, no built system). The rule for
this track: no stage starts before the previous gate is met, and a gate that is not
measured is not a gate.

Hardware/software for all measurements: Ryzen 5 5500, Radeon RX 5500 XT 4 GB,
Vulkan 1.3, Windows 10.0.26100, .NET 10.0.401, ONNX Runtime 1.30.0,
onnxruntime-ggml 0.2.0 (pinned to ORT headers `v1.29.0`, ggml `v0.22.0`).

---

## 0. What was actually measured before writing this plan

These numbers decide the shape of the plan. Anything not listed here is an open
question (§6), not an assumption.

| Fact | Value | How |
|---|---|---|
| ggml kernel execution, 1-token body, 2 outputs, native KV | **4.0–5.0 ms** | `ORT_GGML_LOG=debug`, `compute_ms` per flush (**measured**) |
| Same run: graph emit + gallocr per step | **1.1–3.5 + 1.4–3.6 ms** | same log, `emit_ms` / `alloc_ms` (**measured**) |
| Same run end-to-end | **7.6–7.8 ms** plugin-side, 10.5 ms wall | same log / `bench2out` (**measured**) |
| 50-output body (KV read back), same shapes | **45–52 ms**, readback 30.4 MiB | `body_flat.onnx` + debug log (**measured**) |
| KV upload when content changes every step | +4 ms at kv=160 (15.8 vs 11.9) | `benchchg` (**measured**) |
| `sticky` boundary at kv=96 vs kv=128 | kv=96: `sticky_hits=0, uploads=50, upload=9.2 MiB`; kv=128: `sticky_hits=49, uploads=1` | debug log (**measured**) |
| GPU step, K=1 chunk, kv=96 (no sticky) vs kv=160 (sticky) | 26.1 ms vs 12.6 ms | `ktri`, 9 interleaved rounds (**measured**) |
| Production GPU decoder (native KV) | **claiming nothing** (2 `ArgMax`) | `tools/ggml_probe` (**measured**) |
| Production GPU decoder (5-output body, 2 outputs) | claims 1270/1270, 147.6 MiB resident | `tools/ggml_probe` (**measured**) |
| Multi-token verify, K rows, logits only, GPU | 18.3 / 18.9 / 19.8 ms for K=1/2/4 (kv=96) | `chunkbench` (**measured**) |
| Same graph, GPU compute only (`compute_ms` median of 110 runs) | 5.64 / 6.65 / 6.90 ms for K=1/2/4 | debug log (**measured**) |
| Same graph, CPU | 7.9 / 8.1 / 8.6 ms for K=1/2/4 | `chunkbench` (**measured**) |
| Interleaved K=1/2/4 ratio, CPU | 1.04× / 1.07× at kv=96; 1.05× / 1.09× at kv=160 | `ktri`, 9 rounds × 15 (**measured**) |
| Interleaved K=1/2/4 ratio, GPU | 1.05× / 1.06× at kv=96; 1.14× / 1.10× at kv=160 | `ktri`, 9 rounds × 15 (**measured**) |

The absolute GPU numbers on this machine drift by up to 2× between runs (the plugin
reports `GPU kernel execution 4.0–5.0 ms` and the chunk bench reports 12.6–26.1 ms
for the same graph at different KV lengths); treat **ratios** as the finding and
absolute milliseconds as indicative until a quiet-machine run exists.

Three of these overturn assumptions carried in the previous draft of this plan:

1. **The shipped decoders are structurally single-position.** `/Slice_1` takes
   `starts=[-1]` of the position-embedded input, so the transformer always receives
   `[1,1,512]` regardless of `iy`. `iy` only changes how many *history* rows are
   concatenated into `y_emb`; it never becomes the transformer's sequence length.
   `tools/iy_sweep` therefore measures "longer input history", not "more positions
   per forward" — its 1.04×/1.13×/1.37×/1.81× curve cannot be used as the
   multi-token cost model.
2. **There is no causal mask in the graph.** Attention is a plain
   `MatMul → Softmax → MatMul` over `past_*` plus the single new token, which is
   correct only because the transformer sees one query row. Feeding K rows without
   inserting a mask lets every row attend to every other row, so a mask is mandatory
   for a multi-position graph. Measured caveat: with a correct mask inserted, K=2
   still diverges and it diverges **at position 0**, whose logits cannot depend on
   the mask — so the mask is necessary but was not the cause of the observed gap.
3. **Activation quantization is per-tensor and input-dependent.**
   `DynamicQuantizeLinear` computes `scale = max|x| / 255`. Batching K rows changes
   `max|x|` and therefore the scale: measured `0.1646` for a single row vs `0.2609`
   for the same row inside a K=4 batch. This alone makes batched logits differ from
   sequential logits by O(1), even when K=1 is bit-identical.

Fact 3 is the most consequential and is not a bug in any script: it is how the
exported INT8 model works. Stage 0a is written around it.

A fourth measurement matters for reading every GPU number in this document:
**the plugin's `sticky` cache only engages above 256 KiB per tensor**
(`exec/sticky.rs`, `MIN_BYTES`). At kv=96 a `[1,16,32,96]` K tensor is 192 KiB, so
all 48 KV inputs are re-uploaded every step (`sticky_hits=0, uploads=50,
upload=9.2 MiB`); at kv=128 it is 256 KiB and every one is reused
(`sticky_hits=49`). This is why a K=1 step measured 26.1 ms at kv=96 and 12.6 ms at
kv=160 — the larger cache is *faster* because its contents stop crossing the bus.
Any GPU timing at kv<128 is measuring an upload cost that does not exist in
production (real sentences reach kv 100–300), and any comparison that mixes the two
regimes is invalid.

---

## Stage 0 — Acceptance probe (0.5–1 day)

Purpose: decide whether a multi-token track exists at all. It does not: not because
acceptance is low (unmeasured), but because a K-row forward cannot reproduce the
single-token logits the shipped sampler depends on. Stage 0 is therefore split into
0a (correctness, blocking) and 0b (acceptance, only if 0a passes).

### 0a. Reproduce single-token logits from a multi-position forward

The only correct target is the **production path**: `t2s_stage_decoder_kvlayout.onnx`
at `GraphOptimizationLevel=2` with its normal inputs. Not GOL 0, and not a
hand-modified graph.

Two candidate constructions exist. Both were built and measured, at both GOL 0 and
GOL 2 (the results are identical, so the divergence is not an ORT-fusion artefact):

| Construction | K=1 vs sequential | K=2 | K=4 | Verdict |
|---|---|---|---|---|
| Rewire `Slice_1` + KV reshapes + causal mask (GOL 0) | **0.00** | 0.45 / 4.00 | 0.43 / 3.85 / 0.41 / 0.66 | fails ≥K=2 |
| Same, GOL 2 | 1.9e-06 | 0.45 / 4.00 | 0.43 / 3.85 / 0.41 / 0.66 | fails ≥K=2 |

At K=4 the argmax agrees on positions 0–2 and **disagrees on position 3**, i.e. the
batched forward would already emit a different token by the fourth step. KV columns
differ by 0.045–0.077 for K=2 and ≥0.045 for K=4, so this is not a logits-only
artefact.

**A second implementation exists in the tree and fails differently.**
`tools/converter/export_multiposition.py` (K=4, native-KV model) widens both
last-position `Slice`s, retargets the per-layer K/V reshapes, and fixes the
`Transpose_5` perm — all correct. It then breaks in two ways, both verified:

1. It rewrites the `[-1, 1, 512]` reshape target (attention output → residual) to
   `[-1, K, 512]` while leaving the paired `Transpose_6` perm at `(1,0,2)`. The
   residual stream becomes `[K,4,512]` and layer 1 fails with
   `input shape {4,4,512} requested {1,16,32,4}`. Reverting just that one target
   (`[-1,1,512]`) makes the graph run.
2. It never removes the final `/Gather` (Gather axis 1, index −1) in front of
   `ar_predict_layer`, so `logits` is still `[1,1,1025]` — **one position**, not K.
   Its `--verify` passes because it only compares `y_emb` and `present_*`, never the
   logits. This is the single most important correction for whoever picks up the
   script: a multi-position export that does not check logits has not checked
   anything.

So Stage 0a starts from a concrete, partially-correct script rather than a blank
page, and the two known defects above are 0a-0.

K=1 reproducing the production logits shows the plumbing is sound but is **not**
sufficient evidence: at K=1 the KV `Reshape`+`Transpose` degenerates to a plain
reshape and the causal mask is all zeros, so neither the transpose nor the mask is
exercised (§7.2). The first real test is K=2, and it diverges — including at
*position 0*, whose logits cannot depend on the mask at all. Also measured directly:
`DynamicQuantizeLinear` produces `scale = 0.1646` for one row and `0.2609` for the
same row inside a K=4 batch, because the scale is `max|x| / 255` over the whole
tensor. Every activation quantized per tensor changes when the batch changes.

**Work, in order:**

1. **Stage 0a-0 — fix the two known defects in
   `tools/converter/export_multiposition.py` (0.5 day).** Remove the final
   last-position `Gather` so `logits` is `[1,K,1025]`; stop rewriting the
   `[-1,1,512]` attention-output target (or fix `Transpose_6`'s perm to match).
   **Then make `--verify` compare logits**, one position at a time, against K
   sequential runs. Without that assertion the script reports success on a graph
   that computes one position.
2. **Stage 0a-1 — insert the causal mask and isolate the remaining divergence
   (0.5 day).** The shipped graph has no mask; K rows without one are not causal.
   Build the K=4 graph with `DynamicQuantizeLinear` replaced by a fixed scale taken
   from the K=1 run, then compare. If the gap closes, the cause is quantisation and
   the fix is either (a) per-row quantisation with a scale computed as if K=1, or
   (b) an FP32 variant of the body. If it does not close, the cause is the mask,
   RoPE layout or the KV transpose, and the fix is there, not in the quantiser.
3. **Stage 0a-2 — K=2 sanity (0.25 day).** K=2 is the cheapest possible multi-token
   forward (verify one draft). If K=2 cannot be made exact, stop: there is no point
   costing K=4.
4. **Stage 0a-3 — cost the fix (0.25 day).** A per-row scale means an extra
   reduction and a wider dynamic range in every `MatMulInteger`; an FP32 body means
   abandoning the 147.6 MiB INT8 residency (FP32 is 4× the bytes, ~590 MiB, and the
   measured q8_0/f16 flush already runs 13.1 ms vs 9.4 ms at kv=32 — see
   `DECODER_BASELINE.md`). Write the number down before choosing.

**Gate:** `maxdiff ≤ 1e-4` on logits **for every position**, against the production
GOL=2 path, on at least 8 (kv, iy) combinations, with token sequences sampled from a
real render (not uniform random token IDs), and **argmax agreement at every
position**. K=1 serves as the harness control: it must land within the GOL2
baseline's own noise (measured 1.9e-06; exactly 0 at GOL 0).

**Stop condition:** if neither per-row quantisation nor an FP32 body gets K=2 under
1e-4 without more than ~1.3× the single-step cost, **stop the whole track**. A
multi-token scheme that changes the logits also changes the audio, and this project
requires bit-identical audio for a shipped path (§8).

**Correction to the previous draft:** the earlier gate "`E[accepted] ≥ 1.5` at K=4"
presumed a working K=4 forward and a draft source. Neither exists yet, and
acceptance cannot be measured on a graph that does not reproduce its own reference.
0b is therefore written after 0a, not before.

### 0b. Acceptance measurement (only if 0a passes)

If 0a succeeds, the acceptance rate is the next unknown and is worth measuring
properly, because it multiplies the measured 1.04–1.09× CPU batching gain:

1. Use the **native-KV chunk graph** with per-row-fixed quantisation from 0a.
2. Draft source: n-gram lookup over the token sequence produced by a real render
   (semantic tokens repeat locally; this needs no second model).
3. Run the real decode loop with verification, over ≥20 sentences, and record
   `P(accept≥1)`, `P(accept≥2)`, `P(accept≥3)` — the full distribution, not the mean.

**Gate:** `E[accepted] ≥ 1.5` at K=4 on the CPU path (where batching costs 1.04–1.09×)
→ continue. Otherwise stop: CPU batching at ~1.07× with `E[accepted]=1.2` is a
regression, and the GPU variant is 2.3× slower to begin with.

**Do not measure acceptance on the GPU.** GPU K-row cost is 18.3–19.8 ms against
CPU 7.9–8.6 ms for the same graph (**measured**); the CPU path is the only one where
batching can pay today.

---

## Stage A — Sampler out of the graph (2–3 days)

Goal: one flush per step instead of four. This is already true for a body graph.
Measured, on the 2-output body:

```
run host_ops=4 device_ops=830 flushes=1 uploads=2 upload=62.1 KiB
    sticky_hits=48 readbacks=2 readback=68.0 KiB
    emit_ms=1.07 alloc_ms=1.40 compute_ms=4.02 total_ms=7.60
```

So the work is not "remove the sampler" — `tools/converter/export_decoder_body.py`
already does that, and the result is one flush. The work is to make that body usable
by the real decode loop.

**Work:**

1. **Export the body from the *shipped* model, not the experiment graph.**
   `export_decoder_body.py` currently emits `present_*` with the old `[S,1,512]`
   declaration and therefore fails `onnx.checker` on
   `t2s_stage_decoder_kvlayout.onnx` (**measured**: exit 1, shape-inference rank
   errors). It must declare K as `[1,16,32,S]` and V as `[1,16,S,32]` when the
   source is the native-KV model. Same blocker in
   `export_multiposition_body.py`, which also assumes the flat model.
2. **Move sampling to C#.** The graph's sampler chain is, in order (node names from
   `t2s_stage_decoder.onnx`, nodes 1272–1293):

   ```
   GatherElements(logits, iy_tokens)            # logits at *every* history token
   Where(value < 0, value*1.35, value/1.35)     # repetition penalty, per occurrence
   ScatterElements(logits, iy_tokens, penalised)
   TopK(k=15) -> Gather(-1) -> Unsqueeze        # threshold = 15th largest value
   Where(logits < threshold, -inf, logits)
   Softmax(axis=-1)
   Div(softmax, RandomNormalLike(softmax))      # Gumbel-style draw
   ArgMax(axis=-1, keepdims=1)
   ```

   Note the penalty applies to **all** history tokens held in `iy`, not just the
   last one, and is applied per occurrence. Reproduce that exactly, with the same
   constants (penalty 1.35, top_k 15, vocab 1025, EOS 1024). Getting the order or
   the broadcast wrong changes the distribution: `Div(Softmax, Random)` is a
   Gumbel-style draw and is *not* equal to "sample from the softmax with a uniform".
   The separate `ArgMax` at node 1273 exists only to build `stop_condition_tensor`
   (greedy argmax == 1024 OR sampled token == 1024).
3. **`y` and `y_emb` assembly in C#.** `y = Concat(iy, token)`; `y_emb =
   Concat(iy_emb, embedding[token])`. The embedding table is
   `ar_audio_embedding.word_embeddings.weight`, `[1025,512]` f32 (**measured**,
   2.1 MB) and must be loaded once.
4. **`stop_condition_tensor` in C#.** It is
   `ReduceSum(Cast(Or(EQ(argmax,1024), EQ(sampled,1024)))) > 0`, i.e. "either the
   greedy argmax or the sampled token is EOS".

**Gate:** on CPU, the C#-sampled token sequence is **identical** to the in-graph
sampler for a full render, with all random draws taken from a seeded generator.
Because `RandomNormalLike` is not reproducible across calls in the same session
(**measured**: same session, four runs of `y` gave `8,5,8,809`; fresh sessions with
the `seed` attribute gave `8,8,8,8`), the comparison must pin the graph sampler's
seed **per session** and the C# sampler's seed explicitly. `flushes=1` in the debug
log.

**Note on seeding, which contradicts the previous draft's "seed everything":** the
ONNX `seed` attribute *does* work, but only across a fresh `InferenceSession`. Within
one session the sampler continues its stream, so an "A-vs-A" control run twice in one
session is **not** identical. `tools/audio_cmp` creates a new engine per render,
which is why its A-vs-A control came out bit-identical. Any new harness must either
create a session per render or reseed via C# sampling (Stage A makes this moot).

---

## Stage B — Fixed-shape KV with in-graph positional update (3–4 days)

Value: makes Stage C possible by giving it one buffer per layer that never has to be
reallocated or rebound. On its own it is **slower** (a `MAX=512` cache is 48 MiB per
hand-off vs ~30 MiB at kv=320), exactly as the previous draft said. Do not ship it
alone.

**The blocking question is answered, and it is good news.** In ggml `v0.22.0`:

- `ggml_set_rows` exists in `include/ggml.h` and `GGML_OP_SET_ROWS` in the op enum.
- The Vulkan backend implements it: `pipeline_set_rows_i32/i64[2][GGML_TYPE_COUNT]`,
  dispatched at `ggml-vulkan.cpp:11368` and executed at `:13174`. It accepts F32 or
  F16 source data with I32 or I64 row indices.

So the limitation is in the plugin, not in ggml: `ScatterElements` is in `HOST_OPS`
(`crates/onnxruntime-ggml/src/exec/ops_extra.rs`) and therefore forces a flush. The
work is to write a device emitter that lowers `ScatterElements` with a scalar
`update` along one axis to `ggml_set_rows` (or `ggml_cpy` into a view). That is a
contained change, not a rewrite — which moves this stage from "2 weeks" to the
3–4 day end of the range.

**Work:**

1. `Concat(past, new_token)` → positional write. Two options, in order of
   preference:
   a. **`ggml_set_rows`** with the cache as destination and `pos` as I64 indices.
      Needs a plugin emitter for `ScatterElements` (`ops_extra.rs`) plus the
      `RUNTIME_OPS`/`DEVICE_OPS` bookkeeping. This keeps the update in place.
   b. **`Where(pos_mask, new, cache)`** — `Where` is already a device op
      (`ops_binary.rs:97`) and its condition is a host parameter, so a one-hot mask
      of shape `[1,16,1,MAX]` is uploaded once and reused. Costs a full-cache write
      per layer, which is why it is second.
2. Attention consumes `Slice(cache, 0, past_len+1)`. `Slice` already emits a
   `ggml_view_slice` for rank ≤ 4 (`ops_shape.rs:63`), so it is free — but the
   *shape* must stay symbolic in `past_len` while the buffer stays fixed, which is
   the actual work: the graph must take a scalar `pos` input rather than deriving
   length from tensor shapes.
3. `MAX` buckets `{64,128,256,512}` with one graph per bucket; both decoders must
   use the same `MAX` and layout, or the first-stage hand-off breaks.
4. Turn off `sticky` for these inputs explicitly. `sticky.rs` marks an input
   volatile after one change and then uploads every step; with fixed buffers that is
   exactly the wrong path.

**Gate:** logits for 48 consecutive steps `maxdiff = 0` against the current
`Concat` path (the cache update is a pure copy, so this must be exact, not 1e-4);
one buffer set per layer, no reallocation across a full sentence (verify with
`ORT_GGML_LOG=debug` that `alloc_ms` does not grow and no new `ggml_backend_buffer`
is created); CPU path unchanged.

---

## Stage C — Plugin allocator + IOBinding (3–4 weeks; the only stage that can win big)

This is where the 14–34 ms of KV readback lives. Measured context:

| Configuration | p50 | readback |
|---|---|---|
| Body, 50 outputs (KV back to host) | 45–52 ms | 30.4 MiB |
| Body, 2 outputs (KV stays put) | 7.6–7.8 ms plugin-side | 68 KiB |
| Body, 2 outputs, KV changes every step | 15.8 ms (kv=160) | 68 KiB + 15 MiB up |

The gap between rows 1 and 2 is the entire argument for this stage. Row 3 is the
warning: **uploads still cost 4 ms when KV changes**, because `sticky` cannot help a
cache that changes by construction. Only a resident device buffer that the next step
reads in place removes that.

One correction to the earlier draft's "sticky cannot help" claim: `sticky` *does*
hold the KV in this decoder today, as long as each tensor is large enough. At
kv ≥ 128 the measured log shows `sticky_hits=49` and one 640-byte upload per step —
the 48 KV tensors stay resident between runs because their content does not change
in a benchmark. In the real loop `present_*` *does* change every step, which is why
the plugin marks those inputs volatile and falls back to full uploads; the benchmark
sees hits only because it re-feeds identical bytes. So the production cost is the
row-3 case, and `sticky` cannot substitute for Stage C.

### C-0. API probe (0.5 day, must be first)

Two questions, both answerable without writing the allocator:

1. **Does ORT 1.30 deliver a non-null `OrtAllocator` to the session?** In the plugin,
   `create_allocator` returns null (`ep/factory.rs:173`), as do
   `create_sync_stream` (`:195`), `create_data_transfer` (`:186`),
   `is_stream_aware` (returns `false`, `:191`), and `default_memory_device`
   (`ep/provider.rs:253`). In ORT's header, `EpDevice_AddAllocatorInfo` is what tells
   ORT which memory info to ask for (`onnxruntime_ep_c_api.h:1111`), and the provider
   currently registers **no** allocator info — it registers on the CPU hardware
   device only (`ep/factory.rs:128`, `"registered on the CPU device"`). So the probe
   is: register a fake allocator, log every `CreateAllocator` call and the requested
   `OrtMemoryInfo`, and see whether ORT asks at all, and with what device type.
2. **Can C# bind device memory?** Yes, the API exists:
   `OrtIoBinding.BindInput/BindOutput(OrtValue)`, `BindOrtAllocation`,
   `BindOutputToDevice`, `BindExternalAllocation`, plus
   `OrtValue.CreateTensorValueWithData(OrtMemoryInfo, …, IntPtr, long)`,
   `InferenceSession.CreateIoBinding()`, `GetMemoryInfosForInputs/Outputs`, and
   `OrtIoBinding.SynchronizeBoundInputs/Outputs` (all present in the 1.30 managed
   package). So the C# side is not the risk; the plugin's allocator is.

**Gate for C-0:** a non-null allocator reaches a real session, and a `BindInput` of a
device-allocated `OrtValue` (size ≥ 1 MiB) reaches the provider without ORT copying
it. If either fails, stop before the Vulkan allocator is written — this is the
"3 days or 2 weeks" decision point, and it is now a "3 days or stop" decision point.

### C-1. Vulkan buffer allocator (~1 week)

Implement `OrtAllocator` over `ggml_backend_buft_alloc_buffer` on the Vulkan buffer
type (`ggml_backend_vk_get_default_buffer_type`, exported from the shipped DLL —
verified in the binary's symbol strings). Requirements:

- `Alloc` must return a device pointer that ggml can wrap with `ggml_new_tensor` +
  `ggml_backend_buffer_init_tensor`.
- `Free` must be safe from ORT's thread, which is not ggml's thread; keep a
  `HashMap<*mut u8, Buffer>` behind a mutex.
- `EpDevice_AddAllocatorInfo` for `DEFAULT` and `HOST_ACCESSIBLE`.
- `CreateDataTransfer` for host↔device copies; without it ORT will refuse bound
  values it does not know how to move.

### C-2. Use device tensors without copying (~1 week)

`exec/runtime.rs` currently receives inputs as `HostTensor` and uploads them
(`upload`, `upload_borrowed`, `dev_f32`). With bound device memory, `Run` must
detect that an input's pointer is already a ggml buffer and skip the upload, and
outputs must be written into the bound buffer instead of read back. This touches
`exec/runtime.rs`, `exec/transfer.rs` (`ensure_host`/`flush`) and `ep/compute.rs`.
Add a debug assertion in `flush` that a device-bound input is never memcpy'd; the
count must be zero.

### C-3. Persistent KV as next-step input (~3 days)

48 buffers per program, indexed by layer, bound as both `output` of step *n* and
`input` of step *n+1*. `sticky` cannot do this: it fingerprints host bytes and
every `present_*` changes each step (`sticky.rs`, `volatile`). This needs explicit
buffer identity, not content hashing. One device buffer set per engine instance —
`TtsEnginePool` with `PoolCapacity > 1` therefore needs one device state per engine,
or the decoder sessions must be serialised.

### C-4. Minimal-patch build integration (~2 days)

Build the plugin as a patch file against upstream `v0.2.0`, pinned, so
`SolTTS.Core/third_party/onnxruntime_ggml.dll` can be rebuilt reproducibly. The repo
already ships that DLL; the patch set lives outside the repo today
(`E:/AI/_work/onnxruntime-ggml`), which must be documented.

**Gate:** round-trip bytes from ~30 MiB to **< 1 MiB** per step (debug log);
logits cosine ≥ 0.99999 vs the current path; single step ≤ **6 ms** at kv=160;
rendered audio **bit-identical** using `tools/audio_cmp`, with an A-vs-A control run
first (§8).

**Stop condition:** if after C-1+C-2 the step is still > 8 ms, what remains is
4.0–5.0 ms of GPU compute plus scheduler overhead, and the rest of the plugin work
cannot pay for itself. Write the number down and stop.

**Feasibility note that changes the risk profile:** even in the best case the
decoder's GPU compute is 4.0–5.0 ms (**measured**), against a CPU step of 7.9–8.6 ms
for the same graph. A perfect zero-overhead wrapper therefore caps out at ~1.7–2.1×.
That is a real win, but it is not the 3–5× the original v2 plan assumed, and it is the
number that decides whether Stage C is worth 3–4 weeks.

---

## Stage D — Persistent graph + command buffers (2–3 weeks; only after C)

Removes the per-step emit (1.1–1.6 ms) and gallocr (1.4–1.6 ms) measured above,
2.5–3.2 ms in total.

**Work:** cache `ggml_cgraph` across `Run` instead of rebuilding from the program IR;
keep `gallocr` reservations across runs; use a persistent Vulkan command buffer and
update push constants instead of re-recording; insert barriers only for real
read/write hazards.

**Gate:** step ≤ 5 ms; `maxdiff = 0`.

**Do not start here.** Before Stage C, these two costs are ~25% of a 10.5 ms step
whose 4 ms of readback dominates; after Stage C they are ~55% of a 6 ms step and are
then worth doing. The ordering is the whole point of this plan.

---

## 4. What the multi-token track is actually worth (measured)

This is the track the previous plan put first, and the measurements reorder it:

| Path | K=1 | K=2 | K=4 | Source |
|---|---|---|---|---|
| CPU chunk, logits only, kv=96 | 7.93 ms | 8.14 ms | 8.60 ms | **measured** |
| GPU chunk, logits only, kv=96 | 18.26 ms | 18.86 ms | 19.75 ms | **measured** |
| GPU chunk, compute only (median of 110) | 5.64 ms | 6.65 ms | 6.90 ms | **measured** |
| Interleaved ratio, CPU | 1.00× | 1.04–1.05× | 1.07–1.09× | **measured**, two KV lengths |

CPU batching is nearly free (1.04–1.09× for 4 tokens, interleaved), which is the good
news and is the one number that survived every re-measurement. GPU batching in
compute is also nearly free (5.64 → 6.90 ms, +22% for 4× the work), but the wall
time is dominated by fixed per-run cost, and at kv < 128 it is dominated again by
KV upload because `sticky` does not engage (§0). Measure the GPU at kv ≥ 128 or the
number is meaningless.

**Where the leverage is:** on the CPU, K=4 verification at 1.07–1.09× the cost turns
`E[accepted]` into nearly a direct speedup. On the GPU, K=4 costs only ~1.10× the
wall time of K=1, so the *relative* benefit of batching is similar — but the absolute
cost is still ~2× the CPU path (12.6–27.7 ms vs 15.6–18.4 ms in the interleaved
runs), so it only matters after Stage C+D land.

Because the divergence in §0a is caused by *activation quantization* and not by
batching geometry, the CPU track can proceed independently of the GPU track: fix the
scale on the CPU first, measure acceptance, and only then decide whether the GPU
version of the same graph is interesting.

---

## 5. Revised order of work

1. **Stage 0a** — fix the two defects in `export_multiposition.py`, remove the
   predict-layer `Gather`, insert a causal mask, then make K=2 reproduce the current
   logits; cost the FP32 alternative. *Stop the track if K=2 cannot be made exact
   cheaply.*
2. **Stage 0b** — acceptance on the CPU chunk graph. *Gate: `E[accepted] ≥ 1.5`.*
3. **Stage A** — C# sampler, body graph from the shipped model. Independent of 0,
   and required by any path that moves the sampler out (including the CPU
   speculative track).
4. **Stage C-0** — allocator API probe. *Stop if a non-null allocator does not
   reach the session.*
5. **Stage B**, then **C-1..C-4**, then **Stage D**. Gates as above.

The reordering versus the previous draft: Stage 0a (correctness) now blocks
everything, because the shipped graph cannot express a multi-position forward at
all; Stage C-0 is a 0.5-day stop/go before a 3-week investment; and Stage B is
demoted from "necessary" to "necessary only for C", since it makes the KV copy
*larger* and has no standalone value.

---

## 6. Open questions, in the order they must be answered

1. **Does per-row-fixed quantisation make K=2 exact?** (Stage 0a-1.) If not, is the
   divergence in the mask or in RoPE position handling?
2. **What is the real acceptance distribution for TTS semantic tokens?**
   (Stage 0b.) Still unmeasured — the one number that decides whether any
   multi-token work pays.
3. **Does ORT 1.30 call the plugin's `CreateAllocator` at all,** given that the
   plugin registers on the CPU hardware device with no `EpDevice_AddAllocatorInfo`?
   (C-0.) This is the binary risk on the whole 3–4 week stage.
4. **After C+D, is the 4.0–5.0 ms GPU compute still launch-bound?** (Stage D gate.)
   If so, the only remaining lever is batching — i.e. back to Stage 0, but on the
   GPU, and the ordering above would then need re-examination.
5. **Does the FP32 body fit?** 147.6 MiB of INT8 becomes ~590 MiB of FP32 on a 4 GB
   card shared with the 250 MB vocoder. Measure before assuming it is available.

---

## 7. Pitfalls already paid for (carry these forward)

1. **A shape equality is not an identity.** A `(2,1,0)` transpose of equal-extent
   axes keeps its shape and reorders data. `drop_noop_layout.py` now compares values,
   not shapes.
2. **`Reshape([1,K,512] → [1,16,32,K])` silently transposes for K>1.** It is correct
   only at K=1, which is why the first chunk graph looked plausible. Multi-token KV
   extraction needs `Reshape` + `Transpose`, and the K=1 case cannot prove it.
3. **The decoder is single-position by construction.** `/Slice_1` (starts `[-1]`)
   truncates the transformer input; `/Gather_output_0` (Gather axis 1, index -1)
   truncates the predict layer input. Both must be rewired, and there is no causal
   mask to inherit — it must be inserted.
4. **Seed the graph sampler per session.** The ONNX `seed` attribute is honoured
   only on a fresh `InferenceSession`; within a session the stream continues
   (measured: `8,5,8,809`). An A-vs-A audio control is only meaningful across fresh
   sessions, which is why `audio_cmp` passes and a naive in-process control does not.
5. **`DynamicQuantizeLinear` is per-tensor and input-dependent.** Any batch that
   changes `max|x|` changes every downstream logit. Measure the scale, not just the
   logits.
6. **ArgMax → TopK when the plugin must claim the graph.** Rewriting `ArgMax` as
   `Squeeze(TopK(k=1))` gets a 1301/1301 claim, but pin the squeeze axis — an
   unpinned squeeze is the bug that was caught.
7. **Verify a claim before trusting a timing.** `tools/decoder_bench` silently ran
   CPU when the probe disagreed about input rank; the "8.6 ms GPU win" was that bug.
   Every number must be preceded by `claimed N of N`.
8. **INT4 is 9× slower on this ORT.** `MatMulNBits` on single-token matvecs is
   100–112 ms/step. Closed.
9. **Packing 48 KV tensors is slower on ORT's own path** (0.70–0.90×,
   `tools/kv_pack_probe`), even though it helps the plugin's per-tensor path. It is
   not a shortcut around Stage C.
10. **`sticky` has a 256 KiB per-tensor floor.** Below it every step re-uploads the
    whole KV set (measured at kv=96: 50 uploads, 9.2 MiB); at kv ≥ 128 all 48 hit.
    A GPU timing taken below the threshold measures a cost production does not pay.
    Conversely, benchmark inputs re-fed unchanged make `sticky` report hits that the
    real loop cannot get, because `present_*` changes every step.

---

## 8. Explicitly not doing

- **Rust rewrite of the decoder.** Measured ceiling 1.05–1.3× for a full rewrite, and
  the matvec part alone is 1.5× (4.89 vs 7.4 ms); the other 34% of the step does not
  move.
- **INT4 weights.** 9× slower on this runtime.
- **Threads / arena / spinning tuning.** 1.00×, all within noise.
- **Device-resident KV via `sticky`.** Cannot work for a per-step-changing cache.
- **Packing KV into one tensor.** Slower on ORT's path; does not remove the boundary.
- **Shipping a GPU decoder that changes the logits.** This project's standard for a
  shipped inference path is bit-identical logits and identical audio; a 1.07× win is
  not worth a distribution change.

---

## 9. Artifacts in this repo that this plan uses

| Path | What it is |
|---|---|
| `docs/DECODER_BASELINE.md` | 697-line measured record: CPU profile, GPU claim experiment, transfer costs, KV layout, `iy` sweep, Rust and INT4 results |
| `tools/decoder_bench` | CPU/GPU single-step latency, `--kv-layout flat|native`, `--provider cpu|ggml` |
| `tools/ggml_probe` | claim coverage and unsupported-op list for a converter output |
| `tools/xfer_probe` | per-tensor transfer cost at the KV boundary |
| `tools/kv_pack_probe` | packed vs 48-tensor KV on ORT's own path |
| `tools/kv_traffic` | KV traffic share of a step |
| `tools/iy_sweep` | cost vs input history length (**not** positions per forward — see §0) |
| `tools/audio_cmp` | audio A-vs-A / A-vs-B with fresh sessions per render |
| `tools/dml_probe`, `tools/rust_probe`, `tools/nbits_probe`, `tools/runtime_ab` | the negative results in §8 |
| `_tmp/gpuplan/mkchunk.py` (scratch, gitignored) | builds the K-row verification graph used for every K-scaling number above; regenerate with `python _tmp/gpuplan/mkchunk.py K KV` |
| `tools/converter/export_decoder_body.py` | sampler-free body; needs the native-KV output declaration fix (Stage A) |
| `tools/converter/fuse_kv_layout.py`, `verify_kv_layout.py` | the shipped 1.07–1.13× layout rewrite |
| `tools/converter/export_multiposition.py` | fixed-K query rewrite; runs, but needs the two fixes in Stage 0a-0 (predict-layer Gather, `Reshape_4`/`Transpose_6` pair) and a logits assertion in `--verify` |
| `tools/converter/drop_noop_layout.py` | value-checked layout-op elimination |
| `tools/converter/rewrite_argmax_as_topk.py` | claim-enabling ArgMax rewrite (experiment only) |

Scratch graphs and scripts used for the measurements above were written to
`_tmp/gpuplan/` (gitignored) and are not part of the deliverable; the numbers and
commands are recorded here so they can be rebuilt.
