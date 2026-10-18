# SolTTS

<p align="center">
  <img src="images/banner.png" alt="SolTTS Banner" width="400">
</p>

<p align="center">
  <strong>🎙️ A fast, self-hosted TTS engine — ONNX Runtime, C#, no Python at runtime</strong>
</p>

<p align="center">
  <a href="README_zh.md">中文文档</a> ·
  <a href="https://github.com/YRwuhu/SolTTS/releases">Downloads</a> ·
  <a href="docs/EMOTION_CONTROL.md">Emotion control</a>
</p>

---

## What is this

SolTTS is a text-to-speech server and CLI. You give it text, it gives you audio. It runs
entirely on ONNX Runtime inside .NET — no Python, no PyTorch, no conda at runtime.

It supports voice cloning from a short reference recording, mixed Chinese/English/Japanese
reading, streaming playback, and (experimentally) emotion guidance without an emotional
reference clip.

**This repository is a fork.** The original project by Blackwood416 is no longer available
at its old address. This fork carries the v1.2.2 codebase plus a set of measured performance
and reliability changes, listed below.

---

## Quick start (Windows)

1. Download `SolTTS-v1.2.2-win64.zip` from [Releases](https://github.com/YRwuhu/SolTTS/releases).
2. Extract it anywhere. Keep the `resources/` folder next to the executables.
3. Run `sol-server.exe`, then open <http://localhost:5000>.
4. Or for a quick one-shot file: `sol-cli.exe -O out.wav -- "今天天气真好。"`

No .NET installation is needed — the package is self-contained.

### Linux / macOS

Build from source (see below). The Linux package does not embed the Python model-converter
environment; run `./init-env.sh` once if you need the converter page.

The macOS build uses the V1 engine only.

---

## What's actually fast, and what isn't

We measured everything on a Ryzen 5 5500 (6C/12T) + Radeon RX 5500 XT. Numbers below are
medians, not best-case picks.

| Case | Latency |
|---|---|
| Single short sentence (~4 s audio) | ~2.0 s |
| Three sentences, non-streaming | ~2.2 s |
| Three sentences, streaming, time to first audio | ~0.55 s |
| Model load (cold start) | ~7 s |

Roughly 85% of a single sentence is the autoregressive semantic decoder: about 95 steps at
~15.7 ms each. That cost is memory-bandwidth-bound (each step reads ~72 MB of INT8 weights),
which is why most "just put it on the GPU" ideas do not work here. See
[Performance notes](#performance-notes) for the experiments and the negative results.

---

## Features

- **Voice cloning** from a few seconds of reference audio.
- **Streaming** output with roughly half a second to first audio.
- **Mixed-language reading** — Chinese/English and Chinese/Japanese in one sentence.
- **Emotion guidance** (experimental) — steer happy/angry/sad etc. by injecting a direction
  vector into the speaker embedding. No emotional reference audio needed.
  See [docs/EMOTION_CONTROL.md](docs/EMOTION_CONTROL.md).
- **WebUI** for voice management, model conversion, and parameter tuning.
- **Concurrent requests** through an inference pool.
- **GPU vocoder** via the ggml/Vulkan plugin, with automatic CPU fallback.

---

## Performance notes

These are the things that were tried, and what actually happened. Keeping the negative
results here so nobody re-runs them.

### What helped

| Change | Effect |
|---|---|
| Vocoder on GPU (ggml/Vulkan) | vits kernels ~5×; ~25% end-to-end non-streaming, ~2× streaming |
| Parallel sentence decoding (`ParallelSentenceDegree: 3`) | 3 sentences: ~3.5 s → ~2.4 s |
| Streaming cross-sentence overlap | Next sentence's decoder starts while the previous sentence's vocoder runs; ~10% off streaming total |
| `VocoderRetriggerFactor: 4.0` | Vocoder calls 56 → 42 per run; vocoder time −29%; streaming total ~−10% |
| BERT feature cache | Saves a ~280–390 ms forward pass on repeated text |
| Rust native audio (`sol_audio`) | Audio pipeline p99 27 µs → 4 µs; ring buffer 0.70 µs → 0.05 µs per chunk |

### What did not help

| Attempt | Result |
|---|---|
| INT8-quantizing the decoders | Already INT8 in the shipped graphs (`MatMulInteger` + `DynamicQuantizeLinear`) |
| BERT on GPU | **1.7× slower** (294 → 489 ms). The plugin refuses the graph ("22 of 1089 nodes use ops this provider lacks: Gelu") |
| ggml `partial` offload of the decoder | 14.63 ms/step vs 15.71 ms on CPU — 7%, not worth the memory |
| Merging decode steps into one ONNX `Loop` | 1.02× / 0.98× — inside noise |
| Shortening the decoder's `iy` input | 109 → 2 tokens changes nothing (15.36 → 15.25 ms/step) |
| Incremental vocoding (decode only new tokens) | Not possible: the VITS `enc_p` attention and flow depend on the whole sequence |
| CoreML on macOS | Slower than CPU for these models |

The honest summary: the decoder is bandwidth-bound on a single CPU core's memory path, and
no configuration switch changes that. Getting a real speedup there needs device-resident KV
caches, fixed-shape export, and IOBinding — a model-side project, not a runtime tweak.

---

## Engine comparison

V1 is the default and is what the numbers above were measured on. V2 is present in the code
but **cannot run as shipped** — there are no `models_v2` files and no V2 converter.

| | V1 (recommended) | V2 (experimental) |
|---|---|---|
| Origin | Genie-TTS architecture | GPT-SoVITS-Minimal |
| Status | Stable | Incomplete, no models shipped |
| Languages | ZH/EN and ZH/JA | ZH/EN |
| Sampling | Deterministic | TopK / temperature |
| Emotion control | Yes | No |
| GPU vocoder | Yes | No |
| Parallel sentences / streaming overlap | Yes | No |

If you want speed, use V1. Completing V2 would at best match V1, because its decode loop
clones the entire KV cache every token.

---

## Configuration

Everything lives in `config.yaml`. Only a few settings matter for performance:

```yaml
ResourcesDir: "./resources"
DefaultAvatarId: chenxing

# Parallel sentence decoding for non-streaming requests. 1 disables it.
ParallelSentenceDegree: 3

# How aggressively the streaming vocoder re-triggers.
# Higher = fewer recomputes = faster, but longer chunks between audio updates.
VocoderRetriggerFactor: 4.0

# GPU vocoder (ggml/Vulkan). Falls back to CPU if the plugin is missing.
UseVocoderGgml: true

# ONNX Runtime threads. 0 lets ORT decide.
IntraOpNumThreads: 0
InterOpNumThreads: 1
```

Two settings are worth leaving alone unless you have a reason: `IntraOpAllowSpinning: true`
contends with the concurrent vocoder, and `GraphOptimizationLevel: 2` adds startup time for
no measurable gain.

See [`config.template.yaml`](config.template.yaml) for every option with comments.

### Resource layout

```text
resources/
├── models_v1/{avatarId}/     # t2s_*.onnx, vits.onnx, prompt_encoder.onnx
├── shared/
│   ├── g2p/                  # dictionaries and G2P models
│   └── v1_extra/             # bert, hubert, speaker_encoder
├── avatars/{avatarId}/       # reference audio
└── emotion_vectors/          # tau vectors for emotion control
```

---

## Building from source

Requirements: .NET 10 SDK. Optional: Rust toolchain for the native audio library.

```bash
dotnet build SolTTS.sln -c Release

# Windows package
./publish.ps1

# Linux
./publish-linux.sh

# macOS
./publish-mac.sh
```

### Native audio library (optional)

The real-time playback path can use a small Rust library instead of the managed
implementation. It is optional — if the DLL is missing everything falls back to C#.

```bash
cd native/sol_audio
cargo build --release
```

Then set `UseNativeAudio: true`. See [native/sol_audio/README.md](native/sol_audio/README.md)
for the equivalence tests and benchmarks.

### Diagnostic tools

```bash
# End-to-end smoke test: both paths must produce non-empty audio
dotnet run --project tools/stream_check -c Release -- --resources resources

# Latency, single/multi-sentence, streaming
dotnet run --project tools/latency_bench -c Release -- --resources resources

# Structural regression for parallel decoding
dotnet run --project tools/audio_regression -c Release -- --resources resources

# Streaming failure stress test
dotnet run --project tools/stream_stress -c Release -- --resources resources
```

---

## Troubleshooting

**"The model converter page does not work."**
The Python runtime is not bundled in the zip (it is ~615 MB). Run `init-env.sh` /
`init-env-mac.sh`, or install the converter dependencies yourself.

**"GPU acceleration seems off."**
Check that `onnxruntime_ggml.dll` sits next to the executables and that your GPU driver
supports Vulkan. The app logs a fallback warning if the plugin cannot load. It cannot tell
you whether the plugin actually claimed the graph — if you want proof, run with
`DebugMode: true` and look for the plugin's capability lines.

**"Streaming stops mid-sentence with an error."**
Older builds could fail the whole request when the decoder's sampling went bad. This fork
degrades instead: it returns the audio it managed to generate. If you still see hard
failures, the model files are likely damaged.

**"It is slower than the numbers above."**
Those numbers assume `ParallelSentenceDegree: 3`, `VocoderRetriggerFactor: 4.0`, and a GPU
vocoder. CPU-only builds are noticeably slower.

---

## License

MIT. Original work © 2026 Blackwood416; fork modifications © 2026 YRwuhu.

## Credits

- [Genie-TTS](https://github.com/High-Logic/Genie-TTS) — V1 architecture reference
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — V2 algorithms
- [ONNX Runtime](https://onnxruntime.ai/) — inference backend
- [NAudio](https://github.com/naudio/NAudio) — audio I/O
- [wasapi_relink](https://github.com/Litttlefish/wasapi_relink) — low-latency WASAPI helper
- [onnxruntime-ggml](https://github.com/thewh1teagle/onnxruntime-ggml) — Vulkan/Metal execution provider
- [BreakingBad (AI-Hobbyist)](https://www.ai-hobbyist.com/thread-1143-1-1.html) — source of the bundled default voices
