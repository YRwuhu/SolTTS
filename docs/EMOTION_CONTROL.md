# Emotion Control (Guiding Emotion Without Reference Audio)

SolTTS can guide the emotional expression of synthesized speech **without providing emotional reference audio**.
The method extracts an "emotion direction vector" τ from public emotional speech datasets and injects it into the speaker embedding space at inference time.

> Status: experimental; disabled by default (`UseEmotionControl: false`). Applies only to the V1 engine (the default).

## How It Works

Pipeline (identical to the actual inference path):

```
wav(16k) ──speaker_encoder.onnx──▶ sv_emb [20480]
                                      │
                    ┌─────────────────┴──────────────────┐
                    │  x_new = x_base + α·||x_base||·τ̂   │  ← emotion injection
                    └─────────────────┬──────────────────┘
wav(32k) ────────────────────────────▶ prompt_encoder.onnx ──▶ ge [1024], ge_advanced [512]
                                                                     │
                                                        vits.onnx (style/timbre branch)
```

- `speaker_encoder.onnx` outputs a **20480-dimensional** speaker embedding (`sv_emb`), the actual speaker
  representation passed to `prompt_encoder` in GPT-SoVITS v4 (not 256-dimensional, and not the 768-dimensional
  frame-level features of CNHuBERT).
- `t2s_encoder.onnx` (the GPT branch) **takes no speaker input** (it receives only `ssl_content [1,768,T]` and similar),
  so emotion injection affects only the `sv_emb → prompt_encoder` path and never enters GPT.
- Injection math: `x_new = x_base + α · ||x_base|| · τ̂`, where `τ̂` is the unit-normalized emotion direction vector.

### Why the Injection Point Is sv_emb

`prompt_encoder` is the only module that consumes both "speaker identity" and "reference-audio prosody"; its outputs
`ge` / `ge_advanced` are the timbre-style conditions for `vits.onnx`. Rewriting `sv_emb` therefore rewrites the style
condition, decoupled from audio content (`ref_audio`), which enables guiding emotion with "neutral reference audio +
emotion direction".

## Extracting τ (Paired-Difference Formula)

For multi-speaker datasets you must use **paired differences within the same speaker**, otherwise τ absorbs speaker identity differences:

```
τ_e = mean_{s,i} ( x_e^{s,i} − x_n^{s} )
```

That is, compute the difference between the emotional mean and the neutral mean for each speaker s, then average across speakers.

Data source: **CSEMOTIONS** (`AIDC-AI/CSEMOTIONS`, now aliased `ATH-MaaS/CSEMOTIONS`)
- 10 professional voice actors (5 male, 5 female), 7 emotions, 48 kHz studio quality
- Extraction results in this repository: **1380 samples / 10 speakers** (20 per speaker per emotion, full coverage)

Extraction command (GPU required; DirectML is used automatically when available, keeping the CPU free):

```bash
# 1) Download the dataset shards (8 parquet files, ~3 GB)
#    https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS/tree/main/data

# 2) Extract τ
python tools/emotion/extract_tau.py \
    --parquet-dir <shard-dir> \
    --resources <SolTTS/resources> \
    --out <SolTTS/resources/emotion_vectors> \
    --per-speaker-emotion 20 --device dml --threads 2
```

Outputs:

| File | Description |
|---|---|
| `tau_<emotion>.f32` | Unit-normalized τ (little-endian float32, 20480 dimensions), read directly by C# |
| `manifest.json` | Dimensions, consistency, and sample statistics |
| `diagnostics.json` | Per-speaker details for each emotion |
| `tau_vectors.npz` | Raw (un-normalized) τ for analysis |

### Quality Metrics (Results from This Repository)

| Emotion | \|τ\| | Cross-speaker consistency | Notes |
|---|---|---|---|
| happy | 23.4 | 0.47 | High-arousal cluster |
| angry | 32.7 | 0.52 | High-arousal cluster |
| sad | 24.1 | 0.52 | Low-arousal cluster |
| surprise | 29.0 | 0.52 | High-arousal cluster |
| playfulness | 22.1 | 0.48 | High-arousal cluster |
| fearful | 21.7 | 0.50 | Low-arousal cluster |

Consistency = `||mean(Δ)|| / mean(||Δ||)`; around 0.5 is normal (emotional expression varies considerably across speakers).

**Cosine similarity between τ vectors** shows a two-cluster structure:

```
              happy  angry   sad   surprise playful fearful
happy        +1.000 +0.736 +0.159  +0.881  +0.820  +0.351
angry        +0.736 +1.000 +0.169  +0.821  +0.762  +0.344
sad          +0.159 +0.169 +1.000  +0.070  +0.133  +0.874
surprise     +0.881 +0.821 +0.070  +1.000  +0.831  +0.264
playfulness  +0.820 +0.762 +0.133  +0.831  +1.000  +0.246
fearful      +0.351 +0.344 +0.874  +0.264  +0.246  +1.000
```

High-arousal emotions (happy/angry/surprise/playfulness) correlate at 0.71–0.86; low-arousal emotions (sad/fearful) at 0.82;
cross-cluster values are only 0.02–0.37. This is common in emotion embedding spaces — they share an "arousal/activation" component.
If you need "purer" directions (for example, valence only), orthogonalize τ within each cluster before use.

## Usage

### Configuration (`config.yaml`)

```yaml
UseEmotionControl: true
EmotionVectorPath: emotion_vectors   # Relative to ResourcesDir
EmotionAlpha: 0.3                    # Default α
```

### WebUI

Synthesis Lab → choose an emotion from the "Emotion Guidance" dropdown and adjust the "Emotion Strength α" slider (0–1.0).
Selecting "Neutral (no injection)" restores the original behavior.

### HTTP API

```bash
curl -X POST http://localhost:5000/api/tts/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"今天天气真好","emotionId":"happy","emotionAlpha":0.3}' \
  -o happy.wav
```

### CLI

```bash
# One-shot
sol-cli -e happy -a 0.3 "今天天气真好"

# Interactive
/emotion happy 0.3
/emotion neutral      # Disable
```

## α Selection Guide (Measured)

Objective metrics for single synthesis runs (chenxing voice, same text, 32 kHz output; F0 is the median from autocorrelation):

| Configuration | Voice similarity (vs. neutral) | F0 (Hz) | Duration |
|---|---|---|---|
| neutral | 1.000 | 227 | 3.48s |
| happy α=0.1 | 0.948 | 230 | 3.44s |
| happy α=0.3 | 0.915 | 251 | 3.76s |
| happy α=0.5 | 0.842 | 269 | 3.52s |
| angry α=0.1 | 0.925 | 229 | 4.16s |
| angry α=0.3 | 0.893 | 266 | 3.52s |
| angry α=0.5 | 0.786 | 271 | 4.68s |

Conclusions and recommendations:

- **F0 rises monotonically with α** (227 → 271 Hz), confirming that prosody is indeed guided by the emotion direction.
- **Voice similarity decreases smoothly with α**: α=0.1 ≈ 0.93–0.95; α=0.3 ≈ 0.89–0.92; α=0.5 ≈ 0.79–0.84.
- The recommended range is **0.2–0.4**: clearly perceptible emotion while retaining sufficient timbre.
  Use 0.5 for stronger expression, accepting timbre shift.
- Note: the T2S decoder contains `RandomNormalLike` random sampling, so **the audio layer is not bit-reproducible**;
  two runs with identical settings have similarity around 0.95–0.96 (the sampling-noise baseline).
  Therefore, judge voice similarity by its "drop relative to that baseline", not by its absolute value.

## Verification

```bash
# Unit tests (τ loading/normalization/α=0 bit identity/dimension mismatch)
dotnet test SolTTS.Tests --filter EmotionDirection

# End-to-end: generate audio for an α grid
dotnet run --project tools/emotion_eval -c Release -- \
  --config config.yaml --out _emotion_eval \
  --emotions neutral,happy,angry,sad,surprise,fearful,playfulness \
  --alphas 0.1,0.2,0.3,0.5 --repeats 2

# Objective metric analysis
python tools/emotion/analyze_eval.py --eval-dir _emotion_eval --resources resources
```

Verified properties and conclusions:

1. **α=0 / neutral does not touch injection**: `EmotionDirectionController.Apply` returns a bit-identical copy when α=0
   (the unit test asserts exact equality), and `TtsEngineV1` performs no embedding rewrite when α≤0.
   At the `sv_emb → ge` level, α=0 gives `max|Δge| = 0.000e+00` (bit-identical; see `tools/emotion/verify_injection.py`).
2. **Random τ works**: a random-direction τ also changes `ge` substantially (comparable in magnitude to real emotions),
   proving the injection path is actually active rather than an inert perturbation.
3. **τ dimension validation**: if τ and sv_emb dimensions differ, the vector is logged and ignored, without affecting normal synthesis.
4. **Missing-file fallback**: if the τ directory/files are missing, `Loaded=false` and emotion control is silently disabled; synthesis continues as usual.

## Known Limitations

- Emotion τ applies only to the **V1 engine** (V2's speaker path is `sv.onnx` ERes2NetV2 + `sv_emb.npy`, a different mechanism).
- τ is extracted from Chinese emotional speech (CSEMOTIONS); **cross-language/cross-voice transfer is not verified**.
  For non-Chinese text or very different voices, consider re-extracting or adjusting α.
- Directions for high-arousal emotions are strongly correlated (0.71–0.86), so fine distinctions (e.g., happy vs. surprise) may not be sharp.
- The dataset is 48 kHz, but the speaker encoder operates on 16 kHz input (consistent with the inference pipeline);
  the extraction script performs resampling and amplitude normalization internally.

## File Inventory

```
SolTTS.Core/Core/EmotionDirectionController.cs   # τ loading + injection computation
SolTTS.Core/Core/TtsEngineV1.cs                  # Injection hook (PrepareReference)
SolTTS.Core/Core/ITtsEngine.cs                   # TtsOptions.EmotionId / EmotionAlpha
SolTTS.Core/Config/TTSConfig.cs                  # UseEmotionControl / EmotionVectorPath / EmotionAlpha
SolTTS.Web/Controllers/TtsController.cs          # API passthrough
SolTTS.Web/Models/TtsRequest.cs                  # Request model
SolTTS.Web/wwwroot/index.html + js/playground.js # WebUI emotion selection + α slider
SolTTS.CLI/Program.cs                            # -e/--emotion, -a/--emotion-alpha, /emotion
tools/emotion/extract_tau.py                       # Dataset → τ extraction (GPU)
tools/emotion/verify_injection.py                  # ge-level verification (α=0 bit identity, sensitivity)
tools/emotion/analyze_eval.py                      # Objective audio metrics (similarity/F0/energy)
tools/emotion_eval/                                # C# end-to-end evaluation tool
SolTTS.Tests/EmotionDirectionTests.cs            # Unit tests
resources/emotion_vectors/                         # τ data (*.f32 must be generated or obtained from releases)
```
