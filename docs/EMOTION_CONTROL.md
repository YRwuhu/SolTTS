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
UseEmotionProsodyPresets: false      # Optional: emotion-driven speaking-rate nudge
```

### Emotion Blending (v1.4.0+)

Besides a single label, `emotionId` also accepts a weighted blend. Components are combined
linearly and the result is re-normalized to a unit vector, so `τ̂` stays a valid direction:

```
/happy:0.6,shy:0.4     happy+shy        高兴:0.7,害羞:0.3
```

`+` and `,` are both accepted as separators; `:weight` is optional (default `1`). Weights must be
finite and `> 0`; the baseline may appear but contributes nothing. The API returns the canonical
expression (e.g. `happy:0.6+shy:0.4`) so clients can log what was actually applied.

```bash
curl -X POST http://localhost:5000/api/tts/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"才、才不是特意等你的呢！","emotionId":"angry:0.4,shy:0.6","emotionAlpha":0.35}' \
  -o tsundere.wav
```

### Emotion Prosody Presets

V1's `t2s_encoder` takes no speaker embedding, so τ cannot change timing/pausing — only the
timbre-style condition (`ge`/`ge_advanced`) that reaches `vits`. `UseEmotionProsodyPresets: true`
compensates with a small speaking-rate offset tied to the active emotion (+6% excited/flustered,
+3% happy/playful/angry, −5% crying/helpless/worried/fearful, −3% shy/embarrassed/nervous),
interpolated by blend weight and scaled by `α/0.3` (capped at 1.2×). It is a heuristic, not a
learned prosody model; blind listening should decide whether to keep it on.

## Decorrelating the Direction Vectors (v1.4.0)

Single-speaker τ vectors share one dominant axis: across the 13 `ready` labels, the first singular
vector explains **75.6%** of the direction energy, and pairwise cosine averages **+0.673**
(`flustered~excited` 0.973, `happy~playful` 0.957, `shy~lovestruck` 0.940, `happy~crying` 0.736).
That axis is mostly shared arousal/loudness, so different labels sound alike.

`tools/emotion/decorrelate_tau.py` attenuates it:

```bash
python tools/emotion/decorrelate_tau.py \
    --in resources/emotion_vectors_atri19_candidate \
    --out _tau_decorr70 --beta 0.70 --fit-ready-only
python tools/emotion/install_emotion_vectors.py --candidate _tau_decorr70 --apply
```

At `β=0.70` pairwise cosine drops to **+0.175**, `happy~crying` to **−0.084**. End-to-end at α=0.3,
the acoustic delta of synthesized audio separates as well (`happy-crying` +0.758 → +0.379,
`crying-excited` +0.774 → +0.366) while speaker similarity stays at 0.87–0.93. The shipped vectors
(`resources/emotion_vectors`, `soltts-package/...`) use β=0.70; the original set is untouched in
`resources/emotion_vectors_atri19_candidate` for rollback — re-running
`install_emotion_vectors.py --apply` restores it.

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

# Blended (v1.4.0+)
sol-cli -e "happy:0.6,shy:0.4" -a 0.35 "能和你一起走真开心。"

# Interactive
/emotion happy 0.3
/emotion 高兴:0.7,害羞:0.3 0.35
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
- τ is extracted from Chinese emotional speech (CSEMOTIONS) / a single ATRI speaker; **cross-voice transfer is not verified**.
  For very different voices, consider re-extracting or adjusting α.
- Directions for high-arousal emotions are strongly correlated (0.71–0.86 before decorrelation), so fine distinctions
  (e.g., happy vs. surprise) may not be sharp; β=0.70 attenuation helps but does not make them orthogonal.
- The dataset is 48 kHz, but the speaker encoder operates on 16 kHz input (consistent with the inference pipeline);
  the extraction script performs resampling and amplitude normalization internally. Note that the runtime reference-audio
  path (`AudioHelper.ReadWav`) does not apply the extraction script's DC-removal + 0.9 peak normalization, so loudness
  differences in reference audio can shift `sv_emb` slightly (measured cosine 0.986 between raw and normalized input).
- Low-sample labels (`fearful` 7, `embarrassed` 5, `disgust` 14) remain experimental.

## File Inventory

```
SolTTS.Core/Core/EmotionDirectionController.cs   # τ loading + injection + blend composition
SolTTS.Core/Core/EmotionCatalog.cs               # 19 labels, aliases, blend parser, speed presets
SolTTS.Core/Core/TtsEngineV1.cs                  # Injection hook (PrepareReference), ge cache, speed preset
SolTTS.Core/Core/ITtsEngine.cs                   # TtsOptions.EmotionId / EmotionAlpha
SolTTS.Core/Config/TTSConfig.cs                  # UseEmotionControl / EmotionVectorPath / EmotionAlpha / UseEmotionProsodyPresets
SolTTS.Web/Controllers/TtsController.cs          # API passthrough
SolTTS.Web/Models/TtsRequest.cs                  # Request model
SolTTS.Web/Services/TtsRequestValidator.cs       # Single-label and blend validation before audio headers
SolTTS.Web/wwwroot/index.html + js/playground.js # WebUI emotion selection + α slider
SolTTS.CLI/Program.cs                            # -e/--emotion (incl. blends), -a/--emotion-alpha, /emotion
tools/emotion/extract_tau.py                     # Dataset → τ extraction (GPU)
tools/emotion/extract_tau_atri.py                # ATRI single-speaker → 19-label τ
tools/emotion/decorrelate_tau.py                 # PC1 arousal-axis attenuation (β)
tools/emotion/install_emotion_vectors.py         # Validate + install τ into every runtime dir
tools/emotion/verify_injection.py                # ge-level verification (α=0 bit identity, sensitivity)
tools/emotion/analyze_eval.py                    # Objective audio metrics (similarity/F0/energy)
tools/emotion_eval/                              # C# end-to-end evaluation tool
SolTTS.Tests/EmotionDirectionTests.cs            # Unit tests (loading, α=0 identity, blending)
resources/emotion_vectors/                       # τ data in use (β=0.70)
resources/emotion_vectors_atri19_candidate/      # Original τ set (rollback reference)
```
