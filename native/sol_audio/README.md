# sol_audio — SolTTS Native Real-Time Audio Primitives

A Rust `cdylib` that exposes four hot-path primitives for the SolTTS real-time playback path through a C ABI:

| Export | Replaces managed implementation | Description |
|---|---|---|
| `sol_ring_create/write/read/available/clear/destroy` | `LockFreeWaveProvider` | SPSC ring buffer (bytes); capacity is rounded up to a power of two |
| `sol_hpf_process` | `HighPassFilter.Process` | First-order high-pass; state is caller-owned and bit-exact |
| `sol_crossfade` / `sol_fade_out` | Crossfade/fade-out loops in `AudioPipeline` | 20 ms overlap-add |
| `sol_f32_to_i16` | `AudioPipeline` float→PCM16 conversion | Matches C# `(short)` truncation toward zero |
| `sol_audio_version` | — | Version handshake: `major<<16 \| minor<<8 \| patch`, currently `0x000100` |

Model inference (ORT sessions, ggml EP, WASAPI) remains entirely on the C# side; this crate depends on no inference library.

## Build

Requires a Rust toolchain (the local machine uses `x86_64-pc-windows-msvc`):

```bash
cd native/sol_audio
cargo build --release
```

Artifacts:
- Windows: `target/release/sol_audio.dll`
- Linux: `target/release/libsol_audio.so`
- macOS: `target/release/libsol_audio.dylib`

`SolTTS.Core.csproj` copies an existing artifact to the output directory during the build; if it is absent, the step is
skipped, `NativeAudio.IsAvailable == false` on the managed side, and the original C# implementation is used automatically.
Builds and runs never fail because of this.

## Enabling

Set `UseNativeAudio: true` (default `false`). CLI streaming playback then uses the native ring buffer and
`AudioPipeline`; if the DLL is missing, the version does not match, or the platform is unsupported, it falls back silently.

## Equivalence Guarantees

`SolTTS.Tests/NativeAudioTests.cs` covers:

- Ring buffer: byte-identical to the managed implementation (including wraparound, whole-block drop on insufficient capacity, and silence-fill accounting)
- High-pass filter: bit-identical (`BitConverter.SingleToInt32Bits` comparison)
- Crossfade / fade-out: bit-identical
- float→PCM16: exactly matches `(short)(clamp(x) * 32767)`
- `AudioPipeline` end to end: native and managed PCM output are byte-identical
- Concurrent producer/consumer: 2000 blocks read/written across threads with no loss

These tests are skipped automatically when the native DLL is absent (`SkippableFact`).

## Measured (6C/12T Ryzen 5 5500, 32 kHz, 1024 samples/block)

| Item | Managed | Native |
|---|---|---|
| AudioPipeline mean | 12.1 µs | 3.3 µs |
| AudioPipeline p99 | 27.0 µs | 4.1 µs |
| Ring buffer write+read (1024 B) | 0.92 µs | 0.047 µs |

Both are zero-managed-allocation. Reproduce with:

```bash
dotnet run --project tools/audio_bench -c Release
```

## Known Limitations

- The ring buffer remains SPSC: the write side and read side must each be single-threaded (same as the managed implementation).
- On insufficient capacity, the entire block is dropped and 0 is returned without blocking the producer; callers (such as CLI playback) must apply their own rate limiting.
- When the version changes, `NativeAudio.ExpectedVersion` must be updated as well; otherwise the library is treated as incompatible and the managed path is used.
