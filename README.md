# SolTTS

<p align="center">
  <img src="images/banner.png" alt="SolTTS Banner" width="400">
</p>

<p align="center">
  Local text-to-speech with voice cloning, streaming playback and mixed-language support.<br>
  Runs on ONNX Runtime and .NET. No Python required at runtime.
</p>

<p align="center">
  <a href="README_zh.md">中文</a> ·
  <a href="https://github.com/YRwuhu/SolTTS/releases">Downloads</a>
</p>

## Usage

Download the Windows package from [Releases](https://github.com/YRwuhu/SolTTS/releases), extract it, and keep `resources/` next to the executables. No .NET installation is needed.

Run `sol-server.exe` and open <http://localhost:5000> for the WebUI, or use the CLI:

```bash
sol-cli.exe -O out.wav -- "今天天气真好。"
```

The WebUI also handles voice management, model conversion and settings. On Linux, run `./init-env.sh` first if you need the model converter. macOS supports the V1 engine only.

The default V1 engine supports Chinese/English and Chinese/Japanese, plus experimental [emotion control](docs/EMOTION_CONTROL.md). V2 is incomplete and has no bundled models.

## Build

Requires .NET 10 SDK.

```bash
dotnet build SolTTS.sln -c Release

./publish.ps1        # Windows
./publish-linux.sh   # Linux
./publish-mac.sh     # macOS
```

See [`config.template.yaml`](config.template.yaml) for configuration options. The optional Rust audio library has its own [build instructions](native/sol_audio/README.md).

## Performance

On a Ryzen 5 5500 and Radeon RX 5500 XT, a short sentence takes about 2 seconds to synthesize. Streaming playback starts in about 0.55 seconds, with a cold-start model load of around 7 seconds.

These figures use parallel sentence decoding and the GPU vocoder. Benchmarks, implementation notes and less successful experiments are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## License

MIT. Original work © 2026 Blackwood416; fork modifications © 2026 YRwuhu.

Based on [AstraTTS](https://github.com/Blackwood416/AstraTTS), with architecture and algorithms from [Genie-TTS](https://github.com/High-Logic/Genie-TTS) and [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).
