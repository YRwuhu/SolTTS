# third_party

## onnxruntime_ggml.dll

The ggml plugin execution provider (Windows x64) from
[onnxruntime-ggml](https://github.com/thewh1teagle/onnxruntime-ggml) v0.2.0. It is required for the vocoder hybrid mode
(`UseVocoderGgml: true`); if missing, that mode falls back to CPU automatically without affecting other features.

How to obtain it (either option):

1. Download from GitHub Releases and extract into this directory:
   ```
   https://github.com/thewh1teagle/onnxruntime-ggml/releases/download/v0.2.0/onnxruntime-ggml-0.2.0-windows-amd64.tar.gz
   ```
   Place the resulting `onnxruntime_ggml.dll` in this directory (it is copied to the output directory during the build).

2. Skip the build integration and point to the DLL at runtime with an environment variable:
   ```
   set SOLTTS_GGML_LIBRARY=D:\path\to\onnxruntime_ggml.dll
   ```

For other platforms (Linux/macOS), download the matching package from the same Releases page;
the files are named `libonnxruntime_ggml.so` / `libonnxruntime_ggml.dylib`.
