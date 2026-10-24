# rust_probe

Measures whether a hand-written INT8 matvec kernel can beat ONNX Runtime's, at the
decoder's exact weight shapes (97 matmuls, 72.5 MiB of int8).

Result on Ryzen 5 5500: 4.89 ms single-threaded (14.5 GB/s) vs ORT's 7.4 ms
(9.8 GB/s) for the whole step — but the matvecs are only 66% of an ORT step, so a
full Rust decoder lands at 1.05–1.3×, not the 1.5× the kernel alone suggests.
More threads make it slower.

```bash
cargo run --release --offline
```

See `docs/DECODER_BASELINE.md`, "Rust rewrite: measured, and the ceiling is ~1.5×".
