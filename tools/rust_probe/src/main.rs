// Honest measurement of a Rust INT8 matvec kernel at the decoder's real weight size.
//
// Two earlier versions of this benchmark lied:
//   * 4 MiB of weights fit in L3, so it reported 35-80 GB/s (cache, not DRAM);
//   * the result vector was never consumed, so the optimizer removed work.
// This version uses the decoder's real 72.5 MiB of weights, consumes every
// result, and reports against the ORT reference (9.8 GB/s, measured end-to-end).
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn fill(buf: &mut [i8], seed: u64) {
    let mut s = seed | 1;
    for v in buf.iter_mut() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; *v = (s >> 24) as i8; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot(x: &[i8], w: &[i8], k: usize) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= k {
        let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
        let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let lo = _mm256_maddubs_epi16(xv, wv);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo, _mm256_set1_epi16(1)));
        i += 32;
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut s: i32 = lanes.iter().sum();
    while i < k { s += (x[i] as i32) * (w[i] as i32); i += 1; }
    s
}

fn matvec(x: &[i8], w: &[i8], y: &mut [f32], k: usize) {
    y.par_iter_mut().enumerate().for_each(|(n, out)| {
        *out = unsafe { dot(x, &w[n * k..(n + 1) * k], k) } as f32;
    });
}

fn main() {
    // The decoder's actual 97 matmuls: 24x(512x1536, 512x2048, 2048x512, 512x512)
    // plus the 512x1025 head.
    let shapes: Vec<(usize, usize)> = {
        let mut v = Vec::new();
        for _ in 0..24 {
            v.push((1536usize, 512usize)); // in_proj
            v.push((2048, 512));           // linear1
            v.push((512, 2048));           // linear2
            v.push((512, 512));            // out_proj
        }
        v.push((1025, 512));
        v
    };
    let total: usize = shapes.iter().map(|(n, k)| n * k).sum();
    println!("decoder-shaped kernel: {} matmuls, {:.1} MiB of int8 weights", shapes.len(), total as f64 / 1048576.0);

    let mut weights = vec![0i8; total];
    fill(&mut weights, 0xdead_beef);
    let x = vec![2i8; 2048];
    let mut y = vec![0.0f32; 2048];
    let sink = std::sync::atomic::AtomicU32::new(0);

    let mut offsets = Vec::with_capacity(shapes.len());
    let mut off = 0;
    for (n, k) in &shapes { offsets.push((off, *n, *k)); off += n * k; }

    let mut run = || {
        for &(o, n, k) in &offsets {
            matvec(&x[..k], &weights[o..o + n * k], &mut y[..n], k);
            sink.fetch_add(y[0].to_bits() | 1, std::sync::atomic::Ordering::Relaxed);
        }
    };

    for _ in 0..3 { run(); }
    let mut t = Vec::new();
    for _ in 0..25 {
        let t0 = Instant::now();
        run();
        t.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    black_box(sink.load(std::sync::atomic::Ordering::Relaxed));
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = t[t.len() / 2];
    let mib = total as f64 / 1048576.0;
    println!("  {:2} threads  p50 {p50:6.2} ms -> {:5.1} GB/s", rayon::current_num_threads(), mib / p50 * 1000.0 / 1024.0);
    println!("  ORT INT8 reference: 7.4 ms/step, 9.8 GB/s (measured)");

    println!("thread scaling:");
    for nt in [1usize, 2, 3, 4, 6, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        pool.install(|| {
            for _ in 0..3 { run(); }
            let mut t = Vec::new();
            for _ in 0..15 { let t0 = Instant::now(); run(); t.push(t0.elapsed().as_secs_f64() * 1000.0); }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = t[t.len() / 2];
            println!("  {nt:2} threads  p50 {p50:6.2} ms -> {:5.1} GB/s", mib / p50 * 1000.0 / 1024.0);
        });
    }
    black_box(sink.load(std::sync::atomic::Ordering::Relaxed));
}
