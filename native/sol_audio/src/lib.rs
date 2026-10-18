//! SolTTS native real-time audio primitives.
//!
//! Design constraints:
//! - Pure C ABI; panics never cross the FFI boundary (all exports wrap `catch_unwind` and return error codes).
//! - Every function is stateless or uses an explicit handle, with no global state, so multiple engine instances can use it concurrently.
//! - Bit-exact (or extremely low-error) equivalence with the managed implementation, enabling gradual replacement and gray-rollback.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

pub const SOL_OK: i32 = 0;
pub const SOL_ERR_NULL: i32 = -1;
pub const SOL_ERR_OVERFLOW: i32 = -2;
pub const SOL_ERR_PANIC: i32 = -3;

// ---------------------------------------------------------------------------
// SPSC ring buffer (byte stream)
// ---------------------------------------------------------------------------

/// Single-producer/single-consumer ring buffer.
///
/// Works in bytes, aligned with NAudio's `IWaveProvider.Read(byte[])` (16-bit PCM, etc.);
/// positions use free-running `u64` counters and indices are taken modulo capacity; each of
/// the read and write sides is called by only one thread, so no lock is needed. When capacity
/// is insufficient, the whole block is dropped and 0 is returned (same as the managed implementation).
pub struct RingBuffer {
    data: Box<[u8]>,
    /// Capacity is rounded up to a power of two so modulo can be done with a bit mask.
    mask: u64,
    write_pos: u64,
    read_pos: u64,
}

impl RingBuffer {
    fn new(capacity_bytes: usize) -> Self {
        let cap = capacity_bytes.next_power_of_two().max(2);
        Self {
            data: vec![0u8; cap].into_boxed_slice(),
            mask: (cap - 1) as u64,
            write_pos: 0,
            read_pos: 0,
        }
    }

    #[inline]
    fn available(&self) -> u64 {
        self.write_pos.wrapping_sub(self.read_pos)
    }

    fn write(&mut self, src: &[u8]) -> usize {
        let free = (self.mask + 1) - self.available().min(self.mask + 1);
        if src.len() as u64 > free {
            return 0; // Drop the whole block, matching LockFreeWaveProvider behavior
        }
        let start = (self.write_pos & self.mask) as usize;
        let first = src.len().min(self.data.len() - start);
        self.data[start..start + first].copy_from_slice(&src[..first]);
        if first < src.len() {
            self.data[..src.len() - first].copy_from_slice(&src[first..]);
        }
        // Release semantics: write data first, then publish the write position.
        let next = self.write_pos.wrapping_add(src.len() as u64);
        atomic_store_release(&mut self.write_pos, next);
        src.len()
    }

    fn read(&mut self, dst: &mut [u8]) -> usize {
        let available = self.available() as usize;
        let n = available.min(dst.len());
        if n == 0 {
            return 0;
        }
        let start = (self.read_pos & self.mask) as usize;
        let first = n.min(self.data.len() - start);
        dst[..first].copy_from_slice(&self.data[start..start + first]);
        if first < n {
            dst[first..n].copy_from_slice(&self.data[..n - first]);
        }
        let next = self.read_pos.wrapping_add(n as u64);
        atomic_store_release(&mut self.read_pos, next);
        n
    }

    fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.data.fill(0);
    }
}

/// Note: `RingBuffer` is passed across FFI as a raw pointer; the caller guarantees each of the producer/consumer is single-threaded.
unsafe impl Send for RingBuffer {}

#[inline]
fn atomic_store_release(slot: &mut u64, value: u64) {
    // Raw pointer + volatile write; on x86-64, stores already have release semantics.
    unsafe { std::ptr::write_volatile(slot as *mut u64, value) }
}

#[inline]
fn atomic_load_acquire(slot: &u64) -> u64 {
    unsafe { std::ptr::read_volatile(slot as *const u64) }
}

#[no_mangle]
pub extern "C" fn sol_ring_create(capacity_bytes: usize) -> *mut RingBuffer {
    let result = catch_unwind(|| Box::into_raw(Box::new(RingBuffer::new(capacity_bytes))));
    result.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn sol_ring_destroy(ring: *mut RingBuffer) {
    if ring.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(ring));
    }));
}

/// Writes bytes and returns the number actually written (returns 0 and drops the whole block if capacity is insufficient).
#[no_mangle]
pub extern "C" fn sol_ring_write(ring: *mut RingBuffer, src: *const u8, len: usize) -> usize {
    if ring.is_null() || (src.is_null() && len > 0) {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let rb = &mut *ring;
        let samples = slice::from_raw_parts(src, len);
        rb.write(samples)
    }));
    result.unwrap_or(0)
}

/// Reads bytes and returns the number actually read.
#[no_mangle]
pub extern "C" fn sol_ring_read(ring: *mut RingBuffer, dst: *mut u8, len: usize) -> usize {
    if ring.is_null() || (dst.is_null() && len > 0) {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let rb = &mut *ring;
        let out = slice::from_raw_parts_mut(dst, len);
        rb.read(out)
    }));
    result.unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn sol_ring_available(ring: *const RingBuffer) -> usize {
    if ring.is_null() {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let rb = &*ring;
        atomic_load_acquire(&rb.write_pos).wrapping_sub(atomic_load_acquire(&rb.read_pos)) as usize
    }));
    result.unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn sol_ring_clear(ring: *mut RingBuffer) {
    if ring.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        (*ring).clear();
    }));
}

// ---------------------------------------------------------------------------
// First-order high-pass filter (state passed explicitly to align with the managed HighPassFilter)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn sol_hpf_alpha(sample_rate: f32, cutoff_hz: f32) -> f32 {
    if sample_rate <= 0.0 || cutoff_hz <= 0.0 {
        return 0.0;
    }
    let rc = 1.0f32 / (2.0f32 * std::f32::consts::PI * cutoff_hz);
    let dt = 1.0f32 / sample_rate;
    rc / (rc + dt)
}

/// In-place high-pass filtering. `state` is `[prev_input, prev_output]`, updated in place.
///
/// The floating-point operation order must exactly match `HighPassFilter.Process`; tests require bit identity.
#[no_mangle]
pub extern "C" fn sol_hpf_process(
    samples: *mut f32,
    len: usize,
    alpha: f32,
    state: *mut f32,
) -> i32 {
    if (samples.is_null() && len > 0) || state.is_null() {
        return SOL_ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let buf = slice::from_raw_parts_mut(samples, len);
        let st = slice::from_raw_parts_mut(state, 2);
        let mut prev_input = st[0];
        let mut prev_output = st[1];
        for x in buf.iter_mut() {
            let input = *x;
            let output = alpha * (prev_output + input - prev_input);
            prev_input = input;
            prev_output = output;
            *x = output;
        }
        st[0] = prev_input;
        st[1] = prev_output;
    }));
    match result {
        Ok(()) => SOL_OK,
        Err(_) => SOL_ERR_PANIC,
    }
}

// ---------------------------------------------------------------------------
// Crossfade
// ---------------------------------------------------------------------------

/// Fades out `src[..n]` and fades in `dst[..n]`: dst = src*(1-a) + dst*a.
///
/// Equivalent to `input[i] = prevTail[i]*(1-alpha) + input[i]*alpha`
/// in `AudioPipeline.ProcessChunk` (dst=input, src=prevTail).
#[no_mangle]
pub extern "C" fn sol_crossfade(dst: *mut f32, src: *const f32, n: usize, denom: usize) -> i32 {
    if n == 0 {
        return SOL_OK;
    }
    if dst.is_null() || src.is_null() || denom == 0 {
        return SOL_ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let out = slice::from_raw_parts_mut(dst, n);
        let inp = slice::from_raw_parts(src, n);
        let denom = denom as f32;
        for i in 0..n {
            let alpha = i as f32 / denom;
            out[i] = inp[i] * (1.0 - alpha) + out[i] * alpha;
        }
    }));
    match result {
        Ok(()) => SOL_OK,
        Err(_) => SOL_ERR_PANIC,
    }
}

/// Applies a linear fade-out over `samples[start..]`, matching the managed isFinal branch.
#[no_mangle]
pub extern "C" fn sol_fade_out(samples: *mut f32, start: usize, n: usize, denom: usize) -> i32 {
    if n == 0 {
        return SOL_OK;
    }
    if samples.is_null() || denom == 0 {
        return SOL_ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let buf = slice::from_raw_parts_mut(samples.add(start), n);
        let denom = denom as f32;
        for i in 0..n {
            let factor = 1.0 - i as f32 / denom;
            buf[i] *= factor;
        }
    }));
    match result {
        Ok(()) => SOL_OK,
        Err(_) => SOL_ERR_PANIC,
    }
}

// ---------------------------------------------------------------------------
// float -> PCM16
// ---------------------------------------------------------------------------

/// Converts float to 16-bit PCM: `out[i] = (short)(clamp(x,-1,1) * 32767)`.
///
/// Truncation/rounding must match the C# `(short)` conversion (truncation toward zero).
#[no_mangle]
pub extern "C" fn sol_f32_to_i16(src: *const f32, dst: *mut i16, len: usize) -> i32 {
    if len == 0 {
        return SOL_OK;
    }
    if src.is_null() || dst.is_null() {
        return SOL_ERR_NULL;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let input = slice::from_raw_parts(src, len);
        let output = slice::from_raw_parts_mut(dst, len);
        for i in 0..len {
            output[i] = (input[i].clamp(-1.0, 1.0) * 32767.0) as i16;
        }
    }));
    match result {
        Ok(()) => SOL_OK,
        Err(_) => SOL_ERR_PANIC,
    }
}

/// Returns the version for the managed side to verify that the DLL and bindings match.
#[no_mangle]
pub extern "C" fn sol_audio_version() -> u32 {
    (env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap_or(0) << 16)
        | (env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap_or(0) << 8)
        | env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap_or(0)
}
