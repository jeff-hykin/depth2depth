//! Depth Anything through TensorRT (feature `tensorrt`), for Jetsons, where candle's CUDA
//! path is bound by kernel launches: on an Orin the same 364x448 frame took 190 ms in candle.
//!
//! TensorRT builds an engine from the ONNX export once (minutes on an Orin) and caches it
//! at `engine_path`; an engine is tied to the GPU and the TensorRT version, and one that no
//! longer loads is rebuilt.

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::{Arc, Mutex, Once, Weak};

extern "C" {
    fn d2d_trt_version() -> c_int;
    fn d2d_trt_open(
        onnx: *const c_void,
        onnx_size: usize,
        prebuilt: *const c_void,
        prebuilt_size: usize,
        engine_path: *const c_char,
        error: *mut c_char,
        error_length: usize,
    ) -> *mut c_void;
    fn d2d_trt_input_size(handle: *mut c_void, height: *mut c_int, width: *mut c_int);
    fn d2d_trt_infer(handle: *mut c_void, input: *const f32, output: *mut f32) -> c_int;
    fn d2d_trt_close(handle: *mut c_void);
    fn atexit(callback: extern "C" fn()) -> c_int;
}

struct Handle(*mut c_void);

// The engine is only touched under the mutex, one inference at a time.
unsafe impl Send for Handle {}

/// Every open engine, for `close_all_at_exit`.
static OPEN: Mutex<Vec<Weak<Mutex<Handle>>>> = Mutex::new(Vec::new());
static REGISTER_AT_EXIT: Once = Once::new();

/// A process that exits while some thread still owns an engine never drops it, and TensorRT then tears it down
/// after CUDA has unloaded, logging an error. Exit handlers run newest first, so this one, registered once CUDA is
/// up, frees every engine still open while CUDA is still there (after any inference in flight).
extern "C" fn close_all_at_exit() {
    let open = std::mem::take(&mut *OPEN.lock().unwrap_or_else(|e| e.into_inner()));
    for engine in open.iter().filter_map(Weak::upgrade) {
        let mut handle = engine.lock().unwrap_or_else(|e| e.into_inner());
        if !handle.0.is_null() {
            unsafe { d2d_trt_close(handle.0) };
            handle.0 = std::ptr::null_mut();
        }
    }
}

pub struct TrtDepth {
    handle: Arc<Mutex<Handle>>,
    pub height: usize,
    pub width: usize,
}

impl TrtDepth {
    /// `prebuilt` (a serialized engine; empty for none) when it loads here, else the engine cached at
    /// `engine_path`, built from `onnx` (the export's bytes) when that is missing or stale.
    pub fn open(onnx: &[u8], prebuilt: &[u8], engine_path: &str) -> Result<Self, String> {
        let engine = CString::new(engine_path).map_err(|e| e.to_string())?;
        let mut error = vec![0 as c_char; 512];
        let handle = unsafe {
            d2d_trt_open(
                onnx.as_ptr().cast(),
                onnx.len(),
                prebuilt.as_ptr().cast(),
                prebuilt.len(),
                engine.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if handle.is_null() {
            let message = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) };
            return Err(message.to_string_lossy().into_owned());
        }
        let (mut height, mut width) = (0, 0);
        unsafe { d2d_trt_input_size(handle, &mut height, &mut width) };
        let handle = Arc::new(Mutex::new(Handle(handle)));
        let mut open = OPEN.lock().unwrap();
        open.retain(|engine| engine.strong_count() > 0);
        open.push(Arc::downgrade(&handle));
        drop(open);
        REGISTER_AT_EXIT.call_once(|| unsafe {
            atexit(close_all_at_exit);
        });
        Ok(Self {
            handle,
            height: height as usize,
            width: width as usize,
        })
    }

    /// `input` is the normalised 3xHxW image at the engine's size; returns HxW meters.
    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(
            input.len(),
            3 * self.height * self.width,
            "input must be 3xHxW at the engine's size"
        );
        let mut output = vec![0f32; self.height * self.width];
        let handle = self.handle.lock().unwrap();
        if handle.0.is_null() {
            return Err("TensorRT engine already closed (the process is exiting)".into());
        }
        match unsafe { d2d_trt_infer(handle.0, input.as_ptr(), output.as_mut_ptr()) } {
            0 => Ok(output),
            code => Err(format!("TensorRT inference failed (step {code})")),
        }
    }
}

/// The linked TensorRT's version, as major*10000 + minor*100 + patch.
pub fn version() -> i32 {
    unsafe { d2d_trt_version() }
}

impl Drop for TrtDepth {
    fn drop(&mut self) {
        let mut handle = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        if !handle.0.is_null() {
            unsafe { d2d_trt_close(handle.0) };
            handle.0 = std::ptr::null_mut();
        }
    }
}
