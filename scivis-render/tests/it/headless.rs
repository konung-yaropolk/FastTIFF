//! A GPU to test on, with nothing on screen: an OpenGL context for the glow
//! backend, and a wgpu device for the other.
//!
//! Either may be missing, and a test that needs one skips when it is: a CI
//! runner has no GPU, and the GL a bare Windows offers (opengl32.dll's own,
//! GL 1.1) cannot compile GLSL 1.40. The GL context is made the way each
//! platform allows without a window on screen:
//!
//! * Windows: WGL on a hidden window, asking for a 3.3 core profile, the
//!   strictest context `#version 140` compiles in, so that a program that
//!   relies on what only a compatibility context allows fails here too. A
//!   driver that will not make one leaves the compatibility context WGL makes
//!   first, and the tests say so.
//! * Linux: EGL with no surface at all, on Mesa's surfaceless platform (what
//!   a machine with no display server has; software llvmpipe where there is
//!   no GPU), else the default display.
//! * elsewhere: none. The glow backend is the Linux release renderer, and
//!   macOS ships with wgpu.
//!
//! What GPU a test ran on, and why one skipped, is written past libtest's
//! capture ([`say`]), so that it reaches the log of a run that passes: a
//! skipped test passes too. Where a GL context is expected, set
//! [`REQUIRE_GL`] and failing to make one fails the test instead.
//!
//! Each half builds with the backend it is for, so that a copy of this crate
//! built with one backend, or none, still builds its tests (see `main.rs`).

#![cfg(not(target_arch = "wasm32"))]

use std::sync::{Mutex, MutexGuard};

/// Held by every test that drives a GPU, for as long as it does. Tests run on
/// threads of their own; this keeps to one GL context at a time and keeps GL
/// work and wgpu work from running at once on the same card, a combination
/// the app never makes (it draws with one backend) and nothing here measures.
pub fn exclusive() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Writes `what` as a line of the test process's own stderr. libtest
/// captures what `eprintln!` writes and prints it only for a test that
/// fails, so a skip said that way reads, in a CI log, as a pass; a write to
/// the stream itself is not captured.
pub fn say(what: std::fmt::Arguments<'_>) {
    use std::io::Write as _;
    let _ = writeln!(std::io::stderr().lock(), "scivis-render tests: {what}");
}

/// The environment variable that, set to anything but `0`, makes a machine
/// that cannot make a GL context fail the tests that need one rather than
/// skip them. The Linux CI job sets it, having installed Mesa for these
/// tests: there a skip would mean that the only run of the glow programs on
/// a real driver in CI had quietly stopped.
#[cfg(feature = "backend-glow")]
pub const REQUIRE_GL: &str = "SCIVIS_RENDER_REQUIRE_GL";

/// An OpenGL context, current on the thread that made it until dropped.
#[cfg(feature = "backend-glow")]
pub struct Gl {
    // Dropped in this order: the functions, then the context they call into.
    pub gl: std::sync::Arc<glow::Context>,
    _context: platform::Context,
}

/// A GL 3.1 or later context for this thread, or `None` (having said why)
/// when the machine cannot make one and [`REQUIRE_GL`] is not set.
#[cfg(feature = "backend-glow")]
pub fn gl() -> Option<Gl> {
    use glow::HasContext as _;
    let unavailable = |why: String| {
        if std::env::var_os(REQUIRE_GL).is_some_and(|v| v != "0") {
            panic!(
                "{REQUIRE_GL} is set, but this machine made no OpenGL context to test on: {why}"
            );
        }
        say(format_args!("no OpenGL context ({why}); skipping"));
        None
    };
    let context = match platform::Context::new() {
        Ok(context) => context,
        Err(why) => return unavailable(why),
    };
    // SAFETY: the context is current on this thread and stays so until `Gl`
    // drops, after the functions loaded from it.
    let gl = unsafe { glow::Context::from_loader_function_cstr(|name| context.proc_address(name)) };
    // SAFETY: as above.
    let (renderer, version) = unsafe {
        (
            gl.get_parameter_string(glow::RENDERER),
            gl.get_parameter_string(glow::VERSION),
        )
    };
    let v = gl.version();
    if v.is_embedded || (v.major, v.minor) < (3, 1) {
        return unavailable(format!(
            "OpenGL {version} on {renderer}, which cannot compile GLSL 1.40"
        ));
    }
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| say(format_args!("glow on OpenGL {version}, {renderer}")));
    Some(Gl {
        gl: std::sync::Arc::new(gl),
        _context: context,
    })
}

/// What `draw` draws into a `width` x `height` RGBA8 framebuffer made for
/// it, with the viewport set to all of it: the pixels, rows top to bottom.
/// Dithering is off, as wgpu has none to compare with. The framebuffer is
/// gone afterwards, and the default one bound again.
#[cfg(feature = "backend-glow")]
pub fn draw_offscreen(
    gl: &glow::Context,
    (width, height): (u32, u32),
    draw: impl FnOnce(),
) -> Vec<u8> {
    use glow::HasContext as _;
    // SAFETY: a context is current on this thread (`gl` was loaded from it);
    // every object made here is deleted here.
    unsafe {
        let texture = gl.create_texture().expect("texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        let fb = gl.create_framebuffer().expect("framebuffer");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fb));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );
        assert_eq!(
            gl.check_framebuffer_status(glow::FRAMEBUFFER),
            glow::FRAMEBUFFER_COMPLETE
        );
        gl.viewport(0, 0, width as i32, height as i32);
        gl.disable(glow::DITHER);
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.clear(glow::COLOR_BUFFER_BIT);
        draw();
        let mut bottom_up = vec![0u8; (width * height * 4) as usize];
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
        gl.read_pixels(
            0,
            0,
            width as i32,
            height as i32,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut bottom_up)),
        );
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.delete_framebuffer(fb);
        gl.delete_texture(texture);
        bottom_up
            .chunks(width as usize * 4)
            .rev()
            .flatten()
            .copied()
            .collect()
    }
}

/// That `gl` has recorded no error since it was last asked, `what` being
/// what ran in between. GL reports a call it refused (a uniform set with the
/// wrong function for its type, say) only here, and draws on as if the call
/// had not been made.
#[cfg(feature = "backend-glow")]
pub fn assert_no_error(gl: &glow::Context, what: &str) {
    use glow::HasContext as _;
    // SAFETY: a context is current on this thread.
    let error = unsafe { gl.get_error() };
    let name = match error {
        glow::NO_ERROR => return,
        glow::INVALID_ENUM => "GL_INVALID_ENUM",
        glow::INVALID_VALUE => "GL_INVALID_VALUE",
        glow::INVALID_OPERATION => "GL_INVALID_OPERATION",
        glow::INVALID_FRAMEBUFFER_OPERATION => "GL_INVALID_FRAMEBUFFER_OPERATION",
        glow::OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
        _ => "an error",
    };
    panic!("{what}: the driver reported {name} (0x{error:04X})");
}

/// The wgpu device the tests share.
#[cfg(feature = "backend-wgpu")]
pub struct Wgpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// The one wgpu device of this test process, opened the first time it is
/// asked for: several opened side by side in one process hang on the drivers
/// this workspace was measured on. `None` (having said why) without an
/// adapter, or with only a software one: the comparisons are of what GPUs
/// draw, and their tolerances were measured on one.
///
/// Vulkan, DX12 or Metal, never wgpu's own GL backend: looking for a GL
/// adapter makes a GL context of wgpu's current on the calling thread, and a
/// test's own context is then no longer current. Ask for this before
/// [`gl`] all the same, as no backend promises to leave the thread alone.
#[cfg(feature = "backend-wgpu")]
pub fn wgpu() -> Option<&'static Wgpu> {
    static DEVICE: std::sync::OnceLock<Option<Wgpu>> = std::sync::OnceLock::new();
    let device = DEVICE.get_or_init(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("scivis-render tests"),
            required_features: scivis_render::wgpu_backend::optional_features(&adapter),
            ..Default::default()
        }))
        .ok()?;
        say(format_args!("wgpu on {} ({:?})", info.name, info.backend));
        Some(Wgpu { device, queue })
    });
    if device.is_none() {
        say(format_args!("no hardware wgpu adapter; skipping"));
    }
    device.as_ref()
}

#[cfg(all(feature = "backend-glow", windows))]
mod platform {
    use std::ffi::{c_void, CStr};
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::Graphics::Gdi::{GetDC, ReleaseDC, HDC};
    use windows_sys::Win32::Graphics::OpenGL::{
        wglCreateContext, wglDeleteContext, wglGetProcAddress, wglMakeCurrent, ChoosePixelFormat,
        SetPixelFormat, HGLRC, PFD_DOUBLEBUFFER, PFD_DRAW_TO_WINDOW, PFD_MAIN_PLANE,
        PFD_SUPPORT_OPENGL, PFD_TYPE_RGBA, PIXELFORMATDESCRIPTOR,
    };
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleHandleW, GetProcAddress, LoadLibraryA,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, CS_OWNDC, WNDCLASSW,
        WS_OVERLAPPEDWINDOW,
    };

    // WGL_ARB_create_context and WGL_ARB_create_context_profile.
    const WGL_CONTEXT_MAJOR_VERSION_ARB: i32 = 0x2091;
    const WGL_CONTEXT_MINOR_VERSION_ARB: i32 = 0x2092;
    const WGL_CONTEXT_PROFILE_MASK_ARB: i32 = 0x9126;
    const WGL_CONTEXT_CORE_PROFILE_BIT_ARB: i32 = 0x1;
    type CreateContextAttribs = unsafe extern "system" fn(HDC, HGLRC, *const i32) -> HGLRC;

    /// A WGL context on a window that is never shown.
    pub struct Context {
        window: HWND,
        dc: HDC,
        context: HGLRC,
    }

    impl Context {
        pub fn new() -> Result<Context, String> {
            // SAFETY: plain Win32 calls on handles made here; every one is
            // checked, and what was made is released by `drop`.
            unsafe {
                let class: Vec<u16> = "scivis-render-test\0".encode_utf16().collect();
                let instance = GetModuleHandleW(null());
                let wc = WNDCLASSW {
                    style: CS_OWNDC,
                    lpfnWndProc: Some(DefWindowProcW),
                    hInstance: instance,
                    lpszClassName: class.as_ptr(),
                    ..Default::default()
                };
                // A second registration fails, as the class is there already.
                RegisterClassW(&wc);
                let window = CreateWindowExW(
                    0,
                    class.as_ptr(),
                    class.as_ptr(),
                    WS_OVERLAPPEDWINDOW,
                    0,
                    0,
                    16,
                    16,
                    null_mut(),
                    null_mut(),
                    instance,
                    null(),
                );
                if window.is_null() {
                    return Err("CreateWindowExW failed".into());
                }
                let mut this = Context {
                    window,
                    dc: GetDC(window),
                    context: null_mut(),
                };
                let mut pfd: PIXELFORMATDESCRIPTOR = std::mem::zeroed();
                pfd.nSize = size_of::<PIXELFORMATDESCRIPTOR>() as u16;
                pfd.nVersion = 1;
                pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
                pfd.iPixelType = PFD_TYPE_RGBA;
                pfd.cColorBits = 32;
                pfd.cAlphaBits = 8;
                pfd.iLayerType = PFD_MAIN_PLANE as u8;
                let format = ChoosePixelFormat(this.dc, &pfd);
                if format == 0 || SetPixelFormat(this.dc, format, &pfd) == 0 {
                    return Err("no OpenGL pixel format".into());
                }
                // A legacy context first: the function that makes a core one
                // can only be looked up with a context current.
                let legacy = wglCreateContext(this.dc);
                if legacy.is_null() {
                    return Err("wglCreateContext failed".into());
                }
                this.context = legacy;
                if wglMakeCurrent(this.dc, legacy) == 0 {
                    return Err("wglMakeCurrent failed".into());
                }
                let create = wglGetProcAddress(c"wglCreateContextAttribsARB".as_ptr().cast());
                let Some(create) = create else {
                    // opengl32.dll's own GL 1.1, with no driver behind it.
                    return Ok(this);
                };
                let create: CreateContextAttribs = std::mem::transmute(create);
                let attribs = [
                    WGL_CONTEXT_MAJOR_VERSION_ARB,
                    3,
                    WGL_CONTEXT_MINOR_VERSION_ARB,
                    3,
                    WGL_CONTEXT_PROFILE_MASK_ARB,
                    WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                    0,
                ];
                let core = create(this.dc, null_mut(), attribs.as_ptr());
                if !core.is_null() && wglMakeCurrent(this.dc, core) != 0 {
                    wglDeleteContext(legacy);
                    this.context = core;
                    return Ok(this);
                }
                // A failed wglMakeCurrent leaves no context current at all.
                if !core.is_null() {
                    wglDeleteContext(core);
                }
                if wglMakeCurrent(this.dc, legacy) == 0 {
                    return Err("wglMakeCurrent failed".into());
                }
                super::say(format_args!(
                    "the driver made no OpenGL 3.3 core context; testing on the \
                     compatibility context instead, which allows more than core does"
                ));
                Ok(this)
            }
        }

        /// A GL function of the current context: `wglGetProcAddress` knows
        /// those past GL 1.1, and opengl32.dll exports the rest itself.
        pub fn proc_address(&self, name: &CStr) -> *const c_void {
            // SAFETY: lookups by a NUL-terminated name.
            unsafe {
                let found = wglGetProcAddress(name.as_ptr().cast()).map_or(0, |f| f as usize);
                // Some drivers answer 1, 2, 3 or -1 for "not mine".
                if !matches!(found, 0 | 1 | 2 | 3 | usize::MAX) {
                    return found as *const c_void;
                }
                let opengl32 = LoadLibraryA(c"opengl32.dll".as_ptr().cast());
                GetProcAddress(opengl32, name.as_ptr().cast())
                    .map_or(null(), |f| f as *const c_void)
            }
        }
    }

    impl Drop for Context {
        fn drop(&mut self) {
            // SAFETY: the handles `new` made, on the thread that made them.
            unsafe {
                wglMakeCurrent(null_mut(), null_mut());
                if !self.context.is_null() {
                    wglDeleteContext(self.context);
                }
                ReleaseDC(self.window, self.dc);
                DestroyWindow(self.window);
            }
        }
    }
}

#[cfg(all(feature = "backend-glow", target_os = "linux"))]
mod platform {
    use khronos_egl as egl;
    use std::ffi::{c_void, CStr};

    /// `EGL_MESA_platform_surfaceless`: a display with no display server.
    const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

    /// An EGL context with no surface, current on this thread.
    pub struct Context {
        egl: egl::DynamicInstance<egl::EGL1_4>,
        display: egl::Display,
        context: egl::Context,
    }

    impl Context {
        pub fn new() -> Result<Context, String> {
            // SAFETY: loads the system's libEGL, whose functions khronos-egl
            // declares as the specification does.
            let egl = unsafe { egl::DynamicInstance::<egl::EGL1_4>::load_required() }
                .map_err(|e| format!("no libEGL: {e}"))?;
            let surfaceless = egl.upcast::<egl::EGL1_5>().and_then(|egl| {
                // SAFETY: a platform that takes no native display.
                unsafe {
                    egl.get_platform_display(
                        PLATFORM_SURFACELESS_MESA,
                        egl::DEFAULT_DISPLAY,
                        &[egl::ATTRIB_NONE],
                    )
                }
                .ok()
            });
            // SAFETY: the default display needs no native handle either.
            let display = surfaceless
                .or_else(|| unsafe { egl.get_display(egl::DEFAULT_DISPLAY) })
                .ok_or("no EGL display")?;
            egl.initialize(display)
                .map_err(|e| format!("eglInitialize: {e}"))?;
            let made = (|| {
                egl.bind_api(egl::OPENGL_API)
                    .map_err(|e| format!("no desktop OpenGL through EGL: {e}"))?;
                let config = egl
                    .choose_first_config(
                        display,
                        &[
                            egl::RENDERABLE_TYPE,
                            egl::OPENGL_BIT,
                            egl::SURFACE_TYPE,
                            egl::PBUFFER_BIT,
                            egl::NONE,
                        ],
                    )
                    .map_err(|e| format!("eglChooseConfig: {e}"))?
                    .ok_or("no EGL config for desktop OpenGL")?;
                let context = egl
                    .create_context(
                        display,
                        config,
                        None,
                        &[
                            egl::CONTEXT_MAJOR_VERSION,
                            3,
                            egl::CONTEXT_MINOR_VERSION,
                            3,
                            egl::CONTEXT_OPENGL_PROFILE_MASK,
                            egl::CONTEXT_OPENGL_CORE_PROFILE_BIT,
                            egl::NONE,
                        ],
                    )
                    .map_err(|e| format!("no GL 3.3 core context: {e}"))?;
                // With no surface: EGL_KHR_surfaceless_context.
                if let Err(e) = egl.make_current(display, None, None, Some(context)) {
                    let _ = egl.destroy_context(display, context);
                    return Err(format!(
                        "cannot make a context current without a surface: {e}"
                    ));
                }
                Ok(context)
            })();
            match made {
                Ok(context) => Ok(Context {
                    egl,
                    display,
                    context,
                }),
                Err(why) => {
                    let _ = egl.terminate(display);
                    Err(why)
                }
            }
        }

        pub fn proc_address(&self, name: &CStr) -> *const c_void {
            name.to_str()
                .ok()
                .and_then(|name| self.egl.get_proc_address(name))
                .map_or(std::ptr::null(), |f| f as *const c_void)
        }
    }

    impl Drop for Context {
        fn drop(&mut self) {
            let _ = self.egl.make_current(self.display, None, None, None);
            let _ = self.egl.destroy_context(self.display, self.context);
            let _ = self.egl.terminate(self.display);
        }
    }
}

#[cfg(all(feature = "backend-glow", not(any(windows, target_os = "linux"))))]
mod platform {
    use std::ffi::{c_void, CStr};

    pub struct Context;

    impl Context {
        pub fn new() -> Result<Context, String> {
            Err("these tests make a GL context on Windows and Linux only".into())
        }

        pub fn proc_address(&self, _: &CStr) -> *const c_void {
            std::ptr::null()
        }
    }
}
