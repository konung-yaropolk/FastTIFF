//! How much GPU memory there is to spend, asked of the driver.
//!
//! wgpu has no API for it, so this reaches under it (`as_hal`) to the native
//! one for the backend wgpu chose:
//!
//! * **Vulkan** — `VK_EXT_memory_budget`, which wgpu enables whenever the
//!   driver has it. The heap is the *largest* device-local one: NVIDIA also
//!   lists a small device-local window onto VRAM that the CPU can map (214 MiB
//!   on a Quadro P620), which is not where textures go. Only where wgpu builds
//!   its Vulkan backend by default (Windows, Linux, Android, FreeBSD): on Apple
//!   platforms it has none, and its types to ask through do not exist.
//! * **DX12** — DXGI's `QueryVideoMemoryInfo` on wgpu's own adapter, and the
//!   fallback for Vulkan on Windows when the extension is missing.
//! * **OpenGL** (the glow backend) — `GL_NVX_gpu_memory_info` or
//!   `GL_ATI_meminfo`; see the glow backend.
//! * Elsewhere (Metal, the browser): unknown, and the caller falls back to a
//!   fixed budget.
//!
//! A budget is the operating system's figure for what this process may use,
//! and usage is what it uses; both move as the app allocates. wgpu allocates
//! in large blocks, so usage moves in steps (256 MiB on the P620), and a fresh
//! device already uses a couple of hundred megabytes.

/// What the driver says this process may use of the GPU's own memory, and
/// how much of it it uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramBudget {
    pub budget: u64,
    pub usage: u64,
}

impl VramBudget {
    /// What is left: the budget less what is in use.
    pub fn headroom(&self) -> u64 {
        self.budget.saturating_sub(self.usage)
    }
}

/// The budget of the adapter wgpu is using, if the driver will say.
///
/// `None` for an integrated GPU (or a CPU one) whatever the driver says: its
/// "device-local" memory is the host's RAM, which the volume already takes
/// once on the CPU side, so a budget from it would let the two copies take
/// that memory twice over.
pub fn wgpu_budget(adapter: &wgpu::Adapter) -> Option<VramBudget> {
    let info = adapter.get_info();
    if !matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::VirtualGpu
    ) {
        return None;
    }
    match info.backend {
        #[cfg(windows)]
        wgpu::Backend::Dx12 => dx12_budget(adapter),
        #[cfg(any(
            windows,
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd"
        ))]
        wgpu::Backend::Vulkan => vulkan_budget(adapter).or_else(|| {
            #[cfg(windows)]
            {
                dxgi_budget(info.vendor, info.device)
            }
            #[cfg(not(windows))]
            {
                None
            }
        }),
        _ => None,
    }
}

/// DXGI's figures for wgpu's own DX12 adapter.
#[cfg(windows)]
fn dx12_budget(adapter: &wgpu::Adapter) -> Option<VramBudget> {
    use windows::Win32::Graphics::Dxgi::{
        IDXGIAdapter3, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    // SAFETY: the raw adapter is only read, and only while `hal` keeps it alive.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }?;
    let raw: &IDXGIAdapter3 = hal.as_raw();
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: a plain query into a struct we own.
    unsafe { raw.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.ok()?;
    Some(VramBudget {
        budget: info.Budget,
        usage: info.CurrentUsage,
    })
}

/// DXGI's figures for the adapter with this PCI vendor and device id, found
/// through a factory of our own (for a non-DX12 wgpu backend on Windows).
#[cfg(windows)]
fn dxgi_budget(vendor: u32, device: u32) -> Option<VramBudget> {
    use windows::core::Interface as _;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory1,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    // SAFETY: plain COM factory creation and enumeration; every object is
    // released when it drops.
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
    for index in 0.. {
        let adapter: IDXGIAdapter1 = unsafe { factory.EnumAdapters1(index) }.ok()?;
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if desc.VendorId != vendor || desc.DeviceId != device {
            continue;
        }
        // DXGI 1.4, Windows 10 and later.
        let adapter: IDXGIAdapter3 = adapter.cast().ok()?;
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
            .ok()?;
        return Some(VramBudget {
            budget: info.Budget,
            usage: info.CurrentUsage,
        });
    }
    None
}

/// `VK_EXT_memory_budget`'s figures for the largest device-local heap.
#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
fn vulkan_budget(adapter: &wgpu::Adapter) -> Option<VramBudget> {
    use ash::vk;
    // SAFETY: the raw handles are only read, while `hal` keeps them alive.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    if !hal
        .physical_device_capabilities()
        .supports_extension(ash::ext::memory_budget::NAME)
    {
        return None;
    }
    let shared = hal.shared_instance();
    let device = hal.raw_physical_device();
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    if shared
        .extensions()
        .contains(&ash::khr::get_physical_device_properties2::NAME)
    {
        let loader = ash::khr::get_physical_device_properties2::Instance::new(
            shared.entry(),
            shared.raw_instance(),
        );
        // SAFETY: a query into structs we own, chained as the spec requires.
        unsafe { loader.get_physical_device_memory_properties2(device, &mut props) };
    } else if shared.instance_api_version() >= vk::API_VERSION_1_1 {
        // SAFETY: as above, through the core 1.1 entry point.
        unsafe {
            shared
                .raw_instance()
                .get_physical_device_memory_properties2(device, &mut props)
        };
    } else {
        return None;
    }
    let memory = props.memory_properties;
    (0..memory.memory_heap_count as usize)
        .filter(|&i| {
            memory.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .max_by_key(|&i| memory.memory_heaps[i].size)
        .map(|i| VramBudget {
            budget: budget.heap_budget[i],
            usage: budget.heap_usage[i],
        })
}
