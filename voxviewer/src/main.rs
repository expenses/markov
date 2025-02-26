use dolly::prelude::*;
use egui_winit::egui;
use winit::{event_loop::EventLoop, window::Window};

mod app;
mod gpu_resources;
mod resource_loading;
use app::App;
use gpu_resources::Pipelines;
use resource_loading::load_resource_bytes;

const USE_SPIRV_SHADER: bool = false;

enum PromizeResult {
    Cancelled,
    Load(tree64::Tree64<PackedMaterial>),
    Saved,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        //backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: if USE_SPIRV_SHADER {
                    wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                } else {
                    Default::default()
                },
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0].add_srgb_suffix();

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    config.view_formats = vec![swapchain_format];
    config.present_mode = wgpu::PresentMode::AutoVsync;
    surface.configure(&device, &config);

    log::info!("{:?}\n{:?}", &config, &swapchain_capabilities.formats);

    let pipelines = Pipelines::new(&device, &queue, swapchain_format).await;

    let tree64 = tree64::Tree64::deserialize(std::io::Cursor::new(
        load_resource_bytes("sponza.tree64").await,
    ))
    .unwrap();

    queue.write_buffer(
        &pipelines.tree_nodes,
        0,
        bytemuck::cast_slice(&tree64.nodes.inner),
    );
    queue.write_buffer(
        &pipelines.leaf_data,
        0,
        bytemuck::cast_slice(&tree64.data.inner),
    );

    let mut app = App {
        cached_textures: app::CachedTextures::new(&device),
        egui_renderer: egui_wgpu::Renderer::new(&device, swapchain_format, None, 1, false),
        egui_state: egui_winit::State::new(
            egui::Context::default(),
            egui::viewport::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        ),
        left_mouse_down: false,
        right_mouse_down: false,
        window: &window,
        config,
        surface,
        device,
        queue,
        accumulated_frame_index: 0,
        frame_index: 0,
        settings: Settings::default(),
        camera: CameraRig::builder()
            .with(Position::new(glam::Vec3::new(0.0, 0.0, 0.0)))
            .with(YawPitch::new().yaw_degrees(90.).pitch_degrees(-25.0))
            .with(Smooth::new_position_rotation(0.25, 0.25))
            .with(Arm::new(glam::Vec3::Z * 175.0))
            .build(),
        pipelines,
        tree64,
        hide_ui: false,
        padded_uniform_buffer: encase::UniformBuffer::new(vec![]),
        promise: None,
        create_material: Default::default(),
    };

    event_loop.run_app(&mut app).unwrap()
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowAttributes::default();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowAttributesExtWebSys;
        let canvas = wgpu::web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = event_loop.create_window(builder).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}

struct Settings {
    sun_long: f32,
    sun_lat: f32,
    enable_shadows: bool,
    sun_apparent_size: f32,
    accumulate_samples: bool,
    max_bounces: u32,
    background_colour: [f32; 3],
    sun_colour: [f32; 3],
    sun_strength: f32,
    background_strength: f32,
    vertical_fov: f32,
    show_heatmap: bool,
    edit_distance: f32,
    edit_size: f32,
    resolution_scaling: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sun_long: -90.0_f32,
            sun_lat: 85.0_f32,
            enable_shadows: true,
            sun_apparent_size: 1.0_f32,
            accumulate_samples: true,
            max_bounces: 1,
            background_colour: [0.1; 3],
            sun_colour: [1.0; 3],
            sun_strength: 20.0,
            background_strength: 1.0,
            vertical_fov: 45.0_f32,
            show_heatmap: false,
            edit_distance: 10.0,
            edit_size: 10.0,
            resolution_scaling: 0.5,
        }
    }
}

#[derive(encase::ShaderType)]
struct CameraUniforms {
    p_inv: glam::Mat4,
    view: glam::Mat4,
    view_inv: glam::Mat4,
    pos: glam::Vec3,
    forward: glam::Vec3,
    up: glam::Vec3,
    right: glam::Vec3,
}

#[derive(encase::ShaderType)]
struct SunUniforms {
    direction: glam::Vec3,
    emission: glam::Vec3,
    cosine_apparent_size: f32,
}

#[derive(encase::ShaderType)]
struct TreeUniforms {
    offset: glam::IVec3,
    num_levels: u32,
    root_node_index: u32,
}

#[derive(encase::ShaderType)]
struct Uniforms {
    camera: CameraUniforms,
    sun: SunUniforms,
    tree: TreeUniforms,
    background_colour: glam::Vec3,
    resolution: glam::UVec2,
    settings: i32,
    frame_index: u32,
    accumulated_frame_index: u32,
    max_bounces: u32,
}

#[derive(
    Clone,
    Copy,
    Default,
    PartialEq,
    Debug,
    Hash,
    PartialOrd,
    Ord,
    Eq,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
#[repr(C, packed)]
struct PackedMaterial {
    base_colour: [u8; 3],
    ty_and_aux_value: u8,
}
