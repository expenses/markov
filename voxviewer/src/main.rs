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

fn copy_aligned(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    data: &[u8],
    range: std::ops::Range<usize>,
) {
    let aligned_range = std::ops::Range {
        start: range
            .start
            .saturating_sub(wgpu::COPY_BUFFER_ALIGNMENT as _)
            .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as _),

        end: range.end.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as _),
    };
    if aligned_range.end <= data.len() {
        queue.write_buffer(
            buffer,
            aligned_range.start as _,
            &data[aligned_range.clone()],
        );
    } else {
        let mut aligned_data = vec![0; aligned_range.len()];
        aligned_data[..data.len() - aligned_range.start]
            .copy_from_slice(&data[aligned_range.start..data.len()]);
        queue.write_buffer(buffer, aligned_range.start as _, &aligned_data);
    }
}

enum PromizeResult {
    Cancelled,
    Load(tree64::Tree64<u8>, Option<Vec<Material>>),
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

    let palette = [
        (65, 59, 47, 255),
        (113, 99, 77, 255),
        (146, 129, 95, 255),
        (156, 142, 113, 255),
        (175, 159, 130, 255),
        (174, 154, 116, 255),
        (192, 174, 135, 255),
        (203, 187, 154, 255),
        (104, 86, 60, 255),
        (114, 99, 77, 255),
        (125, 107, 81, 255),
        (134, 119, 95, 255),
        (153, 135, 107, 255),
        (172, 156, 131, 255),
        (189, 174, 152, 255),
        (206, 193, 175, 255),
        (38, 38, 38, 255),
        (85, 76, 76, 255),
        (107, 100, 97, 255),
        (115, 114, 119, 255),
        (148, 138, 138, 255),
        (147, 118, 107, 255),
        (157, 163, 173, 255),
        (192, 191, 198, 255),
        (77, 68, 59, 255),
        (111, 97, 81, 255),
        (134, 120, 101, 255),
        (155, 139, 116, 255),
        (168, 150, 125, 255),
        (178, 162, 136, 255),
        (190, 172, 140, 255),
        (196, 181, 158, 255),
        (85, 80, 72, 255),
        (117, 110, 95, 255),
        (136, 126, 109, 255),
        (148, 141, 123, 255),
        (162, 152, 132, 255),
        (169, 161, 144, 255),
        (176, 170, 151, 255),
        (188, 180, 158, 255),
        (78, 73, 66, 255),
        (95, 90, 82, 255),
        (117, 107, 92, 255),
        (140, 131, 117, 255),
        (172, 157, 133, 255),
        (184, 173, 156, 255),
        (201, 188, 169, 255),
        (210, 200, 186, 255),
        (29, 31, 32, 255),
        (43, 44, 45, 255),
        (53, 54, 56, 255),
        (83, 78, 69, 255),
        (98, 94, 86, 255),
        (119, 110, 98, 255),
        (141, 133, 123, 255),
        (162, 157, 150, 255),
        (23, 25, 25, 255),
        (35, 39, 39, 255),
        (44, 48, 47, 255),
        (51, 57, 56, 255),
        (58, 63, 63, 255),
        (66, 73, 72, 255),
        (104, 106, 105, 255),
        (173, 173, 173, 255),
        (51, 57, 56, 255),
        (62, 71, 70, 255),
        (69, 76, 75, 255),
        (73, 82, 82, 255),
        (79, 86, 85, 255),
        (82, 92, 92, 255),
        (89, 96, 95, 255),
        (102, 110, 110, 255),
        (9, 9, 9, 255),
        (9, 9, 9, 255),
        (42, 38, 34, 255),
        (120, 110, 97, 255),
        (149, 135, 117, 255),
        (165, 153, 135, 255),
        (182, 167, 142, 255),
        (192, 179, 157, 255),
        (85, 72, 58, 255),
        (118, 105, 85, 255),
        (136, 120, 96, 255),
        (151, 130, 100, 255),
        (157, 141, 111, 255),
        (164, 145, 111, 255),
        (175, 159, 133, 255),
        (173, 152, 113, 255),
        (2, 86, 10, 255),
        (18, 100, 39, 255),
        (104, 117, 57, 255),
        (6, 102, 17, 255),
        (113, 91, 25, 255),
        (36, 118, 58, 255),
        (22, 115, 43, 255),
        (167, 131, 50, 255),
        (98, 99, 98, 255),
        (125, 97, 30, 255),
        (8, 65, 116, 255),
        (26, 64, 130, 255),
        (45, 81, 146, 255),
        (12, 75, 135, 255),
        (30, 77, 147, 255),
        (167, 131, 50, 255),
        (44, 41, 29, 255),
        (60, 71, 40, 255),
        (78, 81, 63, 255),
        (86, 81, 42, 255),
        (106, 94, 86, 255),
        (108, 93, 66, 255),
        (124, 115, 89, 255),
        (136, 128, 120, 255),
        (53, 44, 48, 255),
        (83, 71, 75, 255),
        (108, 92, 95, 255),
        (138, 123, 121, 255),
        (164, 157, 157, 255),
        (179, 165, 161, 255),
        (183, 175, 170, 255),
        (206, 194, 188, 255),
        (36, 89, 10, 255),
        (71, 129, 44, 255),
        (94, 144, 65, 255),
        (102, 149, 46, 255),
        (100, 135, 14, 255),
        (137, 172, 78, 255),
        (167, 66, 7, 255),
        (234, 165, 0, 255),
        (2, 2, 1, 255),
        (30, 26, 21, 255),
        (65, 59, 47, 255),
        (108, 97, 77, 255),
        (138, 124, 100, 255),
        (149, 136, 114, 255),
        (164, 149, 120, 255),
        (172, 159, 136, 255),
    ];

    let mut materials = palette
        .map(|(r, g, b, _)| Material {
            // Boost values for brighter bounces.
            base_colour: [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)],
            linear_roughness: 1.0,
            metallic: 0.0,
            emission_factor: 0.0,
            ..Default::default()
        })
        .to_vec();

    for _ in 0..255 - materials.len() {
        materials.push(Material {
            base_colour: [1.0; 3],
            linear_roughness: 1.0,
            ..Default::default()
        });
    }

    let tree64 = tree64::Tree64::deserialize(std::io::Cursor::new(
        load_resource_bytes("sponza.tree64").await,
    ))
    .unwrap();

    queue.write_buffer(&pipelines.materials, 0, bytemuck::cast_slice(&materials));
    queue.write_buffer(
        &pipelines.tree_nodes,
        0,
        bytemuck::cast_slice(&tree64.nodes.inner),
    );

    copy_aligned(
        &queue,
        &pipelines.leaf_data,
        &tree64.data,
        0..tree64.data.len(),
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
            .with(Position::new(glam::Vec3::new(350.0, 150.0, 230.0)))
            .with(YawPitch::new().yaw_degrees(90.).pitch_degrees(-25.0))
            .with(Smooth::new_position_rotation(0.25, 0.25))
            .with(Arm::new(glam::Vec3::Z * 175.0))
            .build(),
        pipelines,
        materials,
        tree64,
        selected_material: 0,
        hide_ui: false,
        padded_uniform_buffer: encase::UniformBuffer::new(vec![]),
        promise: None,
    };

    event_loop.run_app(&mut app).unwrap()
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;

    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
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
            sun_long: 90.0_f32,
            sun_lat: 45.0_f32,
            enable_shadows: true,
            sun_apparent_size: 1.0_f32,
            accumulate_samples: true,
            max_bounces: 1,
            background_colour: [0.01; 3],
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
    scale: u32,
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

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod, Default)]
#[repr(C)]
struct Material {
    base_colour: [f32; 3],
    emission_factor: f32,
    linear_roughness: f32,
    metallic: f32,
    _padding: [u32; 2],
}
