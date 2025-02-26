use crate::resource_loading::{load_resource_bytes, load_resource_str};
use crate::Material;
use crate::USE_SPIRV_SHADER;
use wgpu::util::DeviceExt;

pub struct Pipelines {
    pub blit_srgb: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub filtering_sampler: wgpu::Sampler,
    pub non_filtering_sampler: wgpu::Sampler,
    pub trace: wgpu::ComputePipeline,
    pub trace_bgl: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub tree_nodes: wgpu::Buffer,
    pub leaf_data: wgpu::Buffer,
    pub blit_uniform_buffer: wgpu::Buffer,
    pub materials: wgpu::Buffer,
    pub tonemapping_lut: wgpu::Texture,
}

impl Pipelines {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        swapchain_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_entry = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding, visibility, filtering| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Sampler(if filtering {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            count: None,
        };
        let compute_buffer = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                sampler_entry(2, wgpu::ShaderStages::FRAGMENT, true),
                texture_entry(3, wgpu::ShaderStages::FRAGMENT),
                sampler_entry(4, wgpu::ShaderStages::FRAGMENT, false),
            ],
        });

        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                compute_buffer(1),
                compute_buffer(2),
                compute_buffer(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                texture_entry(5, wgpu::ShaderStages::COMPUTE),
                sampler_entry(6, wgpu::ShaderStages::COMPUTE, false),
            ],
        });

        let blit_shader =
            wgpu::ShaderSource::Wgsl(load_resource_str("shaders/blit_srgb.wgsl").await.into());

        Self {
            non_filtering_sampler: device.create_sampler(&wgpu::SamplerDescriptor::default()),
            filtering_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            trace: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&trace_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &if USE_SPIRV_SHADER {
                    unsafe {
                        device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                            label: None,
                            source: std::borrow::Cow::Borrowed(bytemuck::cast_slice(
                                &load_resource_bytes("shaders/raytrace.spv").await,
                            )),
                        })
                    }
                } else {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(
                            load_resource_str("shaders/raytrace.wgsl").await.into(),
                        ),
                    })
                },
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            }),
            blit_srgb: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                label: Default::default(),
                depth_stencil: Default::default(),
                multiview: Default::default(),
                multisample: Default::default(),
                primitive: Default::default(),
                cache: Default::default(),
                vertex: wgpu::VertexState {
                    module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: blit_shader.clone(),
                    }),
                    entry_point: Some("VSMain"),
                    compilation_options: Default::default(),
                    buffers: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: blit_shader,
                    }),
                    entry_point: Some("PSMain"),
                    compilation_options: Default::default(),
                    targets: &[Some(swapchain_format.into())],
                }),
            }),
            bgl,
            trace_bgl,
            uniform_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 1024,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blit_uniform_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<[u32; 4]>() as _,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tree_nodes: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 350226124,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            leaf_data: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 1598646616,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            materials: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<[Material; 255]>() as _,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tonemapping_lut: device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: None,
                    mip_level_count: 1,
                    size: wgpu::Extent3d {
                        width: 48,
                        height: 48,
                        depth_or_array_layers: 48,
                    },
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgb9e5Ufloat,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &ddsfile::Dds::read(std::io::Cursor::new(
                    load_resource_bytes("tony_mc_mapface.dds").await,
                ))
                .unwrap()
                .data,
            ),
        }
    }
}
