use bevy::{
    core_pipeline::core_3d::graph::Core3d,
    ecs::{
        query::QueryItem,
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    math::*,
    pbr::{MeshPipelineKey, SetMeshViewBindGroup},
    prelude::*,
    render::primitives::Aabb,
    render::render_graph::{NodeRunError, RenderGraphContext, ViewNodeRunner},
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{self, RenderGraphApp, RenderLabel},
        render_phase::*,
        render_resource::*,
        renderer::RenderContext,
        view::*,
        *,
    },
};

mod phase_item;
mod pipeline;

use pipeline::{RenderingPipeline, VoxelRendererPipeline};

use phase_item::{queue_voxel_phase_items, VoxelBinnedPhaseItem};
use renderer::RenderDevice;

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_plugins(DefaultPlugins)
        .add_plugins(bevy_panorbit_camera::PanOrbitCameraPlugin)
        .add_plugins(VoxelRendererPlugin)
        .run();
}

fn setup(mut commands: Commands, render_device: Res<RenderDevice>) {
    commands.spawn((
        Visibility::default(),
        Transform::default(),
        // This `Aabb` is necessary for the visibility checks to work.
        Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(0.5),
        },
        VoxelModel {
            buffer: {
                let mut array = [1; 8 * 8 * 8];
                for z in 0..8 {
                    if z % 2 == 0 {
                        for i in 0..8 * 8 {
                            array[i + z * 8 * 8] = 0;
                        }
                    }
                }
                array[0] = 2;
                let svo_dag = svo_dag::SvoDag::new(&array, 8, 8, 8, 256);

                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("VoxelRenderer::nodes"),
                    contents: svo_dag.node_bytes(),
                    usage: bevy::render::render_resource::BufferUsages::STORAGE,
                })
            },
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 1000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 2.0, 0.0)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 4.)),
    ));

    commands.spawn((
        Transform::from_xyz(-20.0, 20.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy_panorbit_camera::PanOrbitCamera::default(),
    ));
}

fn prepare_voxel_render_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<RenderingPipeline>>,
    pipeline: Res<RenderingPipeline>,
    views: Query<(Entity, &ExtractedView, &Msaa)>,
) {
    for (entity, view, msaa) in &views {
        let view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, view_key);

        commands
            .entity(entity)
            .insert(VoxelRenderPipeline(pipeline_id));
    }
}

#[derive(Component)]
struct VoxelRenderPipeline(CachedRenderPipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VoxelRendererLabel;

struct VoxelRendererPlugin;

impl Plugin for VoxelRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<VoxelModel>::default())
            .add_systems(
                PostUpdate,
                view::check_visibility::<With<VoxelModel>>
                    .in_set(VisibilitySystems::CheckVisibility),
            );
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<DrawFunctions<VoxelBinnedPhaseItem>>()
            .init_resource::<ViewBinnedRenderPhases<VoxelBinnedPhaseItem>>()
            .init_resource::<SpecializedRenderPipelines<RenderingPipeline>>()
            .add_render_command::<VoxelBinnedPhaseItem, DrawVoxelCubesCommands>()
            .add_systems(
                Render,
                (
                    queue_voxel_phase_items.in_set(RenderSet::Queue),
                    prepare_voxel_render_pipelines.in_set(RenderSet::Prepare),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<VoxelRendererNode>>(Core3d, VoxelRendererLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    VoxelRendererLabel,
                    bevy::core_pipeline::core_3d::graph::Node3d::MainOpaquePass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<VoxelRendererPipeline>();
        render_app.init_resource::<RenderingPipeline>();
    }
}

type DrawVoxelCubesCommands = (SetMeshViewBindGroup<0>, DrawVoxelCubes);

#[derive(Clone, Component, ExtractComponent)]
struct VoxelModel {
    buffer: Buffer,
    //svo_dag: svo_dag::SvoDag<u32>,
}

struct DrawVoxelCubes;

impl<P> RenderCommand<P> for DrawVoxelCubes
where
    P: PhaseItem,
{
    type Param = SRes<VoxelRendererPipeline>;

    type ViewQuery = ();

    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, Self::ItemQuery>>,
        custom_phase_item_buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let pipeline = custom_phase_item_buffers.into_inner();
        pass.draw_indirect(&pipeline.draw_indirect, 0);

        RenderCommandResult::Success
    }
}

#[derive(Default)]
struct VoxelRendererNode {
    ready: bool,
}

impl render_graph::ViewNode for VoxelRendererNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<VoxelRendererPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if !self.ready {
            match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
                CachedPipelineState::Ok(_) => {
                    self.ready = true;
                }
                #[cfg(target_arch = "wasm32")]
                CachedPipelineState::Err(
                    err @ bevy::render::render_resource::PipelineCacheError::ShaderNotLoaded(_),
                ) => {
                    log::debug!("Caught {err}. Optimistically assuming that the shader is still being downloaded.");
                }
                CachedPipelineState::Err(err) => {
                    panic!("{err}");
                }
                _ => {}
            }
        }
    }

    type ViewQuery = (
        Entity,
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static VoxelRenderPipeline,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext<'_>,
        render_context: &mut RenderContext<'w>,
        (view, camera, target, depth, render_pipeline): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        if !self.ready {
            return Ok(());
        }

        let voxel_phases = world.resource::<ViewBinnedRenderPhases<VoxelBinnedPhaseItem>>();
        let voxel_phase = voxel_phases.get(&view).unwrap();

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<VoxelRendererPipeline>();

        let view_entity = graph.view_entity();

        let draw_functions = world.resource::<DrawFunctions<VoxelBinnedPhaseItem>>();
        let draw_function_id = draw_functions.read().id::<DrawVoxelCubesCommands>();

        {
            let mut draw_functions = draw_functions.write();
            draw_functions.prepare(world);
        }

        for &(ref key, entity) in voxel_phase.non_mesh_items.iter() {
            let color_attachments = [Some(target.get_color_attachment())];
            let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

            render_context.add_command_buffer_generation_task(move |render_device| {
                // Command encoder setup
                let mut command_encoder =
                    render_device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("voxel_command_encoder"),
                    });

                let render_pipeline = match pipeline_cache.get_render_pipeline(render_pipeline.0) {
                    Some(render_pipeline) => render_pipeline,
                    None => return command_encoder.finish(),
                };

                command_encoder.clear_buffer(&pipeline.cubes, 0, Some(4));
                command_encoder.clear_buffer(&pipeline.draw_indirect, 0, Some(4));

                command_encoder.copy_buffer_to_buffer(
                    &pipeline.u32_1_buffer,
                    0,
                    &pipeline.work_items[0].buffer,
                    0,
                    4,
                );

                command_encoder.copy_buffer_to_buffer(
                    &pipeline.first_work_item,
                    0,
                    &pipeline.work_items[0].buffer,
                    // There's padding after the atomic in the buffer as it's not tightly packed.
                    16,
                    std::mem::size_of::<pipeline::WorkItem>() as u64,
                );

                command_encoder.copy_buffer_to_buffer(
                    &pipeline.u32_1_buffer,
                    0,
                    &pipeline.work_items[0].dispatch,
                    0,
                    std::mem::size_of::<u32>() as u64,
                );

                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.pipeline)
                    .unwrap();

                for (i, uniform_bind_group) in pipeline.uniform_bind_groups.iter().rev().enumerate()
                {
                    command_encoder.clear_buffer(
                        &pipeline.work_items[(i + 1) % 2].buffer,
                        0,
                        Some(4),
                    );
                    command_encoder.clear_buffer(
                        &pipeline.work_items[(i + 1) % 2].dispatch,
                        0,
                        Some(std::mem::size_of::<u32>() as u64),
                    );

                    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("voxel_command_encoder_compute_pass"),
                        ..Default::default()
                    });

                    pass.set_pipeline(init_pipeline);
                    pass.set_bind_group(0, &pipeline.node_bind_group, &[]);
                    pass.set_bind_group(1, &pipeline.base_bind_group, &[]);
                    pass.set_bind_group(2, &pipeline.flip_flop_bind_groups[i % 2], &[]);
                    pass.set_bind_group(3, uniform_bind_group, &[]);
                    pass.dispatch_workgroups_indirect(&pipeline.work_items[i % 2].dispatch, 0);
                }

                {
                    let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some("voxel_command_encoder_render_pass"),
                        color_attachments: &color_attachments,
                        depth_stencil_attachment,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);

                    if let Some(viewport) = camera.viewport.as_ref() {
                        render_pass.set_camera_viewport(viewport);
                    }

                    render_pass.set_render_pipeline(render_pipeline);
                    render_pass.set_bind_group(1, &pipeline.draw_bind_group, &[]);

                    let mut draw_functions = draw_functions.write();
                    let binned_phase_item = VoxelBinnedPhaseItem::new(
                        key.clone(),
                        entity,
                        0..1,
                        PhaseItemExtraIndex(0),
                    );
                    draw_functions
                        .get_mut(draw_function_id)
                        .unwrap()
                        .draw(world, &mut render_pass, view_entity, &binned_phase_item)
                        .unwrap();
                }

                command_encoder.finish()
            });
        }

        Ok(())
    }
}