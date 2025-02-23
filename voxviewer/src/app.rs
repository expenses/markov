use crate::gpu_resources::{Pipelines, Resizables};
use crate::{
    copy_aligned, srgb_to_linear, CameraUniforms, Material, PromizeResult, Settings, SunUniforms,
    TreeUniforms, Uniforms,
};
use dolly::prelude::*;
use glam::swizzles::Vec3Swizzles;
use winit::event::WindowEvent;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::Window;
use winit::window::WindowId;

pub struct App<'a> {
    pub egui_state: egui_winit::State,
    pub window: &'a Window,
    pub left_mouse_down: bool,
    pub right_mouse_down: bool,
    pub config: wgpu::SurfaceConfiguration,
    pub surface: wgpu::Surface<'a>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub accumulated_frame_index: u32,
    pub frame_index: u32,
    pub settings: Settings,
    pub camera: CameraRig,
    pub resizables: Resizables,
    pub pipelines: Pipelines,
    pub materials: Vec<Material>,
    pub egui_renderer: egui_wgpu::Renderer,
    pub tree64: tree64::Tree64<u8>,
    pub selected_material: usize,
    pub hide_ui: bool,
    pub padded_uniform_buffer: encase::UniformBuffer<Vec<u8>>,
    pub promise: Option<poll_promise::Promise<PromizeResult>>,
}

impl App<'_> {
    fn draw_egui(&mut self, raw_input: egui::RawInput) -> egui::FullOutput {
        let Self {
            egui_state,
            settings,
            materials,
            queue,
            pipelines,
            tree64,
            ..
        } = self;

        let mut reset_accumulation = false;

        let output = egui_state.egui_ctx().run(raw_input, |ctx| {
            egui::Window::new("Controls")
                .default_width(100.0)
                .show(ctx, |ui| {
                    egui::CollapsingHeader::new("Scene").show(ui, |ui| {
                        let browser = || {
                            rfd::AsyncFileDialog::new()
                                .add_filter("64-tree", &["tree64"])
                                .add_filter("MagicaVoxel .vox", &["vox"])
                        };

                        ui.horizontal(|ui| {
                            if ui.button("Load").clicked() {
                                self.promise =
                                    Some(poll_promise::Promise::spawn_local(async move {
                                        let file = match browser().pick_file().await {
                                            Some(file) => file,
                                            None => return PromizeResult::Cancelled,
                                        };

                                        match file.file_name().rsplit_once('.') {
                                            Some((_, "tree64")) => {
                                                let bytes = file.read().await;
                                                PromizeResult::Load(
                                                    tree64::Tree64::deserialize(
                                                        std::io::Cursor::new(bytes),
                                                    )
                                                    .unwrap(),
                                                    None,
                                                )
                                            }
                                            Some((_, "vox")) => {
                                                let bytes = file.read().await;
                                                let vox = dot_vox::load_bytes(&bytes).unwrap();
                                                PromizeResult::Load(
                                                    tree64::Tree64::new((
                                                        {
                                                            let empty_slice: &[u8] = &[];
                                                            empty_slice
                                                        },
                                                        [0; 3],
                                                    )),
                                                    Some(
                                                        vox.palette
                                                            .into_iter()
                                                            .skip(1)
                                                            .map(|colour| Material {
                                                                base_colour: [
                                                                    srgb_to_linear(colour.r),
                                                                    srgb_to_linear(colour.g),
                                                                    srgb_to_linear(colour.b),
                                                                ],
                                                                linear_roughness: 1.0,
                                                                ..Default::default()
                                                            })
                                                            .collect(),
                                                    ),
                                                )
                                            }
                                            _ => return PromizeResult::Cancelled,
                                        }
                                    }));
                            }

                            if ui.button("Save").clicked() {
                                let mut vec = Vec::new();
                                tree64.serialize(&mut vec).unwrap();
                                self.promise =
                                    Some(poll_promise::Promise::spawn_local(async move {
                                        let file = match browser().save_file().await {
                                            Some(file) => file,
                                            None => return PromizeResult::Cancelled,
                                        };

                                        file.write(&vec).await.unwrap();

                                        PromizeResult::Saved
                                    }));
                            }
                        });
                    });
                    egui::CollapsingHeader::new("Edits")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.add_enabled_ui(tree64.edits.can_undo(), |ui| {
                                    if ui.button("Reset").clicked() {
                                        while tree64.edits.can_undo() {
                                            tree64.edits.undo();
                                        }
                                        reset_accumulation = true;
                                    }
                                });
                                ui.add_enabled_ui(tree64.edits.can_undo(), |ui| {
                                    if ui.button("Undo").clicked() {
                                        tree64.edits.undo();
                                        reset_accumulation = true;
                                    }
                                });
                                ui.add_enabled_ui(tree64.edits.can_redo(), |ui| {
                                    if ui.button("Redo").clicked() {
                                        tree64.edits.redo();
                                        reset_accumulation = true;
                                    }
                                });
                                if ui.button("Expand").clicked() {
                                    reset_accumulation = true;
                                    let range_to_upload = tree64.expand();
                                    queue.write_buffer(
                                        &pipelines.tree_nodes,
                                        range_to_upload.start as u64
                                            * std::mem::size_of::<tree64::Node>() as u64,
                                        bytemuck::cast_slice(&tree64.nodes.inner[range_to_upload]),
                                    );
                                }
                            });

                            ui.label("Edit distance");
                            ui.add(egui::Slider::new(&mut settings.edit_distance, 0.0..=1000.0));
                            ui.label("Edit Size");
                            ui.add(egui::Slider::new(&mut settings.edit_size, 0.5..=1000.0));

                            let mut edit = |value| {
                                let position: glam::Vec3 =
                                    glam::Vec3::from(self.camera.final_transform.position)
                                        + self.camera.final_transform.forward::<glam::Vec3>()
                                            * settings.edit_distance;

                                let ranges = tree64.modify_nodes_in_box(
                                    <[i32; 3]>::from(
                                        (position - settings.edit_size / 2.0).xzy().as_ivec3(),
                                    ),
                                    <[i32; 3]>::from(
                                        (position + settings.edit_size / 2.0).xzy().as_ivec3(),
                                    ),
                                    value,
                                );
                                self.accumulated_frame_index = 0;
                                queue.write_buffer(
                                    &pipelines.tree_nodes,
                                    ranges.nodes.start as u64
                                        * std::mem::size_of::<tree64::Node>() as u64,
                                    bytemuck::cast_slice(&tree64.nodes.inner[ranges.nodes]),
                                );
                                copy_aligned(
                                    queue,
                                    &pipelines.leaf_data,
                                    bytemuck::cast_slice(&tree64.data.inner),
                                    ranges.data,
                                );
                            };

                            ui.horizontal(|ui| {
                                if ui.button("Delete").clicked() {
                                    edit(None);
                                }
                                if ui.button("Create").clicked() {
                                    edit(Some(self.selected_material as u8 + 1));
                                }
                            });
                        });
                    egui::CollapsingHeader::new("Rendering")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label("Max bounces");
                            reset_accumulation |= ui
                                .add(egui::Slider::new(&mut settings.max_bounces, 0..=10))
                                .changed();
                            reset_accumulation |= ui
                                .checkbox(&mut settings.accumulate_samples, "Accumulate Samples")
                                .changed();
                        });
                    egui::CollapsingHeader::new("Lighting").show(ui, |ui| {
                        egui::CollapsingHeader::new("Sun")
                            .default_open(true)
                            .show(ui, |ui| {
                                reset_accumulation |= ui
                                    .checkbox(&mut settings.enable_shadows, "Enable shadows")
                                    .changed();
                                ui.label("Latitude");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(&mut settings.sun_lat, 0.0..=90.0))
                                    .changed();
                                ui.label("Longitude");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(&mut settings.sun_long, -180.0..=180.0))
                                    .changed();
                                ui.label("Apparent size");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(
                                        &mut settings.sun_apparent_size,
                                        0.0..=90.0,
                                    ))
                                    .changed();
                                ui.label("Sun Strength");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(
                                        &mut settings.sun_strength,
                                        0.0..=10_000.0,
                                    ))
                                    .changed();
                                ui.label("Colour");
                                reset_accumulation |=
                                    egui::widgets::color_picker::color_edit_button_rgb(
                                        ui,
                                        &mut settings.sun_colour,
                                    )
                                    .changed();
                            });
                        ui.label("Background colour");
                        reset_accumulation |= egui::widgets::color_picker::color_edit_button_rgb(
                            ui,
                            &mut settings.background_colour,
                        )
                        .changed();
                        ui.label("Background Strength");
                        reset_accumulation |= ui
                            .add(egui::Slider::new(
                                &mut settings.background_strength,
                                0.0..=10_000.0,
                            ))
                            .changed();
                    });
                    egui::CollapsingHeader::new("Camera").show(ui, |ui| {
                        ui.label("Field of view (vertical)");
                        reset_accumulation |= ui
                            .add(egui::Slider::new(&mut settings.vertical_fov, 1.0..=120.0))
                            .changed();
                    });
                    egui::CollapsingHeader::new("Settings").show(ui, |ui| {
                        if ui.button("Reset").clicked() {
                            *settings = Settings::default();
                            reset_accumulation = true;
                        }
                    });
                    #[cfg(not(target_arch = "wasm32"))]
                    egui::CollapsingHeader::new("Debugging").show(ui, |ui| {
                        reset_accumulation |= ui
                            .checkbox(&mut settings.show_heatmap, "Show Heatmap")
                            .changed();
                    });
                    egui::CollapsingHeader::new("Materials")
                        .default_open(true)
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical()
                                .max_height(400.0)
                                .show(ui, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        for (i, material) in materials.iter().enumerate() {
                                            let (rect, response) = ui.allocate_at_least(
                                                egui::Vec2::new(35., 20.),
                                                egui::Sense::click(),
                                            );

                                            let stroke_colour = egui::Color32::WHITE
                                                .linear_multiply(
                                                    if response.hovered()
                                                        || i == self.selected_material
                                                    {
                                                        1.0
                                                    } else {
                                                        0.25
                                                    },
                                                );

                                            if response.clicked() {
                                                self.selected_material = i;
                                            }

                                            ui.painter().rect(
                                                rect,
                                                5.0,
                                                egui::Rgba::from_rgb(
                                                    material.base_colour[0],
                                                    material.base_colour[1],
                                                    material.base_colour[2],
                                                ),
                                                egui::Stroke::new(1.0, stroke_colour),
                                                egui::StrokeKind::Middle,
                                            );
                                        }
                                    });
                                });

                            let material = &mut materials[self.selected_material];

                            let mut changed = false;
                            ui.label("Base Colour");
                            changed |= egui::widgets::color_picker::color_edit_button_rgb(
                                ui,
                                &mut material.base_colour,
                            )
                            .changed();
                            ui.label("Emission Factor");
                            changed |= ui
                                .add(egui::Slider::new(
                                    &mut material.emission_factor,
                                    0.0..=10_000.0,
                                ))
                                .changed();
                            ui.label("Linear Roughness");
                            changed |= ui
                                .add(egui::Slider::new(
                                    &mut material.linear_roughness,
                                    0.000..=1.0,
                                ))
                                .changed();
                            ui.label("Metallic Factor");
                            changed |= ui
                                .add(egui::Slider::new(&mut material.metallic, 0.0..=1.0))
                                .changed();
                            if changed {
                                queue.write_buffer(
                                    &pipelines.materials,
                                    (self.selected_material * std::mem::size_of::<Material>()) as _,
                                    bytemuck::bytes_of(&*material),
                                );
                                reset_accumulation = true;
                            };
                        });
                });
        });

        if reset_accumulation {
            self.accumulated_frame_index = 0;
        }

        output
    }

    fn get_egui_render_state(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> (Vec<egui::ClippedPrimitive>, egui_wgpu::ScreenDescriptor) {
        let raw_input = self.egui_state.take_egui_input(self.window);
        let egui_output = self.draw_egui(raw_input);

        let Self {
            egui_state,
            window,
            egui_renderer,
            device,
            queue,
            ..
        } = self;

        egui_state
            .egui_ctx()
            .set_pixels_per_point(window.scale_factor() as _);

        egui_state.handle_platform_output(window, egui_output.platform_output);

        let tris = egui_state
            .egui_ctx()
            .tessellate(egui_output.shapes, window.scale_factor() as _);

        for (id, image_delta) in &egui_output.textures_delta.set {
            egui_renderer.update_texture(device, queue, *id, image_delta);
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            pixels_per_point: window.scale_factor() as _,
            size_in_pixels: [self.config.width, self.config.height],
        };

        let command_buffers =
            egui_renderer.update_buffers(device, queue, encoder, &tris, &screen_descriptor);

        debug_assert!(command_buffers.is_empty());

        (tris, screen_descriptor)
    }

    fn write_uniforms(
        &mut self,
        transform: dolly::transform::Transform<dolly::handedness::RightHanded>,
    ) {
        let settings = &self.settings;
        let root_state = self.tree64.root_state();

        let view_matrix =
            glam::Mat4::look_to_rh(glam::Vec3::ZERO, transform.forward(), transform.up());

        self.padded_uniform_buffer
            .write(&Uniforms {
                camera: CameraUniforms {
                    p_inv: (glam::Mat4::perspective_infinite_reverse_rh(
                        settings.vertical_fov.to_radians(),
                        self.config.width as f32 / self.config.height as f32,
                        0.0001,
                    ) * view_matrix)
                        .inverse(),
                    pos: glam::Vec3::from(transform.position).into(),
                    forward: transform.forward(),
                    view: view_matrix,
                    view_inv: view_matrix.inverse(),
                    up: transform.up(),
                    right: transform.right(),
                },
                sun: SunUniforms {
                    emission: (glam::Vec3::from(settings.sun_colour) * settings.sun_strength)
                        .into(),
                    direction: glam::Vec3::new(
                        settings.sun_long.to_radians().sin() * settings.sun_lat.to_radians().cos(),
                        settings.sun_lat.to_radians().sin(),
                        settings.sun_long.to_radians().cos() * settings.sun_lat.to_radians().cos(),
                    )
                    .into(),
                    cosine_apparent_size: settings.sun_apparent_size.to_radians().cos(),
                },
                tree: TreeUniforms {
                    scale: root_state.num_levels as u32 * 2,
                    root_node_index: root_state.index,
                    offset: <[i32; 3]>::from(root_state.offset).into(),
                },
                resolution: glam::UVec2::new(self.config.width, self.config.height),

                settings: (settings.enable_shadows as i32)
                    | (settings.accumulate_samples as i32) << 1
                    | (settings.show_heatmap as i32) << 2,
                frame_index: self.frame_index,
                accumulated_frame_index: self.accumulated_frame_index,
                max_bounces: settings.max_bounces,
                background_colour: (glam::Vec3::from(settings.background_colour)
                    * settings.background_strength)
                    .into(),
            })
            .unwrap();

        self.queue.write_buffer(
            &self.pipelines.uniform_buffer,
            0,
            self.padded_uniform_buffer.as_ref(),
        );
        self.queue.write_buffer(
            &self.pipelines.blit_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.accumulated_frame_index),
        );
    }
}

impl winit::application::ApplicationHandler for App<'_> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {}

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        #[cfg(not(target_arch = "wasm32"))]
        poll_promise::tick_local();
        if let Some(promise) = self.promise.take() {
            match promise.try_take() {
                Err(promise) => self.promise = Some(promise),
                Ok(value) => {
                    if let PromizeResult::Load(tree64, materials) = value {
                        self.tree64 = tree64;
                        self.queue.write_buffer(
                            &self.pipelines.tree_nodes,
                            0,
                            bytemuck::cast_slice(&self.tree64.nodes.inner),
                        );

                        copy_aligned(
                            &self.queue,
                            &self.pipelines.leaf_data,
                            &self.tree64.data,
                            0..self.tree64.data.len(),
                        );
                        self.accumulated_frame_index = 0;
                        if let Some(materials) = materials {
                            self.materials = materials;
                            self.queue.write_buffer(
                                &self.pipelines.materials,
                                0,
                                bytemuck::cast_slice(&self.materials),
                            );
                        }
                    }
                }
            }
        }
        self.window.request_redraw();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta: (x, y) } if self.left_mouse_down => {
                self.camera
                    .driver_mut::<YawPitch>()
                    .rotate_yaw_pitch((-x / 2.0) as f32, (-y / 2.0) as f32);

                self.accumulated_frame_index = 0;
            }
            DeviceEvent::MouseMotion { delta: (x, y) } if self.right_mouse_down => {
                let transform = self.camera.final_transform;
                // Default strength of 500.0 seems to work well at default v fov (45)
                let strength_divisor = 500.0 * 45.0;
                let arm_distance = self.camera.driver::<Arm>().offset.z / strength_divisor
                    * self.settings.vertical_fov;
                self.camera.driver_mut::<Position>().translate(
                    (transform.up::<glam::Vec3>() * y as f32
                        + transform.right::<glam::Vec3>() * -x as f32)
                        * arm_distance,
                );

                self.accumulated_frame_index = 0;
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let egui_response = self.egui_state.on_window_event(self.window, &event);
        if egui_response.consumed {
            self.left_mouse_down = false;
            self.right_mouse_down = false;
            return;
        }

        let view_format = self.config.view_formats[0];

        match event {
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.left_mouse_down = state == ElementState::Pressed,
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => self.right_mouse_down = state == ElementState::Pressed,
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state,
                        ..
                    },
                ..
            } => {
                if key_code == winit::keyboard::KeyCode::KeyH {
                    self.hide_ui ^= state == ElementState::Pressed;
                }
            }
            WindowEvent::Resized(new_size) => {
                self.accumulated_frame_index = 0;
                // Reconfigure the surface with the new size
                self.config.width = new_size.width.max(1);
                self.config.height = new_size.height.max(1);
                self.surface.configure(&self.device, &self.config);
                self.resizables = Resizables::new(
                    new_size.width.max(1),
                    new_size.height.max(1),
                    &self.device,
                    &self.pipelines,
                );
                // On macos the window needs to be redrawn manually after resizing
                self.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // TODO: this is very WIP.
                let pixel_delta_divisor = if cfg!(target_arch = "wasm32") {
                    1000.0
                } else {
                    100.0
                };
                self.camera.driver_mut::<Arm>().offset.z *= 1.0
                    + match delta {
                        MouseScrollDelta::LineDelta(_, y) => y / 10.0,
                        MouseScrollDelta::PixelDelta(pos) => -pos.y as f32 / pixel_delta_divisor,
                    }
            }
            WindowEvent::RedrawRequested => {
                let previous_transform = self.camera.final_transform;

                let transform = self.camera.update(1.0 / 60.0);

                if glam::Vec3::from(previous_transform.position)
                    .distance_squared(transform.position.into())
                    > (0.05 * 0.05)
                {
                    self.accumulated_frame_index = 0;
                }

                if !self.settings.accumulate_samples || self.settings.show_heatmap {
                    self.accumulated_frame_index = 0;
                }

                self.write_uniforms(transform);

                if self.settings.accumulate_samples {
                    self.accumulated_frame_index += 1;
                }
                self.frame_index += 1;

                let frame = self
                    .surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(view_format),
                    ..Default::default()
                });
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let (tessellated, screen_descriptor) = self.get_egui_render_state(&mut encoder);
                let egui_cmd_buf = encoder.finish();

                let (cmd_buf_a, cmd_buf_b) = rayon::join(
                    || {
                        let mut encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });

                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

                        compute_pass.set_pipeline(&self.pipelines.trace);
                        compute_pass.set_bind_group(
                            0,
                            &self.resizables.trace_bind_groups
                                [self.accumulated_frame_index as usize % 2],
                            &[],
                        );

                        let workgroup_size = 8;

                        compute_pass.dispatch_workgroups(
                            self.config.width.div_ceil(workgroup_size),
                            self.config.height.div_ceil(workgroup_size),
                            1,
                        );

                        drop(compute_pass);

                        encoder.finish()
                    },
                    || {
                        let mut encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });

                        let mut rpass = encoder
                            .begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            })
                            .forget_lifetime();
                        rpass.set_pipeline(&self.pipelines.blit_srgb);
                        rpass.set_bind_group(
                            0,
                            &self.resizables.blit_bind_groups
                                [(self.accumulated_frame_index) as usize % 2],
                            &[],
                        );
                        rpass.draw(0..3, 0..1);
                        if !self.hide_ui {
                            self.egui_renderer
                                .render(&mut rpass, &tessellated, &screen_descriptor);
                        }

                        drop(rpass);

                        encoder.finish()
                    },
                );

                self.queue.submit([egui_cmd_buf, cmd_buf_a, cmd_buf_b]);
                frame.present();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}
