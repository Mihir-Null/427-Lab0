#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod gpu;

use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;  // create_buffer_init()
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};
use gpu::GpuCtx;

// Vertex definition
// each point computed by CPU and laid out as raw bytes then pushed to GPU buffer
// Vertices computed on CPU rn, check if we can do it purely GPU later

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color:    [f32; 3],
}

impl Vertex {
    // shader operates on raw bytes AFTER we push them to the GPU buffer, so we
    // need to tell it how to interpret those bytes in layout()
    // rust OOP kinda messed up ngl
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    // `color` begins right after the two f32s of `position`.
                    offset:          std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format:          wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// rose curve generator
// samples positions along r(θ) = cos(n·θ) and returns them as a vec of vertices
// Params
//   petals: n in r = cos(n·θ). 5 → 5-petal rose.
//   num_points: number of line segments, i.e smoothness
//   time: t in seconds, param for animation
fn make_rose_curve(petals: u32, num_points: u32, time: f32) -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(num_points as usize + 1);
    let phase = time * 0.8;

    for i in 0..=num_points {
        let t     = i as f32 / num_points as f32;
        let theta = t * 2.0 * std::f32::consts::PI + phase;
        let r = (petals as f32 * theta).cos();
        // 80% scaling NDC for margin
        let x = r * theta.cos() * 0.8;
        let y = r * theta.sin() * 0.8;

        // colour cycles purple -> teal as θ advances
        // color = r,g,b
        let color = [
            (theta * 0.5).sin() * 0.5 + 0.5,
            0.2_f32,
            (theta * 0.5).cos() * 0.5 + 0.5,
        ];

        vertices.push(Vertex { position: [x, y], color });
    }

    vertices
}

// WGSL shader source, defns vertex + fragment shaders and their i/o
// vs_main = vertex shader, fs_main = fragment shader
const SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color:    vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       color:         vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // lift the 2d NDC position into 4d clip space
    // z=0 (on near plane), w=1 (no perspective distortion)
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.color         = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // output interpolated vertex colour as fully opaque RGBA
    return vec4<f32>(in.color, 1.0);
}
"#;

// State - owns all GPU resources + the render pipeline
// also holds window ref and manages some ownership so rust is happy
struct State
{
    window:       Arc<Window>,
    gpu:          GpuCtx,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer:   wgpu::Buffer,
    vertex_count:    u32,
    start_time:      Instant,
}

impl State
{
    // all OS calls/interaction need to be async
    pub async fn new(window: Arc<Window>) -> Self
    {
        let gpu = GpuCtx::new(window.clone()).await;

        // checks if shadercode works with pipeline setup + compilation
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Rose Curve Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // any actual pipelining e.g texturing, compute shaders etc.
        // use bind groups and layout, makes more complex pipelines much easier
        let pipeline_layout = gpu.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Pipeline Layout"),
                bind_group_layouts: &[],  // lab0 has no uniforms
                immediate_size:     0,
            }
        );

        // central GPU pipeline object, defines gpu/rendering state machine
        let render_pipeline = gpu.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label:  Some("Rose Curve Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module:              &shader,
                    entry_point:         Some("vs_main"),
                    buffers:             &[Vertex::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.format(),
                        // REPLACE: each frag overwrites whatever was there
                        // ALPHA_BLENDING would let transparent frags blend instead
                        blend:      Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                // linestrip tells gpu that each vertex is an interp point on a line
                // change to trianglelist and add indices for more complex shapes like in webgl for later labs
                primitive: wgpu::PrimitiveState {
                    topology:   wgpu::PrimitiveTopology::LineStrip,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode:  None,  // lines have no face to cull
                    ..Default::default()
                },
                depth_stencil:  None,  // 2d scene, no depth buffer needed
                multisample:    wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache:          None,
            }
        );

        // vertex buffer with init data for gpu
        let initial_verts = make_rose_curve(5, 2000, 0.0);
        let vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Rose Vertex Buffer"),
            contents: bytemuck::cast_slice(&initial_verts),
            usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let vertex_count = initial_verts.len() as u32;

        Self {
            window,
            gpu,
            render_pipeline,
            vertex_buffer,
            vertex_count,
            start_time: Instant::now(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    pub fn render(&mut self) -> Result<(), ()>
    {
        // schedules redraws, winit doesn't do it for us
        self.window.request_redraw();

        // guard: surface.get_current_texture() would panic before configure()
        // on WASM, winit doesn't fire Resized at startup so we configure here on first render
        if !self.gpu.is_configured {
            let s = self.window.inner_size();
            self.gpu.resize(s.width, s.height);
            if !self.gpu.is_configured { return Ok(()); }
        }

        // compute geometry on CPU each frame, then push to GPU buffer
        let t = self.start_time.elapsed().as_secs_f32();
        let verts = make_rose_curve(5, 2000, t);
        self.gpu.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));
        self.vertex_count = verts.len() as u32;

        // get next swap chain texture
        let output = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o)
            | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => {
                self.gpu.surface.configure(&self.gpu.device, &self.gpu.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Validation  => return Err(()),
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // record GPU commands
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Frame Encoder") }
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Rose Curve Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,  // no multisampling
                    depth_slice:    None,
                    ops: wgpu::Operations {
                        // clear screen before draw
                        // LoadOp::Load would keep whatever was there last frame
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.08, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
                multiview_mask:           None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..self.vertex_count, 0..1);
        }

        // submit + present
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}


// Application + Handler <- OS events to State, winit lives here
// native: state is just Option<State>
// wasm: needs Rc<RefCell<Option<State>>> so the async init callback can write it from a different call stack. with_state() hides the difference

pub struct App
{
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<State>,
    #[cfg(target_arch = "wasm32")]
    state: std::rc::Rc<std::cell::RefCell<Option<State>>>,
}

impl App
{
    pub fn new() -> Self
    {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            state: None,
            #[cfg(target_arch = "wasm32")]
            state: std::rc::Rc::new(std::cell::RefCell::new(None)),
        }
    }

    fn with_state<R>(&mut self, f: impl FnOnce(&mut State) -> R) -> Option<R>
    {
        #[cfg(not(target_arch = "wasm32"))]
        { self.state.as_mut().map(f) }
        #[cfg(target_arch = "wasm32")]
        { self.state.borrow_mut().as_mut().map(f) }
    }
}

impl ApplicationHandler for App
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop)
    {
        if self.with_state(|_| ()).is_some() { return; }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("CMSC427 Lab 00 – Rose Curve")
                )
                .unwrap(),
        );

        // wasm: inject canvas into DOM before async init
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("canvas-host")
                    .or_else(|| d.body().map(|b| b.into())))
                .and_then(|host| window.canvas()
                    .and_then(|c| host.append_child(&c).ok()));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)));
        }

        // self.state = Some(pollster::block_on(State::new(window.clone())));
        // window.request_redraw();

        #[cfg(target_arch = "wasm32")]
        {
            let cell   = self.state.clone();
            let window = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let state = State::new(window.clone()).await;
                *cell.borrow_mut() = Some(state);
                window.request_redraw();
            });
        }
    }

    // winit managed events/interaction, kbm input goes here
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event:      WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(s) => {
                self.with_state(|state| state.resize(s.width, s.height));
            }

            // RedrawRequested: we ask for this at the top of render() to create render loop without busy waiting
            // lets us do work btwn frames
            WindowEvent::RedrawRequested => {
                self.with_state(|state| { let _ = state.render(); });
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state:        key_state,
                    ..
                },
                ..
            } => {
                if code == KeyCode::Escape && key_state.is_pressed() {
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }
}

// Entry points

pub fn run()
{
    let event_loop = EventLoop::new().unwrap();
    let mut app    = App::new();
    event_loop.run_app(&mut app).unwrap();
}

// wasm entry point - called by browser after wasm-bindgen setup
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start()
{
    console_log::init_with_level(log::Level::Warn).ok();
    console_error_panic_hook::set_once();

    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().unwrap();
    event_loop.spawn_app(App::new());
}
