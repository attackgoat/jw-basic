mod eval;
mod syntax;
mod token;

#[cfg(test)]
pub(self) mod tests;

use {
    self::eval::{Instruction, Interpreter},
    anyhow::Result,
    bytemuck::cast_slice,
    clap::Parser,
    glam::{vec3, Mat4},
    inline_spirv::inline_spirv,
    screen_13::prelude::*,
    std::{fs::read, sync::Arc},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to load and interpret (.bas format)
    path: String,
}

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    let args = Args::parse();

    let event_loop = EventLoop::new()
        .window(|window| {
            window
                .with_title("JW-Basic v0.1")
                .with_inner_size(LogicalSize::new(160 * 4, 96 * 4))
        })
        .build()?;
    let present_pipeline = create_present_pipeline(&event_loop.device)?;

    // Run with `RUST_LOG=debug` to see the generated instructions
    let program = Instruction::compile(&read(args.path)?)?;

    let mut interpreter = Interpreter::new(&event_loop.device, program)?;

    event_loop.run(|frame| {
        interpreter
            .update(frame.render_graph, frame.events)
            .unwrap();

        present_framebuffer_image(&present_pipeline, frame, interpreter.framebuffer_image());
    })?;

    Ok(())
}

fn create_present_pipeline(device: &Arc<Device>) -> Result<Arc<GraphicPipeline>, DisplayError> {
    let vert = inline_spirv!(
        r#"
        #version 460 core

        const float U[6] = {0, 0, 1, 1, 1, 0};
        const float V[6] = {0, 1, 0, 1, 0, 1};
        const float X[6] = {-1, -1, 1, 1, 1, -1};
        const float Y[6] = {-1, 1, -1, 1, -1, 1};

        vec2 vertex_pos() {
            float x = X[gl_VertexIndex];
            float y = Y[gl_VertexIndex];

            return vec2(x, y);
        }

        vec2 vertex_tex() {
            float u = U[gl_VertexIndex];
            float v = V[gl_VertexIndex];

            return vec2(u, v);
        }

        layout(push_constant) uniform PushConstants {
            layout(offset = 0) mat4 vertex_transform;
        } push_constants;
        
        layout(location = 0) out vec2 texcoord_out;

        void main() {
            texcoord_out = vertex_tex();
            gl_Position = push_constants.vertex_transform * vec4(vertex_pos(), 0, 1);
        }
        "#,
        vert
    );
    let fragment = inline_spirv!(
        r#"
        #version 460 core

        layout(binding = 0) uniform sampler2D image_sampler_nne;

        layout(location = 0) in vec2 uv;

        layout(location = 0) out vec4 color;

        void main() {
            vec3 image_sample = texture(image_sampler_nne, uv).rgb;

            color = vec4(image_sample, 1.0);
        }
        "#,
        frag
    );

    Ok(Arc::new(GraphicPipeline::create(
        device,
        GraphicPipelineInfo::new(),
        [
            Shader::new_vertex(vert.as_slice()),
            Shader::new_fragment(fragment.as_slice()),
        ],
    )?))
}

fn present_framebuffer_image(
    present_pipeline: &Arc<GraphicPipeline>,
    frame: FrameContext,
    framebuffer_image: &Arc<Image>,
) {
    let framebuffer_image = frame.render_graph.bind_node(framebuffer_image);
    let transform = {
        let framebuffer_info = frame.render_graph.node_info(framebuffer_image);
        let (framebuffer_width, framebuffer_height) = (
            framebuffer_info.width as f32,
            framebuffer_info.height as f32,
        );
        let (swapchain_width, swapchain_height) = (frame.width as f32, frame.height as f32);
        let scale =
            (swapchain_width / framebuffer_width).min(swapchain_height / framebuffer_height);

        Mat4::from_scale(vec3(
            scale * framebuffer_width / swapchain_width,
            scale * framebuffer_height / swapchain_height,
            1.0,
        ))
    };

    frame
        .render_graph
        .begin_pass("Present")
        .bind_pipeline(present_pipeline)
        .read_descriptor(0, framebuffer_image)
        .clear_color_value(0, frame.swapchain_image, [0x42, 0x42, 0x42, 0xFF])
        .store_color(0, frame.swapchain_image)
        .record_subpass(move |subpass, _| {
            subpass
                .push_constants(cast_slice(&transform.to_cols_array()))
                .draw(6, 1, 0, 0);
        });
}
