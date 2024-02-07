mod charset;
mod instr;
mod palette;

pub use {charset::ascii_5x6, instr::Instruction, palette::vga_256};

use {
    bytemuck::cast_slice,
    inline_spirv::inline_spirv,
    rand::random,
    screen_13::prelude::*,
    std::{mem::size_of, ops::Range, sync::Arc, time::Instant},
};

fn index_multi_to_linear(indexes: &[i32], subscripts: &[Range<i32>]) -> usize {
    let mut idx = 0;
    let mut mul = 1usize;

    for (index, (start, len)) in indexes.iter().copied().zip(
        subscripts
            .iter()
            .map(|subscript| (subscript.start, subscript.len() + 1)),
    ) {
        idx += (index - start) * mul as i32;
        mul *= len;
    }

    idx as _
}

fn linear_len_from_multi(subscripts: &[Range<i32>]) -> usize {
    subscripts
        .iter()
        .map(|subscript| subscript.len() + 1)
        .product()
}

fn virtual_key_code(key_code: u8) -> KeyCode {
    // TODO: Add more keys or use scan codes or do anything better here
    match key_code {
        0 => KeyCode::Escape,
        1 => KeyCode::ArrowLeft,
        2 => KeyCode::ArrowRight,
        3 => KeyCode::ArrowUp,
        4 => KeyCode::ArrowDown,
        _ => unimplemented!(),
    }
}

pub struct Interpreter {
    color: (u8, u8),
    character_buf: Arc<Buffer>,
    framebuffer_images: [Arc<Image>; 2],
    graphics_buf: Arc<Lease<Buffer>>,
    graphics_data: Vec<u8>,
    graphics_dirty: bool,
    graphics_pipeline: Arc<ComputePipeline>,
    heap: Vec<u8>,
    keyboard: KeyBuf,
    location: (usize, usize),
    palette_buf: Arc<Lease<Buffer>>,
    palette_data: Vec<u8>,
    palette_dirty: bool,
    pool: HashPool,
    program: Vec<Instruction>,
    program_index: usize,
    stack: Vec<Value>,
    started_at: Instant,
    text_buf: Arc<Lease<Buffer>>,
    text_data: Vec<u8>,
    text_dirty: bool,
    text_pipeline: Arc<ComputePipeline>,
}

impl Interpreter {
    const DEFAULT_COLOR: (u8, u8) = (15, 0xFF);
    const HEAP_SIZE: usize = 16_384;
    pub const TEXT_COLS: usize = 32;
    pub const TEXT_ROWS: usize = 16;
    pub const FRAMEBUFFER_WIDTH: u32 = 160;
    pub const FRAMEBUFFER_HEIGHT: u32 = 96;

    pub fn new(device: &Arc<Device>, program: Vec<Instruction>) -> Result<Self, DriverError> {
        let framebuffer_images = [
            Self::create_framebuffer_image(device)?,
            Self::create_framebuffer_image(device)?,
        ];

        let mut pool = HashPool::new(device);

        let graphics_data = vec![0xFF; (Self::FRAMEBUFFER_WIDTH * Self::FRAMEBUFFER_HEIGHT) as _];
        let mut graphics_buf = pool.lease(BufferInfo::new_mappable(
            graphics_data.len() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ))?;
        Buffer::copy_from_slice(&mut graphics_buf, 0, &graphics_data);

        let palette_data = vga_256();
        let mut palette_buf = pool.lease(BufferInfo::new_mappable(
            palette_data.len() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ))?;
        Buffer::copy_from_slice(&mut palette_buf, 0, &palette_data);

        let text_data = vec![0; Self::TEXT_COLS * Self::TEXT_ROWS * 4];
        let mut text_buf = pool.lease(BufferInfo::new_mappable(
            text_data.len() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ))?;
        Buffer::copy_from_slice(&mut text_buf, 0, &text_data);

        let graphics_pipeline = Self::create_graphics_pipeline(device)?;
        let text_pipeline = Self::create_text_pipeline(device)?;

        let mut render_graph = RenderGraph::new();
        let character_buf = Self::create_character_buffer(device, &mut render_graph)?;

        for framebuffer_image in framebuffer_images.iter() {
            let framebuffer_image = render_graph.bind_node(framebuffer_image);
            render_graph.clear_color_image(framebuffer_image);
        }

        let queue_family_index = device
            .physical_device
            .queue_families
            .iter()
            .enumerate()
            .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(index, _)| index)
            .unwrap();

        render_graph
            .resolve()
            .submit(&mut HashPool::new(device), queue_family_index, 0)?;

        const fn zero_byte() -> Value {
            Value::Byte(0)
        }

        const ZERO_BYTE: Value = zero_byte();

        Ok(Self {
            color: Self::DEFAULT_COLOR,
            character_buf,
            framebuffer_images,
            graphics_buf: Arc::new(graphics_buf),
            graphics_data,
            graphics_dirty: false,
            graphics_pipeline,
            heap: vec![0; Self::HEAP_SIZE],
            keyboard: KeyBuf::default(),
            location: (0, 0),
            palette_buf: Arc::new(palette_buf),
            palette_data,
            palette_dirty: false,
            pool,
            program,
            program_index: 0,
            stack: vec![ZERO_BYTE; 4_096],
            started_at: Instant::now(),
            text_buf: Arc::new(text_buf),
            text_data,
            text_dirty: false,
            text_pipeline,
        })
    }

    pub fn color(&mut self, foreground: u8, background: u8) {
        self.color = (foreground, background);
    }

    pub fn cls(&mut self) {
        self.graphics_data.fill(0xFF);
        self.location = (0, 0);
        self.text_data.fill(0);
        self.text_dirty = true;
    }

    fn create_character_buffer(
        device: &Arc<Device>,
        render_graph: &mut RenderGraph,
    ) -> Result<Arc<Buffer>, DriverError> {
        let characters = ascii_5x6();

        let buf_size = characters.len() as vk::DeviceSize * 4;
        let mut temp_buf = Buffer::create(
            device,
            BufferInfo::new_mappable(buf_size, vk::BufferUsageFlags::TRANSFER_SRC),
        )?;

        fn pack_char(char: [[bool; 5]; 6], row: usize, col: usize) -> u32 {
            (char[row][col] as u32) << (col + row * 5)
        }

        let mut packed = [0u32; 96];
        for (index, char) in characters.iter().copied().enumerate() {
            for row in 0..6 {
                for col in 0..5 {
                    packed[index] |= pack_char(char, row, col);
                }
            }
        }

        Buffer::mapped_slice_mut(&mut temp_buf).copy_from_slice(cast_slice(&packed));

        let character_buf = Arc::new(Buffer::create(
            device,
            BufferInfo::new_mappable(
                buf_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            ),
        )?);

        {
            let temp_buf = render_graph.bind_node(temp_buf);
            let character_buf = render_graph.bind_node(&character_buf);

            render_graph.copy_buffer(temp_buf, character_buf);
        }

        Ok(character_buf)
    }

    fn create_framebuffer_image(device: &Arc<Device>) -> Result<Arc<Image>, DriverError> {
        Ok(Arc::new(Image::create(
            device,
            ImageInfo::new_2d(
                vk::Format::R8G8B8A8_UNORM,
                Self::FRAMEBUFFER_WIDTH,
                Self::FRAMEBUFFER_HEIGHT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            ),
        )?))
    }

    fn create_graphics_pipeline(device: &Arc<Device>) -> Result<Arc<ComputePipeline>, DriverError> {
        Ok(Arc::new(ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(
                inline_spirv!(
                    r#"
                    #version 460 core

                    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

                    layout(binding = 0, std140) uniform GraphicsUniform {
                        uvec4 data[960];
                    } graphics_buf;
                    layout(binding = 1, std140) uniform PaletteUniform {
                        uvec4 data[64];
                    } palette_buf;
                    layout(binding = 2, rgba32f) writeonly uniform image2D framebuffer_image;

                    vec3 palette_color(uint index) {
                        uint data = palette_buf.data[index >> 2][index % 4];
                        float r = float( data & 0x000000FF       ) / 255.0;
                        float g = float((data & 0x0000FF00) >>  8) / 255.0;
                        float b = float((data & 0x00FF0000) >> 16) / 255.0;

                        return vec3(r, g, b);
                    }

                    vec4 unpack_pixel(uvec4 graphics_data) {
                        uint x = gl_GlobalInvocationID.x;
                        uint color_index = graphics_data[(x >> 2) % 4];

                        uint shift = gl_GlobalInvocationID.x % 4 << 3;
                        color_index >>= shift;
                        color_index &= 0x000000FF;

                        // 0xFF is "transparent"
                        if (color_index == 0xFF) {
                            return vec4(0);
                        }

                        vec3 color = palette_color(color_index);

                        return vec4(color, 1.0);
                    }

                    void main() {
                        uint x = gl_GlobalInvocationID.x;
                        uint y = gl_GlobalInvocationID.y;

                        uvec4 graphics_data = graphics_buf.data[(x >> 4) + y * 10];
                        vec4 pixel = unpack_pixel(graphics_data);

                        if (pixel.a > 0) {
                            ivec2 coord = ivec2(x, y);
                            imageStore(framebuffer_image, coord, pixel);
                        }
                    }
                    "#,
                    comp,
                )
                .as_slice(),
            ),
        )?))
    }

    fn create_text_pipeline(device: &Arc<Device>) -> Result<Arc<ComputePipeline>, DriverError> {
        Ok(Arc::new(ComputePipeline::create(
            device,
            ComputePipelineInfo::default(),
            Shader::new_compute(
                inline_spirv!(
                    r#"
                    #version 460 core

                    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

                    layout(binding = 0, std140) uniform CharacterUniform {
                        uvec4 data[24];
                    } character_buf;
                    layout(binding = 1, std140) uniform PaletteUniform {
                        uvec4 data[64];
                    } palette_buf;
                    layout(binding = 2, std140) uniform TextUniform {
                        uvec4 data[128];
                    } text_buf;
                    layout(binding = 3, rgba32f) writeonly uniform image2D framebuffer_image;

                    vec3 palette_color(uint index) {
                        uint data = palette_buf.data[index >> 2][index % 4];
                        float r = float( data & 0x000000FF       ) / 255.0;
                        float g = float((data & 0x0000FF00) >>  8) / 255.0;
                        float b = float((data & 0x00FF0000) >> 16) / 255.0;

                        return vec3(r, g, b);
                    }

                    vec4 unpack_char(uint char_data, uint col, uint row, vec3 foreground_color, vec3 background_color) {
                        uint shift = col + (row * 5);
                        uint packed_value = char_data >> shift;
                        float value = float(packed_value & 1);

                        vec3 color = mix(background_color, foreground_color, value);

                        return vec4(color, 1.0);
                    }

                    void main() {
                        uint x = gl_GlobalInvocationID.x * 5;
                        uint y = gl_GlobalInvocationID.y * 6;

                        uint text = text_buf.data[(gl_GlobalInvocationID.x + (gl_GlobalInvocationID.y * 32)) >> 2][gl_GlobalInvocationID.x % 4];
                        uint foreground_index = (text & 0x0000FF00) >> 8;
                        uint background_index = (text & 0x00FF0000) >> 16;
                        uint char_index = (text & 0xFF) - 32;
                        uint char_data = character_buf.data[char_index >> 2][char_index % 4];

                        vec3 foreground_color = palette_color(foreground_index);
                        vec3 background_color = palette_color(background_index);

                        for (uint row = 0; row < 6; row++) {
                            for (uint col = 0; col < 5; col++) {
                                vec4 pixel = unpack_char(char_data, col, row, foreground_color, background_color);
                                ivec2 coord = ivec2(x + col, y + row);
                                imageStore(framebuffer_image, coord, pixel);
                            }
                        }
                    }
                    "#,
                    comp,
                )
                .as_slice(),
            ),
        )?))
    }

    fn deref_index_adresses(&self, index_addresses: &[usize]) -> Box<[i32]> {
        index_addresses
            .iter()
            .copied()
            .map(|index| self.stack[index].integer())
            .collect()
    }

    fn deref_subscript_addresses(&self, subscript_addresses: &[Range<usize>]) -> Box<[Range<i32>]> {
        subscript_addresses
            .iter()
            .map(|subscript| {
                self.stack[subscript.start].integer()..self.stack[subscript.end].integer()
            })
            .collect()
    }

    pub fn framebuffer_image(&self) -> &Arc<Image> {
        &self.framebuffer_images[0]
    }

    #[profiling::function]
    fn get_graphic(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, var_addr: usize, var_index: i32) {
        debug_assert!(y0 <= y1);
        debug_assert!(x0 <= x1);
        debug_assert!(x0 >= 0);
        debug_assert!(x1 >= 0);
        debug_assert!(y0 >= 0);
        debug_assert!(y1 >= 0);
        debug_assert!(x0 < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(x1 < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(y0 < Self::FRAMEBUFFER_HEIGHT as i32);
        debug_assert!(y1 < Self::FRAMEBUFFER_HEIGHT as i32);

        let width = (x1 - x0) as u32 + 1;
        let height = (y1 - y0) as u32 + 1;
        let len = width * height;
        let data = self.stack[var_addr].byte_slice_mut(var_index..var_index + len as i32);

        // TODO: Treat graphics data as u64 and copy 16x faster!

        let mut idx = 0;
        for y in y0..=y1 {
            for x in x0..=x1 {
                data[idx] =
                    self.graphics_data[y as usize * Self::FRAMEBUFFER_WIDTH as usize + x as usize];
                idx += 1;
            }
        }
    }

    #[allow(dead_code)]
    pub fn heap(&self) -> &[u8] {
        &self.heap
    }

    #[allow(dead_code)]
    pub fn heap_mut(&mut self) -> &mut [u8] {
        &mut self.heap
    }

    pub fn is_running(&self) -> bool {
        self.program_index < self.program.len()
    }

    #[profiling::function]
    pub fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u8) {
        debug_assert!(x0 >= 0);
        debug_assert!(x1 >= 0);
        debug_assert!(y0 >= 0);
        debug_assert!(y1 >= 0);
        debug_assert!(x0 < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(x1 < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(y0 < Self::FRAMEBUFFER_HEIGHT as i32);
        debug_assert!(y1 < Self::FRAMEBUFFER_HEIGHT as i32);

        // Bresenham line algorithm
        let dx = x1 - x0;
        let dy = y1 - y0;
        let abs_dx = dx.abs();
        let abs_dy = dy.abs();

        let mut x = x0;
        let mut y = y0;

        self.set_pixel(x, y, color);

        if abs_dx > abs_dy {
            let mut d = 2 * abs_dy - abs_dx;
            for _ in 0..abs_dx {
                x = if dx < 0 { x - 1 } else { x + 1 };
                if d < 0 {
                    d += 2 * abs_dy
                } else {
                    y = if dy < 0 { y - 1 } else { y + 1 };
                    d += 2 * abs_dy - 2 * abs_dx;
                }

                self.set_pixel(x, y, color);
            }
        } else {
            let mut d = 2 * abs_dx - abs_dy;
            for _ in 0..abs_dy {
                y = if dy < 0 { y - 1 } else { y + 1 };
                if d < 0 {
                    d += 2 * abs_dx;
                } else {
                    x = if dx < 0 { x - 1 } else { x + 1 };
                    d += 2 * abs_dx - 2 * abs_dy;
                }

                self.set_pixel(x, y, color);
            }
        }
    }

    #[allow(dead_code)]
    pub fn load_program(&mut self, program: Vec<Instruction>) {
        self.heap.fill(0);
        self.started_at = Instant::now();

        self.program = program;
        self.program_index = 0;
    }

    pub fn locate(&mut self, col: usize, row: usize) {
        self.location = (col.min(Self::TEXT_COLS - 1), row.min(Self::TEXT_ROWS));
    }

    #[allow(dead_code)]
    pub fn palette(&self, color: u8) -> (u8, u8, u8) {
        let color_base = (color as usize) << 2;
        let r = self.palette_data[color_base];
        let g = self.palette_data[color_base + 1];
        let b = self.palette_data[color_base + 2];

        (r, g, b)
    }

    #[allow(dead_code)]
    pub fn pixel(&mut self, x: i32, y: i32) -> u8 {
        debug_assert!(x >= 0);
        debug_assert!(y >= 0);
        debug_assert!(x < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(y < Self::FRAMEBUFFER_HEIGHT as i32);

        self.graphics_data[x as usize + y as usize * Self::FRAMEBUFFER_WIDTH as usize]
    }

    #[profiling::function]
    pub fn print(&mut self, s: &str) {
        let text_area = Self::TEXT_COLS * Self::TEXT_ROWS;

        let (foreground, background) = self.color;
        let (col, row) = self.location;

        let chars = s.as_bytes();
        let mut char_base = 0;

        // Calculate the final row which becomes the new location after printing
        let mut final_row = {
            let advance = chars.is_empty()
                || col + chars.len() != Self::TEXT_COLS
                || col + chars.len() % Self::TEXT_COLS != 0;

            (col + chars.len()) / Self::TEXT_COLS + row + advance as usize
        };

        // Find the location of the first character in the text area
        let mut start = col + row.min(Self::TEXT_ROWS - 1) * Self::TEXT_COLS;

        // If the final row is beyond the screen then we may need to scroll the retained characters
        // up a bit so they remain on-screen
        if final_row > Self::TEXT_ROWS {
            let rows = (final_row - row).min(Self::TEXT_ROWS);
            let scroll = Self::TEXT_ROWS - rows;

            final_row = Self::TEXT_ROWS;
            start -= (rows - 1) * Self::TEXT_COLS;

            if scroll > 0 {
                self.text_data.copy_within(
                    (row - scroll) * Self::TEXT_COLS * 4..(row * Self::TEXT_COLS * 4),
                    0,
                );
                self.text_data[scroll * Self::TEXT_COLS * 4..text_area * 4].fill(0);
                self.text_dirty = true;
            }
        }

        // Handle the case of printing more than the text area can hold; we truncate the start
        if chars.len() > text_area {
            char_base = chars.len() - text_area;
        }

        // From here out start refers to byte index not character location
        start *= 4;

        for char in chars[char_base..].iter().copied() {
            debug_assert!(char >= 32);
            debug_assert!(char < 127);

            let end = start + 4;
            self.text_data[start..end].copy_from_slice(&[char, foreground, background, 0x00]);
            self.text_dirty = true;
            start = end;
        }

        self.location = (0, final_row);
    }

    #[profiling::function]
    fn put_graphic(
        &mut self,
        position: (i32, i32),
        size: (i32, i32),
        var_addr: usize,
        var_index: i32,
        blend_fn: fn(u8, &mut u8),
    ) {
        let (x, y) = position;
        let (width, height) = size;

        debug_assert!(width > 0);
        debug_assert!(height > 0);

        let src_len = width * height;
        let data = self.stack[var_addr].byte_slice(var_index..var_index + src_len);

        // TODO: Treat graphics data as u64 and blend 16x faster!

        let src_x = (0 - x).max(0)..(Self::FRAMEBUFFER_WIDTH as i32 - x).min(width);

        if src_x.is_empty() {
            return;
        }

        let src_y = (0 - y).max(0)..(Self::FRAMEBUFFER_HEIGHT as i32 - y).min(height);
        self.graphics_dirty = !src_y.is_empty();

        for src_y in src_y {
            let dst_y = x + ((y + src_y) * Self::FRAMEBUFFER_WIDTH as i32);
            let src_y = src_y * width;
            for src_x in src_x.clone() {
                let src = data[(src_y + src_x) as usize];
                let dst = &mut self.graphics_data[(dst_y + src_x) as usize];
                blend_fn(src, dst);
            }
        }
    }

    fn put_graphic_and(
        &mut self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        var_addr: usize,
        var_index: i32,
    ) {
        self.put_graphic((x, y), (width, height), var_addr, var_index, |a, b| *b &= a);
    }

    fn put_graphic_or(
        &mut self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        var_addr: usize,
        var_index: i32,
    ) {
        self.put_graphic((x, y), (width, height), var_addr, var_index, |a, b| *b |= a);
    }

    fn put_graphic_pset(
        &mut self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        var_addr: usize,
        var_index: i32,
    ) {
        self.put_graphic((x, y), (width, height), var_addr, var_index, |a, b| *b = a);
    }

    fn put_graphic_preset(
        &mut self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        var_addr: usize,
        var_index: i32,
    ) {
        self.put_graphic((x, y), (width, height), var_addr, var_index, |a, b| *b = !a);
    }

    fn put_graphic_tset(
        &mut self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        var_addr: usize,
        var_index: i32,
    ) {
        self.put_graphic((x, y), (width, height), var_addr, var_index, |a, b| {
            // Only set the color if it is not the default background color (transparent)
            if a != Self::DEFAULT_COLOR.1 {
                *b = a
            }
        });
    }

    fn put_graphic_xor(
        &mut self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        var_addr: usize,
        var_index: i32,
    ) {
        self.put_graphic((x, y), (width, height), var_addr, var_index, |a, b| *b ^= a);
    }

    #[profiling::function]
    pub fn rectangle(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u8, is_filled: bool) {
        debug_assert!(x0 >= 0);
        debug_assert!(x1 >= 0);
        debug_assert!(y0 >= 0);
        debug_assert!(y1 >= 0);
        debug_assert!(x0 < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(x1 < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(y0 < Self::FRAMEBUFFER_HEIGHT as i32);
        debug_assert!(y1 < Self::FRAMEBUFFER_HEIGHT as i32);

        // TODO: Treat graphics data as u64 and fill 16x faster!

        if is_filled {
            for y in y0..=y1 {
                let y = y as usize * Self::FRAMEBUFFER_WIDTH as usize;
                for x in x0..=x1 {
                    let x = x as usize;
                    self.graphics_data[x + y] = color;
                }
            }
        } else {
            let y_offset0 = y0 as usize * Self::FRAMEBUFFER_WIDTH as usize;
            let y_offset1 = y1 as usize * Self::FRAMEBUFFER_WIDTH as usize;
            for x in x0..=x1 {
                let x = x as usize;
                self.graphics_data[x + y_offset0] = color;
                self.graphics_data[x + y_offset1] = color;
            }

            let x0 = x0 as usize;
            let x1 = x1 as usize;
            for y in y0..=y1 {
                self.graphics_data[x0 + y as usize * Self::FRAMEBUFFER_WIDTH as usize] = color;
                self.graphics_data[x1 + y as usize * Self::FRAMEBUFFER_WIDTH as usize] = color;
            }
        }

        self.graphics_dirty = true;
    }

    pub fn set_palette(&mut self, color_index: u8, r: u8, g: u8, b: u8) {
        debug_assert_ne!(color_index, 0xFF);

        let base = (color_index as usize) << 2;

        self.palette_data[base..base + 3].copy_from_slice(&[r, g, b]);
        self.palette_dirty = true;
    }

    pub fn set_pixel(&mut self, x: i32, y: i32, color: u8) {
        debug_assert!(x >= 0);
        debug_assert!(y >= 0);
        debug_assert!(x < Self::FRAMEBUFFER_WIDTH as i32);
        debug_assert!(y < Self::FRAMEBUFFER_HEIGHT as i32);

        self.graphics_data[x as usize + y as usize * Self::FRAMEBUFFER_WIDTH as usize] = color;
        self.graphics_dirty = true;
    }

    fn swap_framebuffer_images(&mut self) {
        self.framebuffer_images.swap(0, 1);
    }

    #[profiling::function]
    pub fn update(
        &mut self,
        render_graph: &mut RenderGraph,
        events: &[Event<()>],
    ) -> Result<(), DriverError> {
        update_keyboard(&mut self.keyboard, events);

        self.tick();

        let previous_framebuffer_image = render_graph.bind_node(&self.framebuffer_images[0]);
        let framebuffer_image = render_graph.bind_node(&self.framebuffer_images[1]);

        render_graph.copy_image(previous_framebuffer_image, framebuffer_image);

        if self.palette_dirty {
            self.palette_dirty = false;
            self.text_dirty = true;
            self.graphics_dirty = true;

            let mut buf = self.pool.lease(BufferInfo::new_mappable(
                self.palette_data.len() as _,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            ))?;
            Buffer::copy_from_slice(&mut buf, 0, &self.palette_data);

            self.palette_buf = Arc::new(buf);
        }

        let palette_buf = render_graph.bind_node(&self.palette_buf);

        if self.text_dirty {
            self.graphics_dirty = true;
            self.text_dirty = false;

            let mut buf = self.pool.lease(BufferInfo::new_mappable(
                self.text_data.len() as _,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            ))?;
            Buffer::copy_from_slice(&mut buf, 0, &self.text_data);

            self.text_buf = Arc::new(buf);

            let character_buf = render_graph.bind_node(&self.character_buf);
            let text_buf = render_graph.bind_node(&self.text_buf);

            render_graph
                .begin_pass("Text")
                .bind_pipeline(&self.text_pipeline)
                .access_descriptor(0, character_buf, AccessType::ComputeShaderReadUniformBuffer)
                .access_descriptor(1, palette_buf, AccessType::ComputeShaderReadUniformBuffer)
                .access_descriptor(2, text_buf, AccessType::ComputeShaderReadUniformBuffer)
                .access_descriptor(3, framebuffer_image, AccessType::ComputeShaderWrite)
                .record_compute(move |compute, _| {
                    compute.dispatch(Self::TEXT_COLS as _, Self::TEXT_ROWS as _, 1);
                });
        }

        if self.graphics_dirty {
            self.graphics_dirty = false;

            let mut buf = self.pool.lease(BufferInfo::new_mappable(
                self.graphics_data.len() as _,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            ))?;
            Buffer::copy_from_slice(&mut buf, 0, &self.graphics_data);

            self.graphics_buf = Arc::new(buf);

            let graphics_buf = render_graph.bind_node(&self.graphics_buf);

            render_graph
                .begin_pass("Graphics")
                .bind_pipeline(&self.graphics_pipeline)
                .access_descriptor(0, graphics_buf, AccessType::ComputeShaderReadUniformBuffer)
                .access_descriptor(1, palette_buf, AccessType::ComputeShaderReadUniformBuffer)
                .access_descriptor(2, framebuffer_image, AccessType::ComputeShaderWrite)
                .record_compute(move |compute, _| {
                    compute.dispatch(
                        Self::FRAMEBUFFER_WIDTH as _,
                        Self::FRAMEBUFFER_HEIGHT as _,
                        1,
                    );
                });
        }

        self.swap_framebuffer_images();

        Ok(())
    }

    #[profiling::function]
    fn tick(&mut self) {
        while let Some(instr) = self.program.get(self.program_index) {
            trace!("Executing {:?}", instr);

            match instr {
                &Instruction::AddBytes(a, b, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[a].byte() + self.stack[b].byte());
                }
                &Instruction::AddFloats(a, b, dst) => {
                    self.stack[dst] =
                        Value::Float(self.stack[a].float().to_owned() + self.stack[b].float());
                }
                &Instruction::AddIntegers(a, b, dst) => {
                    self.stack[dst] = Value::Integer(
                        self.stack[a].integer().to_owned() + self.stack[b].integer(),
                    );
                }
                &Instruction::AddStrings(a, b, dst) => {
                    self.stack[dst] =
                        Value::String(self.stack[a].string().to_owned() + self.stack[b].string());
                }

                &Instruction::SubtractBytes(a, b, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[a].byte() - self.stack[b].byte());
                }
                &Instruction::SubtractFloats(a, b, dst) => {
                    self.stack[dst] =
                        Value::Float(self.stack[a].float().to_owned() - self.stack[b].float());
                }
                &Instruction::SubtractIntegers(a, b, dst) => {
                    self.stack[dst] = Value::Integer(
                        self.stack[a].integer().to_owned() - self.stack[b].integer(),
                    );
                }

                &Instruction::DivideBytes(a, b, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[a].byte() / self.stack[b].byte());
                }
                &Instruction::DivideFloats(a, b, dst) => {
                    self.stack[dst] =
                        Value::Float(self.stack[a].float().to_owned() / self.stack[b].float());
                }
                &Instruction::DivideIntegers(a, b, dst) => {
                    self.stack[dst] = Value::Integer(
                        self.stack[a].integer().to_owned() / self.stack[b].integer(),
                    );
                }

                &Instruction::MultiplyBytes(a, b, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[a].byte() * self.stack[b].byte());
                }
                &Instruction::MultiplyFloats(a, b, dst) => {
                    self.stack[dst] =
                        Value::Float(self.stack[a].float().to_owned() * self.stack[b].float());
                }
                &Instruction::MultiplyIntegers(a, b, dst) => {
                    self.stack[dst] = Value::Integer(
                        self.stack[a].integer().to_owned() * self.stack[b].integer(),
                    );
                }

                &Instruction::ModulusBytes(val_addr, mod_addr, dst_addr) => {
                    self.stack[dst_addr] =
                        Value::Byte(self.stack[val_addr].byte() % self.stack[mod_addr].byte());
                }
                &Instruction::ModulusFloats(val_addr, mod_addr, dst_addr) => {
                    self.stack[dst_addr] =
                        Value::Float(self.stack[val_addr].float() % self.stack[mod_addr].float());
                }
                &Instruction::ModulusIntegers(val_addr, mod_addr, dst_addr) => {
                    self.stack[dst_addr] = Value::Integer(
                        self.stack[val_addr].integer() % self.stack[mod_addr].integer(),
                    );
                }

                &Instruction::ConvertBooleanToByte(src, dst) => {
                    self.stack[dst] = Value::Byte(u8::from(self.stack[src].boolean()))
                }
                &Instruction::ConvertBooleanToFloat(src, dst) => {
                    self.stack[dst] =
                        Value::Float(if self.stack[src].boolean() { 1.0 } else { 0.0 })
                }
                &Instruction::ConvertBooleanToInteger(src, dst) => {
                    self.stack[dst] = Value::Integer(i32::from(self.stack[src].boolean()))
                }
                &Instruction::ConvertBooleanToString(src, dst) => {
                    self.stack[dst] = Value::String(
                        if self.stack[src].boolean() {
                            "TRUE"
                        } else {
                            "FALSE"
                        }
                        .to_owned(),
                    )
                }

                &Instruction::ConvertByteToBoolean(src, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[src].byte() != 0)
                }
                &Instruction::ConvertByteToFloat(src, dst) => {
                    self.stack[dst] = Value::Float(self.stack[src].byte() as _)
                }
                &Instruction::ConvertByteToInteger(src, dst) => {
                    self.stack[dst] = Value::Integer(self.stack[src].byte() as _)
                }
                &Instruction::ConvertByteToString(src, dst) => {
                    self.stack[dst] = Value::String(self.stack[src].byte().to_string())
                }

                &Instruction::ConvertFloatToBoolean(src, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[src].float() != 0.0)
                }
                &Instruction::ConvertFloatToByte(src, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[src].float() as _)
                }
                &Instruction::ConvertFloatToInteger(src, dst) => {
                    self.stack[dst] = Value::Integer(self.stack[src].float() as _)
                }
                &Instruction::ConvertFloatToString(src, dst) => {
                    self.stack[dst] = Value::String(self.stack[src].float().to_string())
                }

                &Instruction::ConvertIntegerToBoolean(src, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[src].integer() != 0)
                }
                &Instruction::ConvertIntegerToByte(src, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[src].integer() as _)
                }
                &Instruction::ConvertIntegerToFloat(src, dst) => {
                    self.stack[dst] = Value::Float(self.stack[src].integer() as _)
                }
                &Instruction::ConvertIntegerToString(src, dst) => {
                    self.stack[dst] = Value::String(self.stack[src].integer().to_string())
                }

                &Instruction::ConvertStringToBoolean(src, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[src].string() == "TRUE")
                }
                &Instruction::ConvertStringToByte(src, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[src].string().parse().unwrap_or_default())
                }
                &Instruction::ConvertStringToFloat(src, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[src].string().parse().unwrap_or_default())
                }
                &Instruction::ConvertStringToInteger(src, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[src].string().parse().unwrap_or_default())
                }

                &Instruction::NotBoolean(src, dst) => {
                    self.stack[dst] = Value::Boolean(!self.stack[src].boolean());
                }
                &Instruction::AndBooleans(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].boolean() && self.stack[b].boolean());
                }
                &Instruction::OrBooleans(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].boolean() || self.stack[b].boolean());
                }
                &Instruction::XorBooleans(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].boolean() ^ self.stack[b].boolean());
                }

                &Instruction::NotByte(src, dst) => {
                    self.stack[dst] = Value::Byte(!self.stack[src].byte());
                }
                &Instruction::AndBytes(a, b, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[a].byte() & self.stack[b].byte());
                }
                &Instruction::OrBytes(a, b, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[a].byte() | self.stack[b].byte());
                }
                &Instruction::XorBytes(a, b, dst) => {
                    self.stack[dst] = Value::Byte(self.stack[a].byte() ^ self.stack[b].byte());
                }

                &Instruction::NotInteger(src, dst) => {
                    self.stack[dst] = Value::Integer(!self.stack[src].integer());
                }
                &Instruction::AndIntegers(a, b, dst) => {
                    self.stack[dst] =
                        Value::Integer(self.stack[a].integer() & self.stack[b].integer());
                }
                &Instruction::OrIntegers(a, b, dst) => {
                    self.stack[dst] =
                        Value::Integer(self.stack[a].integer() | self.stack[b].integer());
                }
                &Instruction::XorIntegers(a, b, dst) => {
                    self.stack[dst] =
                        Value::Integer(self.stack[a].integer() ^ self.stack[b].integer());
                }

                &Instruction::EqualBooleans(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].boolean() == self.stack[b].boolean());
                }
                &Instruction::EqualBytes(a, b, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[a].byte() == self.stack[b].byte());
                }
                &Instruction::EqualFloats(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].float() == self.stack[b].float());
                }
                &Instruction::EqualIntegers(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].integer() == self.stack[b].integer());
                }
                &Instruction::EqualStrings(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].string() == self.stack[b].string());
                }

                &Instruction::NotEqualBooleans(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].boolean() != self.stack[b].boolean());
                }
                &Instruction::NotEqualBytes(a, b, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[a].byte() != self.stack[b].byte());
                }
                &Instruction::NotEqualFloats(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].float() != self.stack[b].float());
                }
                &Instruction::NotEqualIntegers(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].integer() != self.stack[b].integer());
                }
                &Instruction::NotEqualStrings(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].string() != self.stack[b].string());
                }

                &Instruction::GreaterThanEqualBytes(a, b, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[a].byte() >= self.stack[b].byte());
                }
                &Instruction::GreaterThanEqualFloats(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].float() >= self.stack[b].float());
                }
                &Instruction::GreaterThanEqualIntegers(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].integer() >= self.stack[b].integer());
                }

                &Instruction::GreaterThanBytes(a, b, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[a].byte() > self.stack[b].byte());
                }
                &Instruction::GreaterThanFloats(a, b, dst) => {
                    self.stack[dst] = Value::Boolean(self.stack[a].float() > self.stack[b].float());
                }
                &Instruction::GreaterThanIntegers(a, b, dst) => {
                    self.stack[dst] =
                        Value::Boolean(self.stack[a].integer() > self.stack[b].integer());
                }

                &Instruction::AbsFloat(src, dst) => {
                    self.stack[dst] = Value::Float(self.stack[src].float().abs());
                }
                &Instruction::AbsInteger(src, dst) => {
                    self.stack[dst] = Value::Integer(self.stack[src].integer().abs());
                }
                &Instruction::Cos(src, dst) => {
                    self.stack[dst] = Value::Float(self.stack[src].float().cos());
                }
                &Instruction::Sin(src, dst) => {
                    self.stack[dst] = Value::Float(self.stack[src].float().sin());
                }
                Instruction::ClearScreen => self.cls(),
                &Instruction::Color(foreground_addr, background_addr) => self.color(
                    self.stack[foreground_addr].byte(),
                    self.stack[background_addr].byte(),
                ),
                &Instruction::KeyDown(key, dst) => {
                    self.stack[dst] = Value::Boolean(
                        self.keyboard
                            .is_pressed(virtual_key_code(self.stack[key].byte()))
                            || self
                                .keyboard
                                .is_held(virtual_key_code(self.stack[key].byte())),
                    );
                }
                &Instruction::Locate(col, row) => self.locate(
                    self.stack[col].integer() as _,
                    self.stack[row].integer() as _,
                ),
                &Instruction::Line(from_x, from_y, to_x, to_y, color) => {
                    self.line(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[to_x].integer(),
                        self.stack[to_y].integer(),
                        self.stack[color].byte(),
                    );
                }
                &Instruction::Palette(color_index, r, g, b) => {
                    self.set_palette(
                        self.stack[color_index].byte(),
                        self.stack[r].byte(),
                        self.stack[g].byte(),
                        self.stack[b].byte(),
                    );
                }
                &Instruction::Random(addr) => {
                    self.stack[addr] = Value::Float(random());
                }
                &Instruction::Rectangle(from_x, from_y, to_x, to_y, color, is_filled) => {
                    self.rectangle(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[to_x].integer(),
                        self.stack[to_y].integer(),
                        self.stack[color].byte(),
                        self.stack[is_filled].boolean(),
                    );
                }
                &Instruction::SetPixel(x, y, color) => {
                    self.set_pixel(
                        self.stack[x].integer(),
                        self.stack[y].integer(),
                        self.stack[color].byte(),
                    );
                }
                &Instruction::PrintString(addr) => {
                    self.print(self.stack[addr].string().to_owned().as_str());
                }
                &Instruction::Timer(dst) => {
                    self.stack[dst] = Value::Integer(
                        Instant::now()
                            .duration_since(self.started_at)
                            .as_micros()
                            .clamp(0, i32::MAX as _) as _,
                    );
                }

                &Instruction::GetGraphic(from_x, from_y, to_x, to_y, var_addr, var_index_addr) => {
                    self.get_graphic(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[to_x].integer(),
                        self.stack[to_y].integer(),
                        var_addr,
                        self.stack[var_index_addr].integer(),
                    );
                }
                &Instruction::PutGraphicAnd(
                    from_x,
                    from_y,
                    width,
                    height,
                    var_addr,
                    var_index_addr,
                ) => {
                    self.put_graphic_and(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[width].integer(),
                        self.stack[height].integer(),
                        var_addr,
                        self.stack[var_index_addr].integer(),
                    );
                }
                &Instruction::PutGraphicOr(
                    from_x,
                    from_y,
                    width,
                    height,
                    var_addr,
                    var_index_addr,
                ) => {
                    self.put_graphic_or(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[width].integer(),
                        self.stack[height].integer(),
                        var_addr,
                        self.stack[var_index_addr].integer(),
                    );
                }
                &Instruction::PutGraphicPset(
                    from_x,
                    from_y,
                    width,
                    height,
                    var_addr,
                    var_index_addr,
                ) => {
                    self.put_graphic_pset(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[width].integer(),
                        self.stack[height].integer(),
                        var_addr,
                        self.stack[var_index_addr].integer(),
                    );
                }
                &Instruction::PutGraphicPreset(
                    from_x,
                    from_y,
                    width,
                    height,
                    var_addr,
                    var_index_addr,
                ) => {
                    self.put_graphic_preset(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[width].integer(),
                        self.stack[height].integer(),
                        var_addr,
                        self.stack[var_index_addr].integer(),
                    );
                }
                &Instruction::PutGraphicTset(
                    from_x,
                    from_y,
                    width,
                    height,
                    var_addr,
                    var_index_addr,
                ) => {
                    self.put_graphic_tset(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[width].integer(),
                        self.stack[height].integer(),
                        var_addr,
                        self.stack[var_index_addr].integer(),
                    );
                }
                &Instruction::PutGraphicXor(
                    from_x,
                    from_y,
                    width,
                    height,
                    var_addr,
                    var_index_addr,
                ) => {
                    self.put_graphic_xor(
                        self.stack[from_x].integer(),
                        self.stack[from_y].integer(),
                        self.stack[width].integer(),
                        self.stack[height].integer(),
                        var_addr,
                        self.stack[var_index_addr].integer(),
                    );
                }

                &Instruction::PeekBoolean(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.stack[stack_addr] = Value::Boolean(self.heap[heap_addr] == 1);
                }
                &Instruction::PeekByte(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.stack[stack_addr] = Value::Byte(self.heap[heap_addr]);
                }
                &Instruction::PeekFloat(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.stack[stack_addr] = Value::Float(f32::from_ne_bytes(
                        self.heap[heap_addr..heap_addr + size_of::<f32>()]
                            .try_into()
                            .unwrap(),
                    ));
                }
                &Instruction::PeekInteger(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.stack[stack_addr] = Value::Integer(i32::from_ne_bytes(
                        self.heap[heap_addr..heap_addr + size_of::<i32>()]
                            .try_into()
                            .unwrap(),
                    ));
                }
                &Instruction::PeekString(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    let mut end = heap_addr;
                    loop {
                        let char = self.heap[end];
                        if !(32..=127).contains(&char) {
                            break;
                        }

                        end += 1;
                    }

                    self.stack[stack_addr] = Value::String(
                        String::from_utf8(self.heap[heap_addr..end].to_vec()).unwrap(),
                    );
                }

                &Instruction::PokeBoolean(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.heap[heap_addr] = if self.stack[stack_addr].boolean() {
                        0x01
                    } else {
                        0x00
                    };
                }
                &Instruction::PokeByte(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.heap[heap_addr] = self.stack[stack_addr].byte();
                }
                &Instruction::PokeFloat(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.heap[heap_addr..heap_addr + size_of::<f32>()]
                        .copy_from_slice(&self.stack[stack_addr].float().to_ne_bytes());
                }
                &Instruction::PokeInteger(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    self.heap[heap_addr..heap_addr + size_of::<i32>()]
                        .copy_from_slice(&self.stack[stack_addr].integer().to_ne_bytes());
                }
                &Instruction::PokeString(heap_addr, stack_addr) => {
                    let heap_addr = self.stack[heap_addr].integer() as usize;
                    let str = self.stack[stack_addr].string();
                    let str = str.as_bytes();

                    self.heap[heap_addr..heap_addr + str.len()].copy_from_slice(str);
                    self.heap[heap_addr + str.len() + 1] = 0;
                }

                &Instruction::Branch(addr, program_index) => {
                    if self.stack[addr].boolean() {
                        self.program_index = program_index;
                        continue;
                    }
                }
                &Instruction::BranchNot(addr, program_index) => {
                    if !self.stack[addr].boolean() {
                        self.program_index = program_index;
                        continue;
                    }
                }
                Instruction::End => {
                    self.program_index = self.program.len();
                    break;
                }
                &Instruction::Jump(program_index) => {
                    self.program_index = program_index;
                    continue;
                }
                Instruction::Yield => {
                    self.program_index += 1;
                    break;
                }

                &Instruction::Copy(src, dst) => self.stack[dst] = self.stack[src].clone(),
                Instruction::ReadBooleans(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    self.stack[*dst] = Value::Boolean(*self.stack[*src].boolean_mut(&indexes));
                }
                Instruction::ReadBytes(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    self.stack[*dst] = Value::Byte(*self.stack[*src].byte_mut(&indexes));
                }
                Instruction::ReadFloats(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    self.stack[*dst] = Value::Float(*self.stack[*src].float_mut(&indexes));
                }
                Instruction::ReadIntegers(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    self.stack[*dst] = Value::Integer(*self.stack[*src].integer_mut(&indexes));
                }
                Instruction::ReadStrings(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    self.stack[*dst] =
                        Value::String(self.stack[*src].string_mut(&indexes).to_owned());
                }
                Instruction::WriteBooleans(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    *self.stack[*dst].boolean_mut(&indexes) = self.stack[*src].boolean();
                }
                Instruction::WriteBytes(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    *self.stack[*dst].byte_mut(&indexes) = self.stack[*src].byte();
                }
                Instruction::WriteFloats(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    *self.stack[*dst].float_mut(&indexes) = self.stack[*src].float();
                }
                Instruction::WriteIntegers(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    *self.stack[*dst].integer_mut(&indexes) = self.stack[*src].integer();
                }
                Instruction::WriteStrings(src, index_addresses, dst) => {
                    let indexes = self.deref_index_adresses(index_addresses);

                    *self.stack[*dst].string_mut(&indexes) = self.stack[*src].string().to_owned();
                }

                Instruction::DimensionBooleans(subscript_addresses, dst) => {
                    let subscripts = self.deref_subscript_addresses(subscript_addresses);
                    let data_len = linear_len_from_multi(&subscripts);
                    let data = vec![false; data_len].into();

                    self.stack[*dst] = Value::Booleans(subscripts, data)
                }
                Instruction::DimensionBytes(subscript_addresses, dst) => {
                    let subscripts = self.deref_subscript_addresses(subscript_addresses);
                    let data_len = linear_len_from_multi(&subscripts);
                    let data = vec![0u8; data_len].into();

                    self.stack[*dst] = Value::Bytes(subscripts, data)
                }
                Instruction::DimensionFloats(subscript_addresses, dst) => {
                    let subscripts = self.deref_subscript_addresses(subscript_addresses);
                    let data_len = linear_len_from_multi(&subscripts);
                    let data = vec![0f32; data_len].into();

                    self.stack[*dst] = Value::Floats(subscripts, data)
                }
                Instruction::DimensionIntegers(subscript_addresses, dst) => {
                    let subscripts = self.deref_subscript_addresses(subscript_addresses);
                    let data_len = linear_len_from_multi(&subscripts);
                    let data = vec![0i32; data_len].into();

                    self.stack[*dst] = Value::Integers(subscripts, data)
                }
                Instruction::DimensionStrings(subscript_addresses, dst) => {
                    let subscripts = self.deref_subscript_addresses(subscript_addresses);
                    let data_len = linear_len_from_multi(&subscripts);
                    let data = vec!["".to_owned(); data_len].into();

                    self.stack[*dst] = Value::Strings(subscripts, data)
                }

                &Instruction::WriteBoolean(val, dst) => self.stack[dst] = Value::Boolean(val),
                &Instruction::WriteByte(val, dst) => self.stack[dst] = Value::Byte(val),
                &Instruction::WriteFloat(val, dst) => self.stack[dst] = Value::Float(val),
                &Instruction::WriteInteger(val, dst) => self.stack[dst] = Value::Integer(val),
                Instruction::WriteString(val, dst) => self.stack[*dst] = Value::String(val.clone()),
            }

            self.program_index += 1;
        }
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Boolean(bool),
    Booleans(Box<[Range<i32>]>, Box<[bool]>),
    Byte(u8),
    Bytes(Box<[Range<i32>]>, Box<[u8]>),
    Float(f32),
    Floats(Box<[Range<i32>]>, Box<[f32]>),
    Integer(i32),
    Integers(Box<[Range<i32>]>, Box<[i32]>),
    String(String),
    Strings(Box<[Range<i32>]>, Box<[String]>),
}

impl Value {
    fn boolean(&self) -> bool {
        if let Self::Boolean(val) = self {
            *val
        } else {
            unreachable!();
        }
    }

    fn boolean_mut(&mut self, indexes: &[i32]) -> &mut bool {
        if let Self::Booleans(subscripts, data) = self {
            &mut data[index_multi_to_linear(indexes, subscripts)]
        } else {
            unreachable!();
        }
    }

    fn byte(&self) -> u8 {
        if let Self::Byte(val) = self {
            *val
        } else {
            unreachable!();
        }
    }

    fn byte_mut(&mut self, indexes: &[i32]) -> &mut u8 {
        if let Self::Bytes(subscripts, data) = self {
            &mut data[index_multi_to_linear(indexes, subscripts)]
        } else {
            unreachable!();
        }
    }

    fn byte_slice(&self, range: Range<i32>) -> &[u8] {
        if let Self::Bytes(subscripts, data) = self {
            debug_assert_eq!(1, subscripts.len());

            &data[index_multi_to_linear(&[range.start], subscripts)
                ..index_multi_to_linear(&[range.end], subscripts)]
        } else {
            unreachable!()
        }
    }

    fn byte_slice_mut(&mut self, range: Range<i32>) -> &mut [u8] {
        if let Self::Bytes(subscripts, data) = self {
            debug_assert_eq!(1, subscripts.len());

            &mut data[index_multi_to_linear(&[range.start], subscripts)
                ..index_multi_to_linear(&[range.end], subscripts)]
        } else {
            unreachable!()
        }
    }

    fn float(&self) -> f32 {
        if let Self::Float(val) = self {
            *val
        } else {
            unreachable!();
        }
    }

    fn float_mut(&mut self, indexes: &[i32]) -> &mut f32 {
        if let Self::Floats(subscripts, data) = self {
            &mut data[index_multi_to_linear(indexes, subscripts)]
        } else {
            unreachable!();
        }
    }

    fn integer(&self) -> i32 {
        if let Self::Integer(val) = self {
            *val
        } else {
            unreachable!();
        }
    }

    fn integer_mut(&mut self, indexes: &[i32]) -> &mut i32 {
        if let Self::Integers(subscripts, data) = self {
            &mut data[index_multi_to_linear(indexes, subscripts)]
        } else {
            unreachable!();
        }
    }

    fn string(&self) -> &str {
        if let Self::String(val) = self {
            val
        } else {
            unreachable!();
        }
    }

    fn string_mut(&mut self, indexes: &[i32]) -> &mut String {
        if let Self::Strings(subscripts, data) = self {
            &mut data[index_multi_to_linear(indexes, subscripts)]
        } else {
            unreachable!();
        }
    }
}
