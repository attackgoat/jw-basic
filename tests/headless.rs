use {
    jw_basic::eval::{ascii_5x6, vga_256, Instruction, Interpreter},
    lazy_static::lazy_static,
    screen_13::prelude::*,
    std::{
        fs::read,
        path::{Path, PathBuf},
        sync::Arc,
    },
};

lazy_static! {
    static ref CARGO_MANIFEST_DIR: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    static ref BITMAP_DIR: PathBuf = CARGO_MANIFEST_DIR.join("tests/bitmap");
    static ref PROGRAM_DIR: PathBuf = CARGO_MANIFEST_DIR.join("tests/program");
}

#[test]
fn array() {
    let res = Headless::execute("array.bas");

    res.assert_printed((15, 0), (0, 0), "90");
    res.assert_printed((15, 0), (1, 0), "91");
    res.assert_printed((15, 0), (2, 0), "92");
    res.assert_printed((15, 0), (3, 0), "93");
    res.assert_printed((15, 0), (4, 0), "94");
    res.assert_printed((15, 0), (5, 0), "95");
    res.assert_printed((15, 0), (6, 0), "96");
    res.assert_printed((15, 0), (7, 0), "97");
    res.assert_printed((15, 0), (8, 0), "98");
    res.assert_printed((15, 0), (9, 0), "99");
    res.assert_printed((15, 0), (10, 0), "100");
    res.assert_printed((15, 0), (11, 0), "101");
    res.assert_printed((15, 0), (12, 0), "TRUE FALSE 170 0 4.2 0 :)  !");
}

#[test]
fn assign() {
    let res = Headless::execute("assign.bas");

    res.assert_printed((15, 0), (0, 0), "42");
    res.assert_printed((15, 0), (1, 0), "43");
    res.assert_printed((15, 0), (2, 0), "44");
}

#[test]
fn cls() {
    let mut res = Headless::execute("cls.bas");

    res.assert_printed((15, 0), (0, 0), "one");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "two");
}

#[test]
fn color() {
    let res = Headless::execute("color.bas");

    res.assert_printed((14, 4), (0, 0), "   !DANGER!   ");
    res.assert_printed((4, 0), (1, 1), "MAY EXPLODE!");
    res.assert_printed((7, 0), (3, 0), "(do not shake)");
    res.assert_printed((2, 0), (4, 0), "GREEN");
}

#[test]
fn dim() {
    let res = Headless::execute("dim.bas");

    res.assert_printed((15, 0), (0, 0), "TRUE");
    res.assert_printed((15, 0), (1, 0), "0  0");
    res.assert_printed((15, 0), (2, 0), "00");
    res.assert_printed((15, 0), (3, 0), "Hello!");
    res.assert_printed((15, 0), (4, 0), "CAB!");
    res.assert_printed((15, 0), (5, 0), "1 22 0");
    res.assert_printed((15, 0), (6, 0), "1 2 3");
}

#[test]
fn do_loop() {
    let res = Headless::execute("do_loop.bas");

    res.assert_printed((15, 0), (0, 0), "1                               ");
    res.assert_printed((15, 0), (1, 0), "2                               ");
    res.assert_printed((15, 0), (2, 0), "                                ");
    res.assert_printed((15, 0), (3, 0), "1                               ");
    res.assert_printed((15, 0), (4, 0), "0                               ");
    res.assert_printed((15, 0), (5, 0), "                                ");
    res.assert_printed((15, 0), (6, 0), "1                               ");
    res.assert_printed((15, 0), (7, 0), "2                               ");
    res.assert_printed((15, 0), (8, 0), "                                ");
    res.assert_printed((15, 0), (9, 0), "1                               ");
    res.assert_printed((15, 0), (10, 0), "0                               ");
    res.assert_printed((15, 0), (11, 0), "                                ");
    res.assert_printed((15, 0), (12, 0), "1                               ");
    res.assert_printed((15, 0), (13, 0), "2                               ");
}

#[test]
fn exit() {
    let res = Headless::execute("exit.bas");

    res.assert_printed((15, 0), (0, 0), "1                               ");
    res.assert_printed((15, 0), (1, 0), "2                               ");
    res.assert_printed((15, 0), (2, 0), "3                               ");
    res.assert_printed((15, 0), (3, 0), "4                               ");
    res.assert_printed((15, 0), (4, 0), "5                               ");
    res.assert_printed((15, 0), (5, 0), "6                               ");
    res.assert_printed((15, 0), (6, 0), "7                               ");
    res.assert_printed((15, 0), (7, 0), "8                               ");
    res.assert_printed((15, 0), (8, 0), "9                               ");
}

#[test]
fn for_next() {
    let res = Headless::execute("for_next.bas");

    res.assert_printed((15, 0), (0, 0), "0");
    res.assert_printed((15, 0), (1, 0), "1");
    res.assert_printed((15, 0), (2, 0), "2");
    res.assert_printed((15, 0), (3, 0), "3");
    res.assert_printed((4, 0), (4, 0), "0");
    res.assert_printed((4, 0), (5, 0), "2");
    res.assert_printed((1, 0), (6, 0), "4");
    res.assert_printed((1, 0), (7, 0), "5");
    res.assert_printed((1, 0), (8, 0), "6");
    res.assert_printed((2, 0), (9, 0), "216 10");
    res.assert_printed((2, 0), (10, 0), "423 9");
    res.assert_printed((2, 0), (11, 0), "621 8");
    res.assert_printed((2, 0), (12, 0), "810 7");
    res.assert_printed((2, 0), (13, 0), "990 6");
    res.assert_printed((6, 0), (14, 0), "22.18");
    res.assert_printed((6, 0), (15, 0), "OK");
}

#[test]
fn function() {
    let res = Headless::execute("function.bas");

    res.assert_printed((15, 0), (0, 0), "FALSE                           ");
    res.assert_printed((15, 0), (1, 0), "TRUE                            ");
    res.assert_printed((15, 0), (2, 0), "TRUE                            ");
    res.assert_printed((15, 0), (3, 0), "5                               ");
    res.assert_printed((15, 0), (4, 0), "899                             ");
    res.assert_printed((15, 0), (5, 0), "Hello, world                    ");
    res.assert_printed((15, 0), (6, 0), "OK1                             ");
    res.assert_printed((15, 0), (7, 0), "FALSE                           ");
    res.assert_printed((15, 0), (8, 0), "1                               ");
    res.assert_printed((15, 0), (9, 0), "OK2                             ");
}

#[test]
fn get_put() {
    let mut res = Headless::execute("get_put.bas");

    // If you make changes to get_put.bas use this line to update the file!
    // res.save_framebuffer("get_put.bmp");

    res.assert_bitmap("get_put.bmp");

    res.update(&[]);

    res.assert_bitmap("get_put.bmp");
}

#[test]
fn goto() {
    let mut res = Headless::execute("goto.bas");

    res.assert_printed((15, 0), (0, 0), "Hello");

    res.update(&[]);

    res.assert_printed((15, 0), (1, 0), "Hello");

    res.update(&[]);

    res.assert_printed((15, 0), (2, 0), "Hello");
}

#[test]
fn graphics() {
    let mut res = Headless::execute("graphics.bas");

    // If you make changes to graphics.bas use this line to update the file!
    // res.save_framebuffer("graphics1.bmp");

    res.assert_bitmap("graphics1.bmp");

    res.update(&[]);

    res.assert_bitmap("graphics2.bmp");
}

#[test]
fn if_then() {
    let res = Headless::execute("if_then.bas");

    res.assert_printed((15, 0), (0, 0), "AOK");
    res.assert_printed((15, 0), (1, 0), "OK");
    res.assert_printed((15, 0), (2, 0), "Great!");
    res.assert_printed((15, 0), (3, 0), "Hey");
}

#[test]
fn locate() {
    let res = Headless::execute("locate.bas");

    res.assert_printed((15, 0), (0, 0), "HELLO");
    res.assert_printed((15, 0), (0, 30), "10");
    res.assert_printed((15, 0), (7, 14), "BASIC");
    res.assert_printed((15, 0), (15, 0), "GOODBYE");
    res.assert_printed((15, 0), (15, 30), "1");
}

#[test]
fn modulus() {
    let res = Headless::execute("modulus.bas");

    res.assert_printed((15, 0), (0, 0), "0");
    res.assert_printed((15, 0), (1, 0), "10");
    res.assert_printed((15, 0), (2, 0), "1");
    res.assert_printed((15, 0), (3, 0), "0");
    res.assert_printed((15, 0), (4, 0), "10");
    res.assert_printed((15, 0), (5, 0), "1");
    res.assert_printed((15, 0), (6, 0), "0");
    res.assert_printed((15, 0), (7, 0), "10");
    res.assert_printed((15, 0), (8, 0), "1");
}

#[test]
fn peek_poke() {
    let mut res = Headless::execute("peek_poke.bas");

    res.assert_printed((15, 0), (0, 0), "0                               ");
    res.assert_printed((15, 0), (1, 0), "FALSE                           ");
    res.assert_printed((15, 0), (2, 0), "0                               ");
    res.assert_printed((15, 0), (3, 0), "0                               ");
    res.assert_printed((15, 0), (4, 0), "0                               ");
    res.assert_printed((15, 0), (5, 0), "                                ");
    res.assert_printed((15, 0), (6, 0), "OK1                             ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "1                               ");
    res.assert_printed((15, 0), (1, 0), "TRUE                            ");
    res.assert_printed((15, 0), (2, 0), "1                               ");
    res.assert_printed((15, 0), (3, 0), "0.000000000000000000000000000000");
    res.assert_printed((15, 0), (4, 0), "000000000000001                 ");
    res.assert_printed((15, 0), (5, 0), "1                               ");
    res.assert_printed((15, 0), (6, 0), "                                ");
    res.assert_printed((15, 0), (7, 0), "OK2                             ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "255                             ");
    res.assert_printed((15, 0), (1, 0), "FALSE                           ");
    res.assert_printed((15, 0), (2, 0), "255                             ");
    res.assert_printed((15, 0), (3, 0), "0.000000000000000000000000000000");
    res.assert_printed((15, 0), (4, 0), "000000000045916                 ");
    res.assert_printed((15, 0), (5, 0), "32767                           ");
    res.assert_printed((15, 0), (6, 0), "                                ");
    res.assert_printed((15, 0), (7, 0), "OK3                             ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "42                              ");
    res.assert_printed((15, 0), (1, 0), "FALSE                           ");
    res.assert_printed((15, 0), (2, 0), "42                              ");
    res.assert_printed((15, 0), (3, 0), "0.000000000000000000000000000000");
    res.assert_printed((15, 0), (4, 0), "000000000000059                 ");
    res.assert_printed((15, 0), (5, 0), "42                              ");
    res.assert_printed((15, 0), (6, 0), "*                               ");
    res.assert_printed((15, 0), (7, 0), "OK4                             ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "0                               ");
    res.assert_printed((15, 0), (1, 0), "FALSE                           ");
    res.assert_printed((15, 0), (2, 0), "0                               ");
    res.assert_printed((15, 0), (3, 0), "42                              ");
    res.assert_printed((15, 0), (4, 0), "1109917696                      ");
    res.assert_printed((15, 0), (5, 0), "                                ");
    res.assert_printed((15, 0), (6, 0), "OK5                             ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "72                              ");
    res.assert_printed((15, 0), (1, 0), "FALSE                           ");
    res.assert_printed((15, 0), (2, 0), "72                              ");
    res.assert_printed((15, 0), (3, 0), "1143139100000000000000000000    ");
    res.assert_printed((15, 0), (4, 0), "1819043144                      ");
    res.assert_printed((15, 0), (5, 0), "Hello there                     ");
    res.assert_printed((15, 0), (6, 0), "OK6                             ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "Hell   here                     ");
    res.assert_printed((15, 0), (1, 0), "OK7                             ");
}

#[test]
fn rnd() {
    let res = Headless::execute("rnd.bas");

    res.assert_printed((15, 0), (0, 0), "Roll of two dice: ");
    res.assert_printed((15, 0), (0, 20), "and ");
}

#[test]
fn select() {
    let mut res = Headless::execute("select.bas");

    res.assert_printed((15, 0), (0, 0), "0 is 0                          ");
    res.assert_printed((15, 0), (1, 0), "1 is 1                          ");
    res.assert_printed((15, 0), (2, 0), "2 is 2 or 3                     ");
    res.assert_printed((15, 0), (3, 0), "3 is 2 or 3                     ");
    res.assert_printed((15, 0), (4, 0), "4 is 4 to 5                     ");
    res.assert_printed((15, 0), (5, 0), "5 is 4 to 5                     ");
    res.assert_printed((15, 0), (6, 0), "6 is < 7                        ");
    res.assert_printed((15, 0), (7, 0), "7 is <= 8                       ");
    res.assert_printed((15, 0), (8, 0), "8 is <= 8                       ");
    res.assert_printed((15, 0), (9, 0), "9 is <> 10                      ");
    res.assert_printed((15, 0), (10, 0), "10 is 10                       ");
    res.assert_printed((15, 0), (11, 0), "11 is <> 10                    ");
    res.assert_printed((15, 0), (12, 0), "12 is >= 12                    ");
    res.assert_printed((15, 0), (13, 0), "13 is >= 12                    ");
    res.assert_printed((15, 0), (14, 0), "14 is > 13                     ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "foo-bar                         ");
    res.assert_printed((15, 0), (1, 0), "baz-buz                         ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "9                               ");
    res.assert_printed((15, 0), (1, 0), "X                               ");
    res.assert_printed((15, 0), (2, 0), "11                              ");
    res.assert_printed((15, 0), (3, 0), "12                              ");
    res.assert_printed((15, 0), (4, 0), "13                              ");
    res.assert_printed((15, 0), (5, 0), "X                               ");
    res.assert_printed((15, 0), (6, 0), "15                              ");
    res.assert_printed((15, 0), (7, 0), "B                               ");
    res.assert_printed((15, 0), (8, 0), "B                               ");

    res.update(&[]);

    res.assert_printed((15, 0), (0, 0), "X                               ");
    res.assert_printed((15, 0), (1, 0), "Y                               ");
    res.assert_printed((15, 0), (2, 0), "Y                               ");
    res.assert_printed((15, 0), (3, 0), "Y                               ");
    res.assert_printed((15, 0), (4, 0), "X                               ");
}

#[test]
fn sub() {
    let res = Headless::execute("sub.bas");

    res.assert_printed((15, 0), (0, 0), "A                               ");
    res.assert_printed((15, 0), (1, 0), "A                               ");
    res.assert_printed((15, 0), (2, 0), "B                               ");
    res.assert_printed((15, 0), (3, 0), "C                               ");
    res.assert_printed((15, 0), (4, 0), "D                               ");
    res.assert_printed((15, 0), (5, 0), "< >                             ");
    res.assert_printed((15, 0), (6, 0), "OK                              ");
}

#[test]
fn while_wend() {
    let res = Headless::execute("while_wend.bas");

    res.assert_printed((15, 0), (0, 0), "1");
    res.assert_printed((15, 0), (1, 0), "2");
    res.assert_printed((15, 0), (2, 0), "1");
    res.assert_printed((15, 0), (3, 0), "0");
}

/// Helper to pick a queue family for submitting device commands.
fn device_queue_family_index(device: &Device, flags: vk::QueueFlags) -> Option<usize> {
    device
        .physical_device
        .queue_families
        .iter()
        .enumerate()
        .find(|(_, properties)| properties.queue_flags.contains(flags))
        .map(|(index, _)| index)
}

struct Headless {
    device: Arc<Device>,
    framebuffer: [u8; Self::FRAMEBUFFER_LEN],
    interpreter: Interpreter,
    charset: [[[bool; 5]; 6]; 96],
    palette: Vec<u8>,
}

impl Headless {
    const FRAMEBUFFER_LEN: usize =
        4 * (Interpreter::FRAMEBUFFER_WIDTH * Interpreter::FRAMEBUFFER_HEIGHT) as usize;

    fn execute(program: &str) -> Self {
        let program = Instruction::compile(&read(PROGRAM_DIR.join(program)).unwrap()).unwrap();

        let device = Arc::new(Device::create_headless(DeviceInfo::new()).unwrap());
        let interpreter = Interpreter::new(&device, program).unwrap();

        let mut res = Self {
            charset: ascii_5x6(),
            device,
            framebuffer: [0; Self::FRAMEBUFFER_LEN],
            interpreter,
            palette: vga_256(),
        };

        res.update(&[]);

        res
    }

    fn assert_bitmap(&self, path: impl AsRef<Path>) {
        use bmp::open;

        let image = open(BITMAP_DIR.join(path)).unwrap();

        for y in 0..Interpreter::FRAMEBUFFER_HEIGHT {
            for x in 0..Interpreter::FRAMEBUFFER_WIDTH {
                let expected_pixel = image.get_pixel(x, y);
                let (actual_r, actual_g, actual_b, _) = self.pixel(x, y);

                assert_eq!(expected_pixel.r, actual_r);
                assert_eq!(expected_pixel.g, actual_g);
                assert_eq!(expected_pixel.b, actual_b);
            }
        }
    }

    fn assert_printed(&self, color: (u8, u8), location: (u32, u32), s: &str) {
        let (foreground, background) = color;
        let foreground = self.color(foreground);
        let background = self.color(background);

        let (row, col) = location;
        let (x, y) = (col * 5, row * 6);

        for (char_index, char) in s.as_bytes().iter().copied().enumerate() {
            assert!(char >= 32);
            assert!(char <= 127);

            for char_y in 0..6u32 {
                for char_x in 0..5u32 {
                    let char = char - 32;
                    let (expected_red, expected_green, expected_blue) =
                        if self.charset[char as usize][char_y as usize][char_x as usize] {
                            foreground
                        } else {
                            background
                        };

                    let pixel_x = x + char_x + char_index as u32 * 5;
                    let pixel_y = y + char_y;
                    let (actual_red, actual_green, actual_blue, actual_alpha) =
                        self.pixel(pixel_x, pixel_y);

                    let msg = format!(
                        "character {char_index} `{}` at row={row} col={col}, x={pixel_x}, y={pixel_y}",
                        (char + 32) as char,
                    );

                    assert_eq!(expected_red, actual_red, "red {}", msg);
                    assert_eq!(expected_green, actual_green, "green {}", msg);
                    assert_eq!(expected_blue, actual_blue, "blue {}", msg);

                    // The framebuffer alpha should always be fully opaque
                    assert_eq!(actual_alpha, 0xFF);
                }
            }
        }
    }

    fn color(&self, index: u8) -> (u8, u8, u8) {
        let start = index * 4;
        let end = start + 3;
        let data = &self.palette[start as usize..end as usize];

        (data[0], data[1], data[2])
    }

    fn pixel(&self, x: u32, y: u32) -> (u8, u8, u8, u8) {
        let start = y * 4 * Interpreter::FRAMEBUFFER_WIDTH + x * 4;
        let end = start + 4;
        let data = &self.framebuffer[start as usize..end as usize];

        (data[0], data[1], data[2], data[3])
    }

    #[allow(unused)]
    fn save_framebuffer(&self, path: impl AsRef<Path>) {
        use bmp::{Image, Pixel};

        let mut image = Image::new(
            Interpreter::FRAMEBUFFER_WIDTH,
            Interpreter::FRAMEBUFFER_HEIGHT,
        );

        for y in 0..Interpreter::FRAMEBUFFER_HEIGHT {
            for x in 0..Interpreter::FRAMEBUFFER_WIDTH {
                let (r, g, b, _) = self.pixel(x, y);
                image.set_pixel(x, y, Pixel::new(r, g, b));
            }
        }

        image.save(BITMAP_DIR.join(path)).unwrap();
    }

    fn update(&mut self, events: &[Event<()>]) {
        let mut render_graph = RenderGraph::new();
        self.interpreter.update(&mut render_graph, events).unwrap();

        let framebuffer_image = render_graph.bind_node(self.interpreter.framebuffer_image());
        let framebuffer = {
            let temp_buf = render_graph.bind_node(
                Buffer::create(
                    &self.device,
                    BufferInfo::new_mappable(
                        (4 * Interpreter::FRAMEBUFFER_WIDTH * Interpreter::FRAMEBUFFER_HEIGHT) as _,
                        vk::BufferUsageFlags::TRANSFER_DST,
                    ),
                )
                .unwrap(),
            );
            render_graph
                .copy_image_to_buffer(framebuffer_image, temp_buf)
                .unbind_node(temp_buf)
        };

        let queue_family_index = device_queue_family_index(
            &self.device,
            vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER,
        )
        .unwrap();

        render_graph
            .resolve()
            .submit(&mut HashPool::new(&self.device), queue_family_index, 0)
            .unwrap()
            .wait_until_executed()
            .unwrap();
        self.framebuffer
            .copy_from_slice(Buffer::mapped_slice(&framebuffer));
    }
}
