use super::token::Span;

pub fn span<'a>(offset: usize, line: u32, bytes: &'a [u8]) -> Span<'a> {
    assert!(offset < bytes.len());

    unsafe {
        // Safety: We checked the offset above
        Span::new_from_raw_offset(offset, line, &bytes[offset..], ())
    }
}
