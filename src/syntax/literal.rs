use {
    super::{
        super::token::Tokens, bool_lit, bool_ty, debug_location, f32_lit, f32_ty, i32_lit, i32_ty,
        str_lit, str_ty, u8_ty, Span,
    },
    nom::{
        branch::alt,
        combinator::{map, opt, verify},
        sequence::terminated,
        IResult,
    },
    std::fmt::{Debug, Formatter, Result as FmtResult},
};

#[derive(Clone, Copy, PartialEq)]
pub enum Literal<'a> {
    Boolean(bool, Span<'a>),
    Byte(u8, Span<'a>),
    Float(f32, Span<'a>),
    Integer(i32, Span<'a>),
    String(&'a str, Span<'a>),
}

impl<'a> Literal<'a> {
    pub fn location(self) -> Span<'a> {
        match self {
            Self::Boolean(_, res)
            | Self::Byte(_, res)
            | Self::Float(_, res)
            | Self::Integer(_, res)
            | Self::String(_, res) => res,
        }
    }

    pub(super) fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        alt((
            Self::parse_bool,
            Self::parse_u8,
            Self::parse_f32,
            Self::parse_i32,
            Self::parse_str,
        ))(tokens)
    }

    fn parse_bool(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(bool_lit, opt(bool_ty)), |token| {
            Self::Boolean(token.boolean_literal().unwrap(), tokens.location())
        })(tokens)
    }

    fn parse_f32(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(f32_lit, opt(f32_ty)), |token| {
            Self::Float(token.float_literal().unwrap(), tokens.location())
        })(tokens)
    }

    fn parse_i32(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(i32_lit, opt(i32_ty)), |token| {
            Self::Integer(token.integer_literal().unwrap(), tokens.location())
        })(tokens)
    }

    fn parse_str(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(str_lit, opt(str_ty)), |token| {
            Self::String(token.string_literal().unwrap(), tokens.location())
        })(tokens)
    }

    fn parse_u8(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            terminated(
                verify(i32_lit, |token| {
                    let val = token.integer_literal().unwrap();
                    (0..=0xFF).contains(&val)
                }),
                u8_ty,
            ),
            |token| Self::Byte(token.integer_literal().unwrap() as _, tokens.location()),
        )(tokens)
    }
}

impl<'a> Debug for Literal<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("Literal ")?;

        match self {
            Self::Boolean(val, _) => f.write_fmt(format_args!("`{val}` Boolean ")),
            Self::Byte(val, _) => f.write_fmt(format_args!("`{val}` Byte ")),
            Self::Float(val, _) => f.write_fmt(format_args!("`{val}` Float ")),
            Self::Integer(val, _) => f.write_fmt(format_args!("`{val}` Integer ")),
            Self::String(val, _) => f.write_fmt(format_args!("`{val}` String ")),
        }?;

        debug_location(f, self.location())
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::token::{Span, Token},
    };

    #[test]
    fn bool_typed() {
        let span = Span::new(&[]);
        let tokens = [Token::BooleanLiteral(true, span), Token::BooleanType(span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::Boolean(true, span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn bool_untyped() {
        let span = Span::new(&[]);
        let tokens = [Token::BooleanLiteral(true, span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::Boolean(true, span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn u8_typed() {
        let span = Span::new(&[]);
        let tokens = [Token::IntegerLiteral(0xFF, span), Token::ByteType(span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::Byte(0xFF, span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn f32_typed() {
        let span = Span::new(&[]);
        let tokens = [Token::FloatLiteral(42.0, span), Token::FloatType(span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::Float(42.0, span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn f32_untyped() {
        let span = Span::new(&[]);
        let tokens = [Token::FloatLiteral(42.0, span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::Float(42.0, span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }
    #[test]
    fn i32_typed() {
        let span = Span::new(&[]);
        let tokens = [Token::IntegerLiteral(42, span), Token::IntegerType(span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::Integer(42, span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn i32_untyped() {
        let span = Span::new(&[]);
        let tokens = [Token::IntegerLiteral(42, span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::Integer(42, span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }
    #[test]
    fn str_typed() {
        let span = Span::new(&[]);
        let tokens = [Token::StringLiteral("hello", span), Token::StringType(span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::String("hello", span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn str_untyped() {
        let span = Span::new(&[]);
        let tokens = [Token::StringLiteral("hello", span)];
        let tokens = Tokens::new(&tokens);

        let expected = Literal::String("hello", span);

        let (_, result) = Literal::parse(tokens).unwrap();

        assert_eq!(expected, result);
    }
}
