mod input;

pub(super) use self::input::Tokens;

use {
    nom::{
        branch::alt,
        bytes::complete::{escaped, tag, tag_no_case, take, take_until},
        character::complete::{
            alpha1, alphanumeric1, char, digit1, hex_digit1, line_ending, multispace0, one_of,
            satisfy, space0,
        },
        combinator::{map, map_res, not, opt, recognize, rest, value},
        multi::{many0, many1},
        number::complete::float,
        sequence::{delimited, pair, preceded, terminated},
        IResult, InputLength,
    },
    nom_locate::LocatedSpan,
    std::{
        fmt::{Debug, Formatter, Result as FmtResult},
        str::from_utf8_unchecked,
        str::FromStr,
    },
};

pub type Span<'a> = LocatedSpan<&'a [u8]>;

pub fn debug_location(f: &mut Formatter<'_>, location: Span) -> FmtResult {
    f.write_fmt(format_args!(
        "@ {}:{}",
        location.location_offset(),
        location.location_line()
    ))
}

pub fn location_string(location: Span) -> String {
    format!(
        "Line {}, column {}",
        location.location_line(),
        location.get_column(),
    )
}

#[derive(Debug)]
pub struct AsciiError;

macro_rules! token {
    ($func_name: ident, $tag_string: literal, $output_token: expr) => {
        fn $func_name(s: Span<'_>) -> IResult<Span<'_>, Token> {
            map(tag($tag_string), |_| $output_token(s))(s)
        }
    };
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Token<'a> {
    Illegal(Span<'a>),
    Comment(Span<'a>),
    Identifer(&'a str, Span<'a>),
    // Literals
    BooleanLiteral(bool, Span<'a>),
    FloatLiteral(f32, Span<'a>),
    IntegerLiteral(i32, Span<'a>),
    StringLiteral(&'a str, Span<'a>),
    // Operators
    Add(Span<'a>),
    Subtract(Span<'a>),
    Multiply(Span<'a>),
    Divide(Span<'a>),
    Equal(Span<'a>),
    NotEqual(Span<'a>),
    GreaterThanEqual(Span<'a>),
    LessThanEqual(Span<'a>),
    GreaterThan(Span<'a>),
    LessThan(Span<'a>),
    Not(Span<'a>),
    And(Span<'a>),
    Or(Span<'a>),
    Xor(Span<'a>),
    Pset(Span<'a>),
    Preset(Span<'a>),
    Tset(Span<'a>),
    // Reserved Words
    ConvertBoolean(Span<'a>),
    ConvertByte(Span<'a>),
    ConvertFloat(Span<'a>),
    ConvertInteger(Span<'a>),
    ConvertString(Span<'a>),
    Abs(Span<'a>),
    Sin(Span<'a>),
    Call(Span<'a>),
    Cos(Span<'a>),
    ClearScreen(Span<'a>),
    Color(Span<'a>),
    Dimension(Span<'a>),
    Do(Span<'a>),
    Else(Span<'a>),
    End(Span<'a>),
    For(Span<'a>),
    Function(Span<'a>),
    Get(Span<'a>),
    Goto(Span<'a>),
    If(Span<'a>),
    KeyDown(Span<'a>),
    Line(Span<'a>),
    Locate(Span<'a>),
    Loop(Span<'a>),
    Next(Span<'a>),
    Palette(Span<'a>),
    Peek(Span<'a>),
    Poke(Span<'a>),
    Print(Span<'a>),
    Put(Span<'a>),
    Rectangle(Span<'a>),
    Return(Span<'a>),
    Rnd(Span<'a>),
    Step(Span<'a>),
    Sub(Span<'a>),
    Then(Span<'a>),
    Timer(Span<'a>),
    To(Span<'a>),
    Until(Span<'a>),
    Wend(Span<'a>),
    While(Span<'a>),
    Yield(Span<'a>),
    // Punctuation
    Colon(Span<'a>),
    Comma(Span<'a>),
    Continuation(Span<'a>),
    EndOfLine(Span<'a>),
    LeftParenthesis(Span<'a>),
    RightParenthesis(Span<'a>),
    LeftCurlyBracket(Span<'a>),
    RightCurlyBracket(Span<'a>),
    LeftSquareBracket(Span<'a>),
    RightSquareBracket(Span<'a>),
    Semicolon(Span<'a>),
    // Type Specifiers
    BooleanType(Span<'a>),
    ByteType(Span<'a>),
    FloatType(Span<'a>),
    IntegerType(Span<'a>),
    StringType(Span<'a>),
}

impl<'a> Token<'a> {
    pub fn ascii_str(v: &'a [u8]) -> Result<&'a str, AsciiError> {
        for &char in v {
            if !Self::valid_ascii_subset(char.into()) {
                return Err(AsciiError);
            }
        }

        Ok(unsafe {
            // SAFETY: we checked that the bytes `v` are valid UTF-8.
            from_utf8_unchecked(v)
        })
    }

    pub fn boolean_literal(self) -> Option<bool> {
        match self {
            Self::BooleanLiteral(val, _) => Some(val),
            _ => None,
        }
    }

    pub fn float_literal(self) -> Option<f32> {
        match self {
            Self::FloatLiteral(val, _) => Some(val),
            _ => None,
        }
    }

    pub fn integer_literal(self) -> Option<i32> {
        match self {
            Self::IntegerLiteral(val, _) => Some(val),
            _ => None,
        }
    }

    pub fn identifier(self) -> Option<&'a str> {
        match self {
            Self::Identifer(val, _) => Some(val),
            _ => None,
        }
    }

    pub fn lex(bytes: &'a [u8]) -> IResult<Span<'a>, Vec<Self>> {
        let (tokens, mut result) = delimited(
            multispace0,
            many0(delimited(
                space0,
                alt((
                    Self::lex_comment,
                    Self::lex_integer,
                    Self::lex_ty,
                    Self::lex_op,
                    Self::lex_punctuation,
                    Self::lex_str,
                    Self::lex_float,
                    Self::lex_hex,
                    Self::lex_word,
                    Self::lex_illegal,
                )),
                space0,
            )),
            multispace0,
        )(Span::new(bytes))?;

        assert!(tokens.is_empty());

        // Remove comments
        result.retain(|token| !matches!(token, Self::Comment(_)));

        Ok((tokens, result))
    }

    fn lex_comment(input: Span<'a>) -> IResult<Span<'a>, Self> {
        value(
            Self::Comment(input),
            preceded(
                multispace0,
                pair(
                    alt((map(char('\''), |_| ()), map(tag("REM "), |_| ()))),
                    alt((take_until("\n"), rest)),
                ),
            ),
        )(input)
    }

    fn lex_float(input: Span<'a>) -> IResult<Span<'a>, Self> {
        map(float, |f| Self::FloatLiteral(f, input))(input)
    }

    fn lex_hex(input: Span<'a>) -> IResult<Span<'a>, Self> {
        map(
            map_res(
                preceded(
                    tag_no_case("&h"),
                    recognize(many1(terminated(hex_digit1, opt(char('_'))))),
                ),
                |input: Span<'a>| {
                    // TODO: Remove the underscores!
                    i32::from_str_radix(Self::ascii_str(input.fragment()).unwrap(), 16)
                },
            ),
            |int| Self::IntegerLiteral(int, input),
        )(input)
    }

    fn lex_illegal(input: Span<'a>) -> IResult<Span<'a>, Self> {
        map(take(1usize), |_| Self::Illegal(input))(input)
    }

    fn lex_integer(input: Span<'a>) -> IResult<Span<'a>, Self> {
        token!(sub, "-", Token::Subtract);

        map(
            terminated(
                pair::<_, _, i32, _, _, _>(
                    map(opt(value(-1i32, sub)), |mag| mag.unwrap_or(1)),
                    map_res(
                        map_res(digit1, |num: Span<'a>| Self::ascii_str(num.fragment())),
                        FromStr::from_str,
                    ),
                ),
                not(one_of(".eE")),
            ),
            |(mag, num)| Self::IntegerLiteral(mag * num, input),
        )(input)
    }

    fn lex_op(input: Span<'a>) -> IResult<Span<'a>, Self> {
        token!(add, "+", Token::Add);
        token!(sub, "-", Token::Subtract);
        token!(mul, "*", Token::Multiply);
        token!(div, "/", Token::Divide);
        token!(eq, "=", Token::Equal);
        token!(ne, "<>", Token::NotEqual);
        token!(gte, ">=", Token::GreaterThanEqual);
        token!(gt, ">", Token::GreaterThan);
        token!(lte, "<=", Token::LessThanEqual);
        token!(lt, "<", Token::LessThan);
        token!(not, "NOT", Token::Not);
        token!(and, "AND", Token::And);
        token!(or, "OR", Token::Or);
        token!(xor, "XOR", Token::Xor);
        token!(pset, "PSET", Token::Pset);
        token!(preset, "PRESET", Token::Preset);
        token!(tset, "TSET", Token::Tset);

        alt((
            add, sub, mul, div, eq, ne, gte, gt, lte, lt, not, and, or, xor, pset, preset, tset,
        ))(input)
    }

    fn lex_punctuation(input: Span<'a>) -> IResult<Span<'a>, Self> {
        token!(colon, ":", Token::Colon);
        token!(comma, ",", Token::Comma);
        token!(continuation, "_", Token::Continuation);
        token!(l_curly_bracket, "{", Token::LeftCurlyBracket);
        token!(l_paren, "(", Token::LeftParenthesis);
        token!(l_sq_bracket, "[", Token::LeftSquareBracket);
        token!(r_curly_bracket, "}", Token::RightCurlyBracket);
        token!(r_paren, ")", Token::RightParenthesis);
        token!(r_sq_bracket, "]", Token::RightSquareBracket);
        token!(semicolon, ";", Token::Semicolon);

        alt((
            colon,
            comma,
            continuation,
            l_curly_bracket,
            l_paren,
            l_sq_bracket,
            r_curly_bracket,
            r_paren,
            r_sq_bracket,
            semicolon,
            map(many1(line_ending), |_| Self::EndOfLine(input)),
        ))(input)
    }

    fn lex_str(input: Span<'a>) -> IResult<Span<'a>, Self> {
        fn escape(s: Span) -> IResult<Span, Span> {
            escaped(
                satisfy(|c| c != '\\' && c != '\"' && Token::valid_ascii_subset(c)),
                '\\',
                one_of("\\\""),
            )(s)
        }

        map(
            delimited(
                tag("\""),
                map_res(escape, |s| Self::ascii_str(s.fragment())),
                tag("\""),
            ),
            |str| Self::StringLiteral(str, input),
        )(input)
    }

    fn lex_ty(input: Span<'a>) -> IResult<Span<'a>, Self> {
        token!(boolean, "?", Token::BooleanType);
        token!(byte, "@", Token::ByteType);
        token!(integer, "%", Token::IntegerType);
        token!(float, "!", Token::FloatType);
        token!(string, "$", Token::StringType);

        alt((boolean, byte, integer, float, string))(input)
    }

    fn lex_word(input: Span<'a>) -> IResult<Span<'a>, Self> {
        map_res(
            recognize(pair(
                alt((alpha1::<Span<'a>, _>, tag("_"))),
                many0(alt((alphanumeric1, tag("_")))),
            )),
            |s| {
                Self::ascii_str(s.fragment()).map(|str| match str.to_ascii_uppercase().as_str() {
                    "BOOLEAN" => Self::ConvertBoolean(input),
                    "BYTE" => Self::ConvertByte(input),
                    "FLOAT" => Self::ConvertFloat(input),
                    "INT" => Self::ConvertInteger(input),
                    "STR" => Self::ConvertString(input),
                    "ABS" => Self::Abs(input),
                    "SIN" => Self::Sin(input),
                    "CALL" => Self::Call(input),
                    "COS" => Self::Cos(input),
                    "CLS" => Self::ClearScreen(input),
                    "COLOR" => Self::Color(input),
                    "DIM" => Self::Dimension(input),
                    "DO" => Self::Do(input),
                    "ELSE" => Self::Else(input),
                    "END" => Self::End(input),
                    "FOR" => Self::For(input),
                    "FUNCTION" => Self::Function(input),
                    "GET" => Self::Get(input),
                    "GOTO" => Self::Goto(input),
                    "IF" => Self::If(input),
                    "KEYDOWN" => Self::KeyDown(input),
                    "LINE" => Self::Line(input),
                    "LOCATE" => Self::Locate(input),
                    "LOOP" => Self::Loop(input),
                    "NEXT" => Self::Next(input),
                    "PALETTE" => Self::Palette(input),
                    "PEEK" => Self::Peek(input),
                    "POKE" => Self::Poke(input),
                    "PRINT" => Self::Print(input),
                    "PUT" => Self::Put(input),
                    "RECTANGLE" => Self::Rectangle(input),
                    "RETURN" => Self::Return(input),
                    "RND" => Self::Rnd(input),
                    "STEP" => Self::Step(input),
                    "SUB" => Self::Sub(input),
                    "THEN" => Self::Then(input),
                    "TIMER" => Self::Timer(input),
                    "TO" => Self::To(input),
                    "UNTIL" => Self::Until(input),
                    "WEND" => Self::Wend(input),
                    "WHILE" => Self::While(input),
                    "YIELD" => Self::Yield(input),
                    "TRUE" => Self::BooleanLiteral(true, input),
                    "FALSE" => Self::BooleanLiteral(false, input),
                    _ => Self::Identifer(str, input),
                })
            },
        )(input)
    }

    pub fn location(&self) -> Span<'a> {
        match self {
            Self::Illegal(s)
            | Self::Comment(s)
            | Self::Identifer(_, s)
            | Self::BooleanLiteral(_, s)
            | Self::FloatLiteral(_, s)
            | Self::IntegerLiteral(_, s)
            | Self::StringLiteral(_, s)
            | Self::Add(s)
            | Self::Subtract(s)
            | Self::Multiply(s)
            | Self::Divide(s)
            | Self::Equal(s)
            | Self::NotEqual(s)
            | Self::GreaterThanEqual(s)
            | Self::LessThanEqual(s)
            | Self::GreaterThan(s)
            | Self::LessThan(s)
            | Self::Not(s)
            | Self::And(s)
            | Self::Or(s)
            | Self::Xor(s)
            | Self::Pset(s)
            | Self::Preset(s)
            | Self::Tset(s)
            | Self::ConvertBoolean(s)
            | Self::ConvertByte(s)
            | Self::ConvertFloat(s)
            | Self::ConvertInteger(s)
            | Self::ConvertString(s)
            | Self::Abs(s)
            | Self::Sin(s)
            | Self::Call(s)
            | Self::Cos(s)
            | Self::ClearScreen(s)
            | Self::Color(s)
            | Self::Dimension(s)
            | Self::Do(s)
            | Self::Else(s)
            | Self::End(s)
            | Self::For(s)
            | Self::Function(s)
            | Self::Get(s)
            | Self::Goto(s)
            | Self::If(s)
            | Self::KeyDown(s)
            | Self::Line(s)
            | Self::Locate(s)
            | Self::Loop(s)
            | Self::Next(s)
            | Self::Palette(s)
            | Self::Peek(s)
            | Self::Poke(s)
            | Self::Print(s)
            | Self::Put(s)
            | Self::Rectangle(s)
            | Self::Return(s)
            | Self::Rnd(s)
            | Self::Step(s)
            | Self::Sub(s)
            | Self::Then(s)
            | Self::Timer(s)
            | Self::To(s)
            | Self::Until(s)
            | Self::Wend(s)
            | Self::While(s)
            | Self::Yield(s)
            | Self::Colon(s)
            | Self::Comma(s)
            | Self::Continuation(s)
            | Self::EndOfLine(s)
            | Self::LeftParenthesis(s)
            | Self::RightParenthesis(s)
            | Self::LeftCurlyBracket(s)
            | Self::RightCurlyBracket(s)
            | Self::LeftSquareBracket(s)
            | Self::RightSquareBracket(s)
            | Self::Semicolon(s)
            | Self::BooleanType(s)
            | Self::ByteType(s)
            | Self::IntegerType(s)
            | Self::FloatType(s)
            | Self::StringType(s) => *s,
        }
    }

    pub fn string_literal(self) -> Option<&'a str> {
        match self {
            Self::StringLiteral(val, _) => Some(val),
            _ => None,
        }
    }

    /// This is the subset of ascii JW-Basic supports
    fn valid_ascii_subset(c: char) -> bool {
        // 32 -> 127 inclusive
        matches!(c, '\x20'..='\x7F')
    }
}

impl<'a> Debug for Token<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Illegal(..) => f.write_str("Illegal"),
            Self::Comment(..) => f.write_str("Comment"),
            Self::Identifer(id, _) => f.write_fmt(format_args!("Identifer `{id}`")),
            Self::BooleanLiteral(bool, ..) => f.write_fmt(format_args!("BooleanLiteral `{bool}`")),
            Self::FloatLiteral(f32, ..) => f.write_fmt(format_args!("FloatLiteral `{f32}`")),
            Self::IntegerLiteral(i32, ..) => f.write_fmt(format_args!("IntegerLiteral `{i32}`")),
            Self::StringLiteral(str, ..) => f.write_fmt(format_args!("StringLiteral `{str}`")),
            Self::Add(..) => f.write_str("Add"),
            Self::Subtract(..) => f.write_str("Subtract"),
            Self::Multiply(..) => f.write_str("Multiply"),
            Self::Divide(..) => f.write_str("Divide"),
            Self::Equal(..) => f.write_str("Equal"),
            Self::NotEqual(..) => f.write_str("NotEqual"),
            Self::GreaterThanEqual(..) => f.write_str("GreaterThanEqual"),
            Self::LessThanEqual(..) => f.write_str("LessThanEqual"),
            Self::GreaterThan(..) => f.write_str("GreaterThan"),
            Self::LessThan(..) => f.write_str("LessThan"),
            Self::Not(..) => f.write_str("Not"),
            Self::And(..) => f.write_str("And"),
            Self::Or(..) => f.write_str("Or"),
            Self::Xor(..) => f.write_str("Xor"),
            Self::Pset(..) => f.write_str("Pset"),
            Self::Preset(..) => f.write_str("Preset"),
            Self::Tset(..) => f.write_str("Tset"),
            Self::ConvertBoolean(..) => f.write_str("ConvertBoolean"),
            Self::ConvertByte(..) => f.write_str("ConvertByte"),
            Self::ConvertFloat(..) => f.write_str("ConvertFloat"),
            Self::ConvertInteger(..) => f.write_str("ConvertInteger"),
            Self::ConvertString(..) => f.write_str("ConvertString"),
            Self::Abs(..) => f.write_str("Abs"),
            Self::Sin(..) => f.write_str("Sin"),
            Self::Call(..) => f.write_str("Call"),
            Self::Cos(..) => f.write_str("Cos"),
            Self::ClearScreen(..) => f.write_str("ClearScreen"),
            Self::Color(..) => f.write_str("Color"),
            Self::Dimension(..) => f.write_str("Dimension"),
            Self::Do(..) => f.write_str("Do"),
            Self::End(..) => f.write_str("End"),
            Self::Else(..) => f.write_str("Else"),
            Self::For(..) => f.write_str("For"),
            Self::Function(..) => f.write_str("Function"),
            Self::Get(..) => f.write_str("Get"),
            Self::Goto(..) => f.write_str("Goto"),
            Self::If(..) => f.write_str("If"),
            Self::KeyDown(..) => f.write_str("KeyDown"),
            Self::Line(..) => f.write_str("Line"),
            Self::Locate(..) => f.write_str("Locate"),
            Self::Loop(..) => f.write_str("Loop"),
            Self::Next(..) => f.write_str("Next"),
            Self::Palette(..) => f.write_str("Pallete"),
            Self::Peek(..) => f.write_str("Peek"),
            Self::Poke(..) => f.write_str("Poke"),
            Self::Print(..) => f.write_str("Print"),
            Self::Put(..) => f.write_str("Put"),
            Self::Rectangle(..) => f.write_str("Rectangle"),
            Self::Return(..) => f.write_str("Return"),
            Self::Rnd(..) => f.write_str("Return"),
            Self::Step(..) => f.write_str("Step"),
            Self::Sub(..) => f.write_str("Sub"),
            Self::Then(..) => f.write_str("Then"),
            Self::Timer(..) => f.write_str("Timer"),
            Self::To(..) => f.write_str("To"),
            Self::Until(..) => f.write_str("Until"),
            Self::Wend(..) => f.write_str("Wend"),
            Self::While(..) => f.write_str("While"),
            Self::Yield(..) => f.write_str("Yield"),
            Self::Colon(..) => f.write_str("Colon"),
            Self::Comma(..) => f.write_str("Comma"),
            Self::Continuation(..) => f.write_str("Continuation"),
            Self::EndOfLine(..) => f.write_str("EndOfLine"),
            Self::LeftParenthesis(..) => f.write_str("LeftParenthesis"),
            Self::RightParenthesis(..) => f.write_str("RightParenthesis"),
            Self::LeftCurlyBracket(..) => f.write_str("LeftCurlyBracket"),
            Self::RightCurlyBracket(..) => f.write_str("RightCurlyBracket"),
            Self::LeftSquareBracket(..) => f.write_str("LeftSquareBracket"),
            Self::RightSquareBracket(..) => f.write_str("RightSquareBracket"),
            Self::Semicolon(..) => f.write_str("Semicolon"),
            Self::BooleanType(..) => f.write_str("BooleanType"),
            Self::ByteType(..) => f.write_str("ByteType"),
            Self::FloatType(..) => f.write_str("FloatType"),
            Self::IntegerType(..) => f.write_str("IntegerType"),
            Self::StringType(..) => f.write_str("StringType"),
        }?;

        f.write_str(" ")?;
        debug_location(f, self.location())
    }
}

impl<'a> InputLength for Token<'a> {
    #[inline]
    fn input_len(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use {super::*, crate::tests::span};

    #[test]
    fn basic_tokens() {
        let input = b"=\n\t=\n=";

        let expected = vec![
            Token::Equal(span(0, 1, input)),
            Token::EndOfLine(span(1, 1, input)),
            Token::Equal(span(3, 2, input)),
            Token::EndOfLine(span(4, 2, input)),
            Token::Equal(span(5, 3, input)),
        ];

        let (_, result) = Token::lex(input).unwrap();

        assert_eq!(expected, result);

        assert_eq!(result[2].location().get_line_beginning(), b"\t=");
        assert_eq!(result[2].location().get_column(), 2);
        assert_eq!(result[2].location().location_line(), 2);

        assert_eq!(result[4].location().get_line_beginning(), b"=");
        assert_eq!(result[4].location().get_column(), 1);
        assert_eq!(result[4].location().location_line(), 3);
    }

    #[test]
    fn keyword_tokens() {
        let input = b"DIM five = 5\n\
        DIM ten = 10\n\
        DIM add = FUNCTION(x, y) {\n\
            x + y\n\
        }\n\
        DIM result = add(five, ten)";

        let expected = vec![
            Token::Dimension(span(0, 1, input)),
            Token::Identifer("five", span(4, 1, input)),
            Token::Equal(span(9, 1, input)),
            Token::IntegerLiteral(5, span(11, 1, input)),
            Token::EndOfLine(span(12, 1, input)),
            Token::Dimension(span(13, 2, input)),
            Token::Identifer("ten", span(17, 2, input)),
            Token::Equal(span(21, 2, input)),
            Token::IntegerLiteral(10, span(23, 2, input)),
            Token::EndOfLine(span(25, 2, input)),
            Token::Dimension(span(26, 3, input)),
            Token::Identifer("add", span(30, 3, input)),
            Token::Equal(span(34, 3, input)),
            Token::Function(span(36, 3, input)),
            Token::LeftParenthesis(span(44, 3, input)),
            Token::Identifer("x", span(45, 3, input)),
            Token::Comma(span(46, 3, input)),
            Token::Identifer("y", span(48, 3, input)),
            Token::RightParenthesis(span(49, 3, input)),
            Token::LeftCurlyBracket(span(51, 3, input)),
            Token::EndOfLine(span(52, 3, input)),
            Token::Identifer("x", span(53, 4, input)),
            Token::Add(span(55, 4, input)),
            Token::Identifer("y", span(57, 4, input)),
            Token::EndOfLine(span(58, 4, input)),
            Token::RightCurlyBracket(span(59, 5, input)),
            Token::EndOfLine(span(60, 5, input)),
            Token::Dimension(span(61, 6, input)),
            Token::Identifer("result", span(65, 6, input)),
            Token::Equal(span(72, 6, input)),
            Token::Identifer("add", span(74, 6, input)),
            Token::LeftParenthesis(span(77, 6, input)),
            Token::Identifer("five", span(78, 6, input)),
            Token::Comma(span(82, 6, input)),
            Token::Identifer("ten", span(84, 6, input)),
            Token::RightParenthesis(span(87, 6, input)),
        ];

        let (_, result) = Token::lex(input).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn escaped_strings() {
        let input = br#"DIM a = "te\"st""#;

        let expected = vec![
            Token::Dimension(span(0, 1, input)),
            Token::Identifer("a", span(4, 1, input)),
            Token::Equal(span(6, 1, input)),
            Token::StringLiteral(r#"te\"st"#, span(8, 1, input)),
        ];

        let (_, result) = Token::lex(input).unwrap();

        assert_eq!(expected, result);

        assert_eq!(
            expected[3].location().get_line_beginning(),
            b"DIM a = \"te\\\"st\""
        );
        assert_eq!(expected[3].location().get_column(), 9);
        assert_eq!(expected[3].location().location_line(), 1);
    }

    #[test]
    fn string_literals() {
        let input = b"\"foobar\"";

        assert_eq!(
            vec![Token::StringLiteral("foobar", span(0, 1, input))],
            Token::lex(input).unwrap().1
        );

        let input = b"\"foo bar\"";

        assert_eq!(
            vec![Token::StringLiteral("foo bar", span(0, 1, input))],
            Token::lex(input).unwrap().1
        );

        let input = b"\"foo\\\\bar\"";

        assert_eq!(
            vec![Token::StringLiteral("foo\\\\bar", span(0, 1, input))],
            Token::lex(input).unwrap().1
        );

        let input = b"\"foo\\\"bar\"";

        assert_eq!(
            vec![Token::StringLiteral("foo\\\"bar", span(0, 1, input))],
            Token::lex(input).unwrap().1
        );

        let input = b"\"foo\nbar\"";

        assert_eq!(
            vec![
                Token::Illegal(span(0, 1, input)),
                Token::Identifer("foo", span(1, 1, input)),
                Token::EndOfLine(span(4, 1, input)),
                Token::Identifer("bar", span(5, 2, input)),
                Token::Illegal(span(8, 2, input)),
            ],
            Token::lex(input).unwrap().1
        );

        let input = b"\"foo\xF0\x9F\x92\xA9bar\"";

        assert_eq!(
            vec![
                Token::Illegal(span(0, 1, input)),
                Token::Identifer("foo", span(1, 1, input)),
                Token::Illegal(span(4, 1, input)),
                Token::Illegal(span(5, 1, input)),
                Token::Illegal(span(6, 1, input)),
                Token::Illegal(span(7, 1, input)),
                Token::Identifer("bar", span(8, 1, input)),
                Token::Illegal(span(11, 1, input)),
            ],
            Token::lex(input).unwrap().1
        );
    }

    #[test]
    fn integer_literals() {
        let input = b"042";

        assert_eq!(
            vec![Token::IntegerLiteral(42, span(0, 1, input))],
            Token::lex(input).unwrap().1
        );

        let input = b"042%";

        assert_eq!(
            vec![
                Token::IntegerLiteral(42, span(0, 1, input)),
                Token::IntegerType(span(3, 1, input))
            ],
            Token::lex(input).unwrap().1
        );

        let input = b"&hff";

        assert_eq!(
            vec![Token::IntegerLiteral(255, span(0, 1, input))],
            Token::lex(input).unwrap().1
        );

        let input = b"&HfFfF";

        assert_eq!(
            vec![Token::IntegerLiteral(65535, span(0, 1, input))],
            Token::lex(input).unwrap().1
        );
    }

    #[test]
    fn if_tree() {
        let input = b"IF (a = 10) {\n\
            RETURN a\n\
        } ELSE IF (a <> 20) {\n\
            RETURN NOT a\n\
        } ELSE IF (a > 20) {\n\
            RETURN -30 / 40 * 50\n\
        } ELSE IF (a < 30) {\n\
            RETURN TRUE\n\
        }\n\
        RETURN FALSE";

        let expected = vec![
            Token::If(span(0, 1, input)),
            Token::LeftParenthesis(span(3, 1, input)),
            Token::Identifer("a", span(4, 1, input)),
            Token::Equal(span(6, 1, input)),
            Token::IntegerLiteral(10, span(8, 1, input)),
            Token::RightParenthesis(span(10, 1, input)),
            Token::LeftCurlyBracket(span(12, 1, input)),
            Token::EndOfLine(span(13, 1, input)),
            Token::Return(span(14, 2, input)),
            Token::Identifer("a", span(21, 2, input)),
            Token::EndOfLine(span(22, 2, input)),
            Token::RightCurlyBracket(span(23, 3, input)),
            Token::Else(span(25, 3, input)),
            Token::If(span(30, 3, input)),
            Token::LeftParenthesis(span(33, 3, input)),
            Token::Identifer("a", span(34, 3, input)),
            Token::NotEqual(span(36, 3, input)),
            Token::IntegerLiteral(20, span(39, 3, input)),
            Token::RightParenthesis(span(41, 3, input)),
            Token::LeftCurlyBracket(span(43, 3, input)),
            Token::EndOfLine(span(44, 3, input)),
            Token::Return(span(45, 4, input)),
            Token::Not(span(52, 4, input)),
            Token::Identifer("a", span(56, 4, input)),
            Token::EndOfLine(span(57, 4, input)),
            Token::RightCurlyBracket(span(58, 5, input)),
            Token::Else(span(60, 5, input)),
            Token::If(span(65, 5, input)),
            Token::LeftParenthesis(span(68, 5, input)),
            Token::Identifer("a", span(69, 5, input)),
            Token::GreaterThan(span(71, 5, input)),
            Token::IntegerLiteral(20, span(73, 5, input)),
            Token::RightParenthesis(span(75, 5, input)),
            Token::LeftCurlyBracket(span(77, 5, input)),
            Token::EndOfLine(span(78, 5, input)),
            Token::Return(span(79, 6, input)),
            Token::IntegerLiteral(-30, span(86, 6, input)),
            Token::Divide(span(90, 6, input)),
            Token::IntegerLiteral(40, span(92, 6, input)),
            Token::Multiply(span(95, 6, input)),
            Token::IntegerLiteral(50, span(97, 6, input)),
            Token::EndOfLine(span(99, 6, input)),
            Token::RightCurlyBracket(span(100, 7, input)),
            Token::Else(span(102, 7, input)),
            Token::If(span(107, 7, input)),
            Token::LeftParenthesis(span(110, 7, input)),
            Token::Identifer("a", span(111, 7, input)),
            Token::LessThan(span(113, 7, input)),
            Token::IntegerLiteral(30, span(115, 7, input)),
            Token::RightParenthesis(span(117, 7, input)),
            Token::LeftCurlyBracket(span(119, 7, input)),
            Token::EndOfLine(span(120, 7, input)),
            Token::Return(span(121, 8, input)),
            Token::BooleanLiteral(true, span(128, 8, input)),
            Token::EndOfLine(span(132, 8, input)),
            Token::RightCurlyBracket(span(133, 9, input)),
            Token::EndOfLine(span(134, 9, input)),
            Token::Return(span(135, 10, input)),
            Token::BooleanLiteral(false, span(142, 10, input)),
        ];

        let (_, result) = Token::lex(input).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn id_with_numbers() {
        let input = b"hello2 hel301oo120";

        let expected = vec![
            Token::Identifer("hello2", span(0, 1, input)),
            Token::Identifer("hel301oo120", span(7, 1, input)),
        ];

        let (_, result) = Token::lex(input).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn array_tokens() {
        let input = b"[1, 2]";

        let expected = vec![
            Token::LeftSquareBracket(span(0, 1, input)),
            Token::IntegerLiteral(1, span(1, 1, input)),
            Token::Comma(span(2, 1, input)),
            Token::IntegerLiteral(2, span(4, 1, input)),
            Token::RightSquareBracket(span(5, 1, input)),
        ];

        let (_, result) = Token::lex(input).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn hash_tokens() {
        let input = b"{\"hello\": \"world\"}";

        let expected = vec![
            Token::LeftCurlyBracket(span(0, 1, input)),
            Token::StringLiteral("hello", span(1, 1, input)),
            Token::Colon(span(8, 1, input)),
            Token::StringLiteral("world", span(10, 1, input)),
            Token::RightCurlyBracket(span(17, 1, input)),
        ];

        let (_, result) = Token::lex(input).unwrap();

        assert_eq!(expected, result);
    }
}
