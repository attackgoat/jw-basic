mod expr;
mod literal;

pub use self::{
    expr::{Bitwise, Expression, Infix, Prefix, Relation},
    literal::Literal,
};

use {
    super::token::{debug_location, Span, Token, Tokens},
    log::error,
    nom::{
        branch::alt,
        bytes::complete::take,
        combinator::{map, map_res, opt, value, verify},
        error::{Error, ErrorKind},
        multi::{many0, separated_list0, separated_list1},
        sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
        Err, IResult,
    },
    std::{
        error::Error as StdError,
        fmt::{Debug, Display, Formatter, Result as FmtResult},
        ops::Range,
        str::from_utf8,
    },
};

macro_rules! token (
    ($func_name: ident, $token: ident) => (
        fn $func_name(tokens: Tokens) -> IResult<Tokens, Token> {
            map(
                verify(take(1usize),
                   |tokens: &Tokens| matches!(tokens[0], Token::$token(..))
                ),
                |tokens: Tokens| tokens[0]
            )(tokens)
        }
    )
);

// Identifiers
token!(id_token, Identifer);

// Literals
token!(bool_lit, BooleanLiteral);
token!(f32_lit, FloatLiteral);
token!(i32_lit, IntegerLiteral);
token!(str_lit, StringLiteral);

// Operations
token!(add_op, Add);
token!(sub_op, Subtract);
token!(mul_op, Multiply);
token!(div_op, Divide);
token!(eq_op, Equal);
token!(ne_op, NotEqual);
token!(gte_op, GreaterThanEqual);
token!(lte_op, LessThanEqual);
token!(gt_op, GreaterThan);
token!(lt_op, LessThan);
token!(not_op, Not);
token!(and_op, And);
token!(or_op, Or);
token!(xor_op, Xor);

// Reserved Words
token!(cbool_token, ConvertBoolean);
token!(cbyte_token, ConvertByte);
token!(cfloat_token, ConvertFloat);
token!(cint_token, ConvertInteger);
token!(cstr_token, ConvertString);
token!(abs_token, Sin);
token!(sin_token, Sin);
token!(cos_token, Cos);
token!(cls_token, ClearScreen);
token!(color_token, Color);
token!(dim_token, Dimension);
token!(else_token, Else);
token!(end_token, End);
token!(for_token, For);
token!(fn_token, Function);
token!(goto_token, Goto);
token!(if_token, If);
token!(line_token, Line);
token!(next_token, Next);
token!(print_token, Print);
token!(rect_token, Rectangle);
token!(return_token, Return);
token!(step_token, Step);
token!(sub_token, Subroutine);
token!(then_token, Then);
token!(timer_token, Timer);
token!(to_token, To);
token!(while_token, While);
token!(while_end_token, WhileEnd);
token!(yield_token, Yield);

// Punctuations
token!(colon_punc, Colon);
token!(comma_punc, Comma);
token!(continuation_punc, Continuation);
token!(end_of_line_punc, EndOfLine);
token!(l_curly_bracket_punc, LeftCurlyBracket);
token!(l_paren_punc, LeftParenthesis);
token!(l_sq_bracket_punc, LeftSquareBracket);
token!(r_curly_bracket_punc, RightCurlyBracket);
token!(r_paren_punc, RightParenthesis);
token!(r_sq_bracket_punc, RightSquareBracket);
token!(semicolon_punc, Semicolon);

// Types
token!(bool_ty, BooleanType);
token!(u8_ty, ByteType);
token!(f32_ty, FloatType);
token!(i32_ty, IntegerType);
token!(str_ty, StringType);

pub type Ast<'a> = Vec<Syntax<'a>>;
pub type SubscriptRange<'a> = (Option<Expression<'a>>, Expression<'a>);

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Identifier<'a> {
    pub location: Span<'a>,
    pub name: &'a str,
    pub ty: Option<Type>,
}

impl<'a> Identifier<'a> {
    fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            pair(
                map(id_token, |token| token.identifier().unwrap()),
                opt(Type::parse),
            ),
            |(name, ty)| Self {
                location: tokens.location(),
                name,
                ty,
            },
        )(tokens)
    }
}

impl<'a> Debug for Identifier<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!("Identifier `{}` ", self.name))?;

        if let Some(ty) = self.ty {
            <dyn Debug>::fmt(&ty, f)?;
            f.write_str(" ")?;
        }

        debug_location(f, self.location)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Case<'a> {
    Expression(Expression<'a>),
    RangeFull(Expression<'a>, Expression<'a>),
    Relational(Relation, Expression<'a>),
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Label<'a> {
    Name(&'a str, Span<'a>),
    Number(u32, Span<'a>),
}

impl<'a> Label<'a> {
    pub fn location(self) -> Span<'a> {
        match self {
            Self::Name(_, location) | Self::Number(_, location) => location,
        }
    }

    fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        alt((
            map(id_token, |token| {
                Self::Name(token.identifier().unwrap(), tokens.location())
            }),
            map(
                verify(
                    map(i32_lit, |token| token.integer_literal().unwrap()),
                    |lit| *lit >= 0,
                ),
                |lit| Self::Number(lit as u32, tokens.location()),
            ),
        ))(tokens)
    }
}

impl<'a> Debug for Label<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(match self {
            Self::Name(..) => "Name",
            Self::Number(..) => "Number",
        })?;
        f.write_fmt(format_args!(" `{self}` ",))?;
        debug_location(f, self.location())
    }
}

impl<'a> Display for Label<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Name(name, _) => f.write_str(name),
            Self::Number(number, _) => f.write_fmt(format_args!("{number}")),
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Print<'a> {
    Expression(Expression<'a>),
    Tab,
}

impl<'a> Print<'a> {
    fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        alt((
            map(comma_punc, |_| Self::Tab),
            map(
                terminated(Expression::parse, opt(semicolon_punc)),
                Self::Expression,
            ),
        ))(tokens)
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Syntax<'a> {
    Assign(Identifier<'a>, Option<Vec<Expression<'a>>>, Expression<'a>),
    Color(Expression<'a>, Option<Expression<'a>>),
    ClearScreen,
    Dimension(
        Vec<(
            Identifier<'a>,
            Option<Vec<SubscriptRange<'a>>>,
            Option<Expression<'a>>,
        )>,
    ),
    For(
        Identifier<'a>,
        Expression<'a>,
        Expression<'a>,
        Option<Expression<'a>>,
        Ast<'a>,
    ),
    // GetData {
    //     address: usize,
    //     result: StackAddress,
    // },
    // GetPalette {
    //     color: Expression,
    //     result_r: StackAddress,
    //     result_g: StackAddress,
    //     result_b: StackAddress,
    // },
    // GetPixel {
    //     x: Expression,
    //     y: Expression,
    //     result: StackAddress,
    // },
    Goto(Label<'a>),
    If {
        tests: Vec<(Expression<'a>, Ast<'a>)>,
        default: Option<Ast<'a>>,
    },
    Label(Label<'a>),
    Line(
        Option<(Expression<'a>, Expression<'a>)>,
        (Expression<'a>, Expression<'a>),
        Expression<'a>,
    ),
    // Input {
    //     prompt: Expression,
    //     result: StackAddress,
    // },
    // Line {
    //     x0: Expression,
    //     y0: Expression,
    //     x1: Expression,
    //     y2: Expression,
    //     color: Expression,
    // },
    // Locate {
    //     col: Expression,
    //     row: Expression,
    // },
    Print(Vec<Print<'a>>),
    Rectangle(
        Option<(Expression<'a>, Expression<'a>)>,
        (Expression<'a>, Expression<'a>),
        Expression<'a>,
        Option<Expression<'a>>,
    ),
    Select {
        test: Expression<'a>,
        cases: Vec<(Vec<Case<'a>>, Ast<'a>)>,
        default: Option<(Case<'a>, Ast<'a>)>,
    }, // SetData {
    //     address: usize,
    //     value: Expression,
    // },
    // SetPalette {
    //     color: Expression,
    //     r: Expression,
    //     g: Expression,
    //     b: Expression,
    // },
    // SetPixel {
    //     x: Expression,
    //     y: Expression,
    //     color: Expression,
    // },
    While {
        test_expr: Expression<'a>,
        body_ast: Ast<'a>,
    },
    Yield,
}

impl<'a> Syntax<'a> {
    pub fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Ast> {
        many0(delimited(
            many0(end_of_line_punc),
            alt((
                Self::parse_assign,
                Self::parse_cls,
                Self::parse_color,
                Self::parse_dim,
                Self::parse_for,
                Self::parse_goto,
                Self::parse_if,
                Self::parse_line,
                Self::parse_print,
                Self::parse_rect,
                Self::parse_while,
                Self::parse_yield,
                Self::parse_label,
            )),
            many0(end_of_line_punc),
        ))(tokens)
    }

    fn parse_assign(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            terminated(
                tuple((
                    Identifier::parse,
                    opt(delimited(
                        l_paren_punc,
                        separated_list1(comma_punc, Expression::parse),
                        r_paren_punc,
                    )),
                    eq_op,
                    Expression::parse,
                )),
                opt(end_of_line_punc),
            ),
            |(id, indices, _, expr)| Self::Assign(id, indices, expr),
        )(tokens)
    }

    fn parse_cls(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(cls_token, opt(end_of_line_punc)), |_| {
            Self::ClearScreen
        })(tokens)
    }

    fn parse_color(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                color_token,
                pair(
                    Expression::parse,
                    opt(preceded(comma_punc, Expression::parse)),
                ),
                end_of_line_punc,
            ),
            |(foreground, background)| Self::Color(foreground, background),
        )(tokens)
    }

    fn parse_dim(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        fn subscript<'a>(tokens: Tokens<'a>) -> IResult<Tokens<'a>, SubscriptRange<'a>> {
            pair(
                opt(terminated(Expression::parse, to_token)),
                Expression::parse,
            )(tokens)
        }

        fn subscripts<'a>(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Vec<SubscriptRange<'a>>> {
            delimited(
                l_paren_punc,
                separated_list0(comma_punc, subscript),
                r_paren_punc,
            )(tokens)
        }

        map(
            delimited(
                dim_token,
                separated_list1(
                    comma_punc,
                    tuple((
                        Identifier::parse,
                        opt(subscripts),
                        opt(preceded(eq_op, Expression::parse)),
                    )),
                ),
                opt(end_of_line_punc),
            ),
            Self::Dimension,
        )(tokens)
    }

    fn parse_for(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        let (tokens, (_, var, _, start, _, end, step, _)) = tuple((
            for_token,
            Identifier::parse,
            eq_op,
            Expression::parse,
            to_token,
            Expression::parse,
            opt(preceded(step_token, Expression::parse)),
            opt(end_of_line_punc),
        ))(tokens)?;
        let (tokens, ast) = Self::parse(tokens)?;
        let (tokens, (..)) = tuple((
            next_token,
            opt(verify(Identifier::parse, |next_var| {
                next_var.name == var.name && next_var.ty == var.ty
            })),
            opt(end_of_line_punc),
        ))(tokens)?;

        Ok((tokens, Self::For(var, start, end, step, ast)))
    }

    fn parse_goto(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(goto_token, Label::parse, opt(end_of_line_punc)),
            Self::Goto,
        )(tokens)
    }

    fn parse_if(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        fn if_expr_then(tokens: Tokens) -> IResult<Tokens, Expression> {
            delimited(
                if_token,
                Expression::parse,
                pair(then_token, end_of_line_punc),
            )(tokens)
        }

        fn else_if_expr_then(tokens: Tokens) -> IResult<Tokens, Expression> {
            delimited(
                pair(else_token, if_token),
                Expression::parse,
                pair(then_token, end_of_line_punc),
            )(tokens)
        }

        fn else_then(tokens: Tokens) -> IResult<Tokens, ()> {
            value((), tuple((else_token, then_token, end_of_line_punc)))(tokens)
        }

        fn end_if(tokens: Tokens) -> IResult<Tokens, ()> {
            value((), tuple((end_token, if_token, opt(end_of_line_punc))))(tokens)
        }

        map(
            tuple((
                pair(if_expr_then, Self::parse),
                many0(pair(else_if_expr_then, Self::parse)),
                opt(preceded(else_then, Self::parse)),
                end_if,
            )),
            |(test, mut tests, default, ())| {
                tests.insert(0, test);

                Self::If { tests, default }
            },
        )(tokens)
    }

    fn parse_label(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            terminated(Label::parse, opt(alt((end_of_line_punc, colon_punc)))),
            Self::Label,
        )(tokens)
    }

    fn parse_line(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        fn coord<'a>(tokens: Tokens<'a>) -> IResult<Tokens<'a>, (Expression<'a>, Expression<'a>)> {
            delimited(
                l_paren_punc,
                separated_pair(Expression::parse, comma_punc, Expression::parse),
                r_paren_punc,
            )(tokens)
        }

        map(
            tuple((
                line_token,
                opt(terminated(coord, sub_op)),
                coord,
                comma_punc,
                Expression::parse,
            )),
            |(_, from_coord, to_coord, _, color)| Self::Line(from_coord, to_coord, color),
        )(tokens)
    }

    fn parse_print(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(print_token, many0(Print::parse), opt(end_of_line_punc)),
            Self::Print,
        )(tokens)
    }

    fn parse_rect(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        fn coord<'a>(tokens: Tokens<'a>) -> IResult<Tokens<'a>, (Expression<'a>, Expression<'a>)> {
            delimited(
                l_paren_punc,
                separated_pair(Expression::parse, comma_punc, Expression::parse),
                r_paren_punc,
            )(tokens)
        }

        map(
            tuple((
                rect_token,
                opt(terminated(coord, sub_op)),
                coord,
                comma_punc,
                Expression::parse,
                opt(preceded(comma_punc, Expression::parse)),
            )),
            |(_, from_coord, to_coord, _, foreground_color, background_color)| {
                Self::Rectangle(from_coord, to_coord, foreground_color, background_color)
            },
        )(tokens)
    }

    fn parse_while(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            tuple((
                while_token,
                Expression::parse,
                end_of_line_punc,
                Self::parse,
                while_end_token,
                opt(end_of_line_punc),
            )),
            |(_, test_expr, _, body_ast, ..)| Self::While {
                test_expr,
                body_ast,
            },
        )(tokens)
    }

    fn parse_yield(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(yield_token, opt(end_of_line_punc)), |_| {
            Self::Yield
        })(tokens)
    }
}

#[derive(Debug)]
pub struct SyntaxError {
    pub column_number: usize,
    pub line: String,
    pub line_number: u32,
    pub reason: String,
}

impl SyntaxError {
    pub fn from_location(location: Span, reason: impl Into<String>) -> Self {
        Self {
            column_number: location.get_column(),
            line: from_utf8(location.get_line_beginning())
                .unwrap_or_default()
                .to_owned(),
            line_number: location.location_line(),
            reason: reason.into(),
        }
    }
}

impl StdError for SyntaxError {}

impl Display for SyntaxError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "line {}, column {}: {}",
            self.line_number, self.column_number, self.reason
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Type {
    Boolean,
    Byte,
    Float,
    Integer,
    String,
}

impl Type {
    fn parse(tokens: Tokens) -> IResult<Tokens, Self> {
        alt((
            map(bool_ty, |_| Self::Boolean),
            map(u8_ty, |_| Self::Byte),
            map(f32_ty, |_| Self::Float),
            map(i32_ty, |_| Self::Integer),
            map(str_ty, |_| Self::String),
        ))(tokens)
    }

    pub fn symbol(self) -> char {
        match self {
            Self::Boolean => '?',
            Self::Byte => '@',
            Self::Float => '!',
            Self::Integer => '%',
            Self::String => '$',
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(match self {
            Self::Boolean => "boolean",
            Self::Byte => "byte",
            Self::Float => "float",
            Self::Integer => "integer",
            Self::String => "string",
        })
    }
}

#[cfg(test)]
mod tests {
    use {super::*, crate::tests::span};

    /// Asserts the AST defined by two syntaxes is equal; does not compare the original source code
    /// location of each tag.
    fn same_syntax(lhs: &Syntax, rhs: &Syntax) -> bool {
        fn compare_id(lhs: Identifier, rhs: Identifier) -> bool {
            lhs.name == rhs.name && lhs.ty == rhs.ty
        }

        fn compare_lit(lhs: Literal, rhs: Literal) -> bool {
            match lhs {
                Literal::Boolean(lhs, _) => matches!(rhs, Literal::Boolean(rhs, _) if lhs == rhs),
                Literal::Byte(lhs, _) => matches!(rhs, Literal::Byte(rhs, _) if lhs == rhs),
                Literal::Float(lhs, _) => matches!(rhs, Literal::Float(rhs, _) if lhs == rhs),
                Literal::Integer(lhs, _) => matches!(rhs, Literal::Integer(rhs, _) if lhs == rhs),
                Literal::String(lhs, _) => matches!(rhs, Literal::String(rhs, _) if lhs == rhs),
            }
        }

        fn compare_expr(lhs: &Expression, rhs: &Expression) -> bool {
            match lhs {
                Expression::Literal(lhs) => {
                    matches!(rhs, Expression::Literal(rhs) if compare_lit(*lhs, *rhs))
                }
                Expression::Tuple(lhs, _) => {
                    matches!(rhs, Expression::Tuple(rhs, _) if compare_vec(lhs, rhs, compare_expr))
                }
                Expression::Variable(lhs) => {
                    matches!(rhs, Expression::Variable(rhs) if compare_id(*lhs, *rhs))
                }
                Expression::Infix(lhs_op, lhs1, lhs2, _) => {
                    matches!(rhs, Expression::Infix(rhs_op, rhs1, rhs2, _) if *lhs_op == *rhs_op && compare_expr(lhs1, rhs1) && compare_expr(lhs2, rhs2))
                }
                Expression::Prefix(lhs_op, lhs, _) => {
                    matches!(rhs, Expression::Prefix(rhs_op, rhs, _) if lhs_op == rhs_op && compare_expr(lhs, rhs))
                }
                Expression::Function(lhs_id, lhs, _) => {
                    matches!(rhs, Expression::Function(rhs_id, rhs, _) if compare_id(*lhs_id, *rhs_id) && compare_vec(lhs,rhs, compare_expr))
                }
                Expression::ConvertBoolean(lhs, _) => {
                    matches!(rhs, Expression::ConvertBoolean(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::ConvertByte(lhs, _) => {
                    matches!(rhs, Expression::ConvertByte(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::ConvertFloat(lhs, _) => {
                    matches!(rhs, Expression::ConvertFloat(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::ConvertInteger(lhs, _) => {
                    matches!(rhs, Expression::ConvertInteger(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::ConvertString(lhs, _) => {
                    matches!(rhs, Expression::ConvertString(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::Abs(lhs_ty, lhs, _) => {
                    matches!(rhs, Expression::Abs(rhs_ty, rhs, _) if compare_opt(lhs_ty, rhs_ty, Type::eq) && compare_expr(lhs, rhs))
                }
                Expression::Sin(lhs, _) => {
                    matches!(rhs, Expression::Sin(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::Cos(lhs, _) => {
                    matches!(rhs, Expression::Cos(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::Timer(_) => matches!(rhs, Expression::Timer(_)),
            }
        }

        fn compare_subscript_range(lhs: &SubscriptRange, rhs: &SubscriptRange) -> bool {
            compare_opt(&lhs.0, &rhs.0, compare_expr) && compare_expr(&lhs.1, &rhs.1)
        }

        fn compare_dim(
            lhs: &(Identifier, Option<Vec<SubscriptRange>>, Option<Expression>),
            rhs: &(Identifier, Option<Vec<SubscriptRange>>, Option<Expression>),
        ) -> bool {
            compare_id(lhs.0, rhs.0)
                && compare_opt(&lhs.1, &rhs.1, |lhs, rhs| {
                    compare_vec(lhs, rhs, compare_subscript_range)
                })
                && compare_opt(&lhs.2, &rhs.2, compare_expr)
        }

        fn compare_opt<T, F>(lhs: &Option<T>, rhs: &Option<T>, f: F) -> bool
        where
            F: Fn(&T, &T) -> bool,
        {
            (lhs.is_none() && rhs.is_none())
                || (lhs.is_some()
                    && rhs.is_some()
                    && f(lhs.as_ref().unwrap(), rhs.as_ref().unwrap()))
        }

        fn compare_vec<T, F>(lhs: &Vec<T>, rhs: &Vec<T>, f: F) -> bool
        where
            F: Fn(&T, &T) -> bool,
        {
            lhs.len() == rhs.len() && lhs.iter().zip(rhs.iter()).all(|(lhs, rhs)| f(lhs, rhs))
        }

        match lhs {
            Syntax::ClearScreen => matches!(rhs, Syntax::ClearScreen),
            Syntax::Dimension(lhs) => {
                matches!(rhs, Syntax::Dimension(rhs) if compare_vec(lhs, rhs, compare_dim))
            }
            _ => false,
        }
    }

    pub fn same_parsed(lhs: &[u8], rhs: &[u8]) -> bool {
        let (_, lhs) = Token::lex(lhs).unwrap();
        let (_, lhs) = Syntax::parse(Tokens::new(&lhs)).unwrap();

        let (_, rhs) = Token::lex(rhs).unwrap();
        let (_, rhs) = Syntax::parse(Tokens::new(&rhs)).unwrap();

        lhs.len() == rhs.len()
            && lhs
                .iter()
                .zip(rhs.iter())
                .all(|(lhs, rhs)| same_syntax(lhs, rhs))
    }

    #[test]
    fn assign() {
        let input = b"var1 = 5";

        let expected = vec![Syntax::Assign(
            Identifier {
                location: span(0, 1, input),
                name: "var1",
                ty: None,
            },
            None,
            Expression::Literal(Literal::Integer(5, span(7, 1, input))),
        )];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn dim_array1_typed_integer() {
        let input = b"DIM var1%(-100 TO 100) = 5";

        let expected = vec![Syntax::Dimension(vec![(
            Identifier {
                location: span(4, 1, input),
                name: "var1",
                ty: Some(Type::Integer),
            },
            Some(vec![(
                Some(Expression::Literal(Literal::Integer(
                    -100,
                    span(10, 1, input),
                ))),
                Expression::Literal(Literal::Integer(100, span(18, 1, input))),
            )]),
            Some(Expression::Literal(Literal::Integer(5, span(25, 1, input)))),
        )])];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn dim_integer_literal() {
        let input = b"DIM var1 = 5";

        let expected = vec![Syntax::Dimension(vec![(
            Identifier {
                location: span(4, 1, input),
                name: "var1",
                ty: None,
            },
            None,
            Some(Expression::Literal(Literal::Integer(5, span(11, 1, input)))),
        )])];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn dim_integer_typed_literal() {
        let input = b"DIM var1 = 5%";

        let expected = vec![Syntax::Dimension(vec![(
            Identifier {
                location: span(4, 1, input),
                name: "var1",
                ty: None,
            },
            None,
            Some(Expression::Literal(Literal::Integer(5, span(11, 1, input)))),
        )])];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn dim_typed_integer_literal() {
        let input = b"DIM var1% = 5";

        let expected = vec![Syntax::Dimension(vec![(
            Identifier {
                location: span(4, 1, input),
                name: "var1",
                ty: Some(Type::Integer),
            },
            None,
            Some(Expression::Literal(Literal::Integer(5, span(12, 1, input)))),
        )])];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn dim_typed_integer_typed_literal() {
        let input = b"DIM var1% = 5%";

        let expected = vec![Syntax::Dimension(vec![(
            Identifier {
                location: span(4, 1, input),
                name: "var1",
                ty: Some(Type::Integer),
            },
            None,
            Some(Expression::Literal(Literal::Integer(5, span(12, 1, input)))),
        )])];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn dim_integer_literal_string_literal() {
        let input = b"DIM var1 = 5, var2 = \"test\"";

        let expected = vec![Syntax::Dimension(vec![
            (
                Identifier {
                    location: span(4, 1, input),
                    name: "var1",
                    ty: None,
                },
                None,
                Some(Expression::Literal(Literal::Integer(5, span(11, 1, input)))),
            ),
            (
                Identifier {
                    location: span(14, 1, input),
                    name: "var2",
                    ty: None,
                },
                None,
                Some(Expression::Literal(Literal::String(
                    "test",
                    span(21, 1, input),
                ))),
            ),
        ])];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn dim_integer_expression_infix_literals() {
        let input = b"DIM var1 = 1 + 2";

        let expected = vec![Syntax::Dimension(vec![(
            Identifier {
                location: span(4, 1, input),
                name: "var1",
                ty: None,
            },
            None,
            Some(Expression::Infix(
                expr::Infix::Add,
                Box::new(Expression::Literal(Literal::Integer(1, span(11, 1, input)))),
                Box::new(Expression::Literal(Literal::Integer(2, span(15, 1, input)))),
                span(13, 1, input),
            )),
        )])];

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Syntax::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }
}
