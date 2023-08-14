use {
    super::{
        abs_token, add_op, and_op, bool_ty, cbool_token, cbyte_token, cfloat_token, cint_token,
        comma_punc, cos_token, cstr_token, debug_location, div_op, eq_op, f32_ty, gt_op, gte_op,
        i32_ty, key_down_token, l_paren_punc, l_sq_bracket_punc, lt_op, lte_op, mul_op, ne_op,
        not_op, or_op, peek_token, r_paren_punc, r_sq_bracket_punc, rnd_token, sin_token, str_ty,
        sub_op, timer_token, u8_ty, xor_op, Literal, Span, Token, Tokens, Type, Variable,
    },
    nom::{
        branch::alt,
        bytes::complete::take,
        combinator::{map, opt, value},
        multi::separated_list0,
        sequence::{delimited, pair, preceded, tuple},
        IResult,
    },
    std::fmt::{Debug, Formatter, Result as FmtResult},
};

#[derive(Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Expression<'a> {
    // Values
    Literal(Literal<'a>),
    Tuple(Vec<Self>, Span<'a>),
    Variable(Variable<'a>),
    // Operations
    Infix(Infix, Box<Self>, Box<Self>, Span<'a>),
    Prefix(Prefix, Box<Self>, Span<'a>),
    // Functions
    Function(Variable<'a>, Vec<Self>, Span<'a>),
    ConvertBoolean(Box<Self>, Span<'a>),
    ConvertByte(Box<Self>, Span<'a>),
    ConvertFloat(Box<Self>, Span<'a>),
    ConvertInteger(Box<Self>, Span<'a>),
    ConvertString(Box<Self>, Span<'a>),
    Abs(Option<Type>, Box<Self>, Span<'a>),
    Sin(Box<Self>, Span<'a>),
    Cos(Box<Self>, Span<'a>),
    Peek(Option<Type>, Box<Self>, Span<'a>),
    KeyDown(Box<Self>, Span<'a>),
    Random(Span<'a>),
    Timer(Span<'a>),
}

impl<'a> Expression<'a> {
    pub fn location(&self) -> Span {
        match self {
            Self::Literal(lit) => lit.location(),
            Self::Variable(id) => id.location,
            Self::Tuple(.., res)
            | Self::Infix(.., res)
            | Self::Prefix(.., res)
            | Self::Function(.., res)
            | Self::ConvertBoolean(_, res)
            | Self::ConvertByte(_, res)
            | Self::ConvertFloat(_, res)
            | Self::ConvertInteger(_, res)
            | Self::ConvertString(_, res)
            | Self::Abs(.., res)
            | Self::Sin(_, res)
            | Self::Cos(_, res)
            | Self::Peek(.., res)
            | Self::KeyDown(_, res)
            | Self::Random(res)
            | Self::Timer(res) => *res,
        }
    }

    pub(super) fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        if let Ok(tuple) = Self::parse_tuple(tokens) {
            return Ok(tuple);
        }

        Self::parse_pratt(tokens, 0)
    }

    fn parse_infix(
        tokens: Tokens<'a>,
        lhs: Box<Self>,
        precedence: usize,
    ) -> IResult<Tokens<'a>, Self> {
        let location = tokens.location();
        let (tokens, op) = Infix::parse(tokens)?;
        let (tokens, rhs) = Self::parse_pratt(tokens, precedence)?;

        Ok((tokens, Self::Infix(op, lhs, Box::new(rhs), location)))
    }

    fn parse_lhs(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        alt((
            map(
                tuple((
                    cbool_token,
                    opt(bool_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::ConvertBoolean(expr, tokens.location()),
            ),
            map(
                tuple((
                    cbyte_token,
                    opt(u8_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::ConvertByte(expr, tokens.location()),
            ),
            map(
                tuple((
                    cfloat_token,
                    opt(f32_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::ConvertFloat(expr, tokens.location()),
            ),
            map(
                tuple((
                    cint_token,
                    opt(i32_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::ConvertInteger(expr, tokens.location()),
            ),
            map(
                tuple((
                    cstr_token,
                    opt(str_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::ConvertString(expr, tokens.location()),
            ),
            map(
                tuple((
                    abs_token,
                    opt(alt((
                        value(Type::Float, f32_ty),
                        value(Type::Integer, i32_ty),
                    ))),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, ty, _, expr, _)| Self::Abs(ty, expr, tokens.location()),
            ),
            map(
                tuple((
                    sin_token,
                    opt(f32_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::Sin(expr, tokens.location()),
            ),
            map(
                tuple((
                    cos_token,
                    opt(f32_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::Cos(expr, tokens.location()),
            ),
            Self::parse_peek,
            map(
                tuple((
                    key_down_token,
                    opt(bool_ty),
                    l_paren_punc,
                    map(Self::parse, Box::new),
                    r_paren_punc,
                )),
                |(_, _, _, expr, _)| Self::KeyDown(expr, tokens.location()),
            ),
            map(
                tuple((
                    rnd_token,
                    opt(f32_ty),
                    opt(pair(l_paren_punc, r_paren_punc)),
                )),
                |(_, _, _)| Self::Random(tokens.location()),
            ),
            map(
                tuple((timer_token, opt(i32_ty), l_paren_punc, r_paren_punc)),
                |(_, _, _, _)| Self::Timer(tokens.location()),
            ),
            map(
                tuple((
                    Variable::parse,
                    l_paren_punc,
                    separated_list0(comma_punc, Self::parse),
                    r_paren_punc,
                )),
                |(id, _, exprs, _)| Self::Function(id, exprs, tokens.location()),
            ),
            map(Variable::parse, Self::Variable),
            map(Literal::parse, Self::Literal),
            Self::parse_prefix,
            delimited(l_paren_punc, Self::parse, r_paren_punc),
        ))(tokens)
    }

    fn parse_peek(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            pair(
                preceded(peek_token, opt(Type::parse)),
                delimited(l_paren_punc, map(Self::parse, Box::new), r_paren_punc),
            ),
            |(ty, expr)| Self::Peek(ty, expr, tokens.location()),
        )(tokens)
    }

    fn parse_pratt(tokens: Tokens<'a>, precedence: usize) -> IResult<Tokens<'a>, Self> {
        let (tokens, lhs) = Self::parse_lhs(tokens)?;

        Self::parse_rhs(tokens, lhs, precedence)
    }

    fn parse_prefix(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            pair(Prefix::parse, map(Self::parse_lhs, Box::new)),
            |(op, expr)| Self::Prefix(op, expr, tokens.location()),
        )(tokens)
    }

    fn parse_rhs(tokens: Tokens<'a>, lhs: Self, precedence: usize) -> IResult<Tokens<'a>, Self> {
        if tokens.is_empty() {
            return Ok((tokens, lhs));
        }

        let (_, next_token) = take(1usize)(tokens)?;
        let next_precedence = Self::precedence(next_token[0]);

        if precedence < next_precedence {
            let (tokens, rhs) = Self::parse_infix(tokens, Box::new(lhs), next_precedence)?;

            Self::parse_rhs(tokens, rhs, precedence)
        } else {
            Ok((tokens, lhs))
        }
    }

    fn parse_tuple(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                l_sq_bracket_punc,
                separated_list0(comma_punc, Self::parse),
                r_sq_bracket_punc,
            ),
            |exprs| Self::Tuple(exprs, tokens.location()),
        )(tokens)
    }

    fn precedence(token: Token) -> usize {
        match token {
            Token::And(_) | Token::Or(_) | Token::Xor(_) => 1,
            Token::Equal(_)
            | Token::NotEqual(_)
            | Token::LessThanEqual(_)
            | Token::GreaterThanEqual(_)
            | Token::LessThan(_)
            | Token::GreaterThan(_) => 2,
            Token::Add(_) | Token::Subtract(_) => 3,
            Token::Multiply(_) | Token::Divide(_) => 4,
            _ => 0,
        }
    }
}

impl<'a> Debug for Expression<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Literal(lit) => lit.fmt(f)?,
            Self::Tuple(exprs, location) => {
                f.write_str("Tuple (")?;

                let mut first = true;
                for expr in exprs {
                    if first {
                        first = false;
                    } else {
                        f.write_str(", ")?;
                    }

                    expr.fmt(f)?;
                }

                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Variable(var) => var.fmt(f)?,
            Self::Infix(op, lhs, rhs, location) => {
                f.write_str("Infix (")?;
                lhs.fmt(f)?;
                f.write_str(" ")?;
                op.fmt(f)?;
                f.write_str(" ")?;
                rhs.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Prefix(op, expr, location) => {
                f.write_str("Prefix (")?;
                op.fmt(f)?;
                f.write_str(" ")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Function(id, exprs, location) => {
                f.write_fmt(format_args!("Function `{}` ", id.name))?;

                if let Some(ty) = id.ty {
                    f.write_fmt(format_args!("{} ", ty))?;
                }

                f.write_str("(")?;

                let mut first = true;
                for expr in exprs {
                    if first {
                        first = false;
                    } else {
                        f.write_str(", ")?;
                    }

                    expr.fmt(f)?;
                }

                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::ConvertBoolean(expr, location) => {
                f.write_str("ConvertBoolean (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::ConvertByte(expr, location) => {
                f.write_str("ConvertByte (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::ConvertFloat(expr, location) => {
                f.write_str("ConvertFloat (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::ConvertInteger(expr, location) => {
                f.write_str("ConvertInteger (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::ConvertString(expr, location) => {
                f.write_str("ConvertString (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Abs(ty, expr, location) => {
                f.write_str("Abs ")?;

                if let Some(ty) = ty {
                    f.write_fmt(format_args!("{} ", ty))?;
                }

                f.write_str("(")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Sin(expr, location) => {
                f.write_str("Sin (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Cos(expr, location) => {
                f.write_str("Cos (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Peek(ty, expr, location) => {
                f.write_str("Peek ")?;

                if let Some(ty) = ty {
                    f.write_fmt(format_args!("{} ", ty))?;
                }

                f.write_str("(")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::KeyDown(expr, location) => {
                f.write_str("KeyDown (")?;
                expr.fmt(f)?;
                f.write_str(") ")?;
                debug_location(f, *location)?;
            }
            Self::Random(location) => {
                f.write_str("Rnd ")?;
                debug_location(f, *location)?;
            }
            Self::Timer(location) => {
                f.write_str("Timer ")?;
                debug_location(f, *location)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Prefix {
    Minus,
    Negate,
}

impl Prefix {
    fn parse(tokens: Tokens) -> IResult<Tokens, Self> {
        alt((map(sub_op, |_| Self::Minus), map(not_op, |_| Self::Negate)))(tokens)
    }
}

impl Debug for Prefix {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Minus => f.write_str("Minus"),
            Self::Negate => f.write_str("Negate"),
        }
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Infix {
    Add,
    Subtract,
    Divide,
    Multiply,
    Bitwise(Bitwise),
    Relation(Relation),
}

impl Infix {
    fn parse(tokens: Tokens) -> IResult<Tokens, Self> {
        alt((
            map(add_op, |_| Self::Add),
            map(sub_op, |_| Self::Subtract),
            map(div_op, |_| Self::Divide),
            map(mul_op, |_| Self::Multiply),
            map(Bitwise::parse, Self::Bitwise),
            map(Relation::parse, Self::Relation),
        ))(tokens)
    }
}

impl Debug for Infix {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Add => f.write_str("Add"),
            Self::Subtract => f.write_str("Subtract"),
            Self::Divide => f.write_str("Divide"),
            Self::Multiply => f.write_str("Multiply"),
            Self::Bitwise(Bitwise::And) => f.write_str("And"),
            Self::Bitwise(Bitwise::Not) => f.write_str("Not"),
            Self::Bitwise(Bitwise::Or) => f.write_str("Or"),
            Self::Bitwise(Bitwise::Xor) => f.write_str("Xor"),
            Self::Relation(Relation::Equal) => f.write_str("Equal"),
            Self::Relation(Relation::GreaterThan) => f.write_str("GreaterThan"),
            Self::Relation(Relation::GreaterThanEqual) => f.write_str("GreaterThanEqual"),
            Self::Relation(Relation::LessThan) => f.write_str("LessThan"),
            Self::Relation(Relation::LessThanEqual) => f.write_str("LessThanEqual"),
            Self::Relation(Relation::NotEqual) => f.write_str("NotEqual"),
        }
    }
}

impl std::fmt::Display for Infix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => f.write_str("+"),
            Self::Subtract => f.write_str("-"),
            Self::Divide => f.write_str("/"),
            Self::Multiply => f.write_str("*"),
            Self::Bitwise(r) => <dyn Debug>::fmt(r, f),
            Self::Relation(r) => <dyn Debug>::fmt(r, f),
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Bitwise {
    Not,
    And,
    Or,
    Xor,
}

impl Bitwise {
    fn parse(tokens: Tokens) -> IResult<Tokens, Self> {
        alt((
            map(not_op, |_| Self::Not),
            map(and_op, |_| Self::And),
            map(or_op, |_| Self::Or),
            map(xor_op, |_| Self::Xor),
        ))(tokens)
    }
}

impl std::fmt::Display for Bitwise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Not => "NOT",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Xor => "XOR",
        })
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Relation {
    Equal,
    NotEqual,
    GreaterThanEqual,
    LessThanEqual,
    GreaterThan,
    LessThan,
}

impl Relation {
    fn parse(tokens: Tokens) -> IResult<Tokens, Self> {
        alt((
            map(eq_op, |_| Self::Equal),
            map(ne_op, |_| Self::NotEqual),
            map(gte_op, |_| Self::GreaterThanEqual),
            map(lte_op, |_| Self::LessThanEqual),
            map(gt_op, |_| Self::GreaterThan),
            map(lt_op, |_| Self::LessThan),
        ))(tokens)
    }
}

impl std::fmt::Display for Relation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Equal => "=",
            Self::NotEqual => "<>",
            Self::GreaterThanEqual => ">=",
            Self::LessThanEqual => "<=",
            Self::GreaterThan => ">",
            Self::LessThan => "<",
        })
    }
}

#[cfg(test)]
mod tests {
    use {
        super::{
            super::{tests::same_parsed, Token},
            *,
        },
        crate::tests::span,
    };

    #[test]
    fn literal() {
        let input = b"TRUE";

        let expected = Expression::Literal(Literal::Boolean(true, span(0, 1, input)));

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn infix() {
        let input = b"1 + 2";

        let expected = Expression::Infix(
            Infix::Add,
            Box::new(Expression::Literal(Literal::Integer(1, span(0, 1, input)))),
            Box::new(Expression::Literal(Literal::Integer(2, span(4, 1, input)))),
            span(2, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn infix_parens() {
        let input = b"(1 + 2)";

        let expected = Expression::Infix(
            Infix::Add,
            Box::new(Expression::Literal(Literal::Integer(1, span(1, 1, input)))),
            Box::new(Expression::Literal(Literal::Integer(2, span(5, 1, input)))),
            span(3, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn infix_order() {
        let input = b"6 * 2 + 1";

        let expected = Expression::Infix(
            Infix::Add,
            Box::new(Expression::Infix(
                Infix::Multiply,
                Box::new(Expression::Literal(Literal::Integer(6, span(0, 1, input)))),
                Box::new(Expression::Literal(Literal::Integer(2, span(4, 1, input)))),
                span(2, 1, input),
            )),
            Box::new(Expression::Literal(Literal::Integer(1, span(8, 1, input)))),
            span(6, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result, "{:#?}", result);
    }

    #[test]
    fn infix_relations() {
        let input = b"1 = 1 AND TRUE = FALSE";

        let expected = Expression::Infix(
            Infix::Bitwise(Bitwise::And),
            Box::new(Expression::Infix(
                Infix::Relation(Relation::Equal),
                Box::new(Expression::Literal(Literal::Integer(1, span(0, 1, input)))),
                Box::new(Expression::Literal(Literal::Integer(1, span(4, 1, input)))),
                span(2, 1, input),
            )),
            Box::new(Expression::Infix(
                Infix::Relation(Relation::Equal),
                Box::new(Expression::Literal(Literal::Boolean(
                    true,
                    span(10, 1, input),
                ))),
                Box::new(Expression::Literal(Literal::Boolean(
                    false,
                    span(17, 1, input),
                ))),
                span(15, 1, input),
            )),
            span(6, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn prefix_found() {
        let input = b"-a + 1";

        let expected = Expression::Infix(
            Infix::Add,
            Box::new(Expression::Prefix(
                Prefix::Minus,
                Box::new(Expression::Variable(Variable {
                    name: "a",
                    location: span(1, 1, input),
                    ty: None,
                })),
                span(0, 1, input),
            )),
            Box::new(Expression::Literal(Literal::Integer(1, span(5, 1, input)))),
            span(3, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn prefix_not_found() {
        let input = b"-5 + 1";

        let expected = Expression::Infix(
            Infix::Add,
            Box::new(Expression::Literal(Literal::Integer(-5, span(0, 1, input)))),
            Box::new(Expression::Literal(Literal::Integer(1, span(5, 1, input)))),
            span(3, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn math_expressions1() {
        let input = b"sin(1.2)";

        let expected = Expression::Sin(
            Box::new(Expression::Literal(Literal::Float(1.2, span(4, 1, input)))),
            span(0, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn math_expressions2() {
        let input = b"1.0 + sin(1.2) * 2.0";

        let expected = Expression::Infix(
            Infix::Add,
            Box::new(Expression::Literal(Literal::Float(1.0, span(0, 1, input)))),
            Box::new(Expression::Infix(
                Infix::Multiply,
                Box::new(Expression::Sin(
                    Box::new(Expression::Literal(Literal::Float(1.2, span(10, 1, input)))),
                    span(6, 1, input),
                )),
                Box::new(Expression::Literal(Literal::Float(2.0, span(17, 1, input)))),
                span(15, 1, input),
            )),
            span(4, 1, input),
        );

        let (_, tokens) = Token::lex(input).unwrap();
        let (_, result) = Expression::parse(Tokens::new(&tokens)).unwrap();

        assert_eq!(expected, result);
    }

    #[test]
    fn op_precedence() {
        assert!(same_parsed(b"DIM a = 1 + 1", b"DIM a = 1 + 1",));
        assert!(same_parsed(b"DIM a = 6 * 2 + 1", b"DIM a = (6 * 2) + 1",));
        assert!(same_parsed(
            b"DIM a = 6 * 2 + 16 / 2",
            b"DIM a = (6 * 2) + (16 / 2)",
        ));
        assert!(same_parsed(b"DIM a = -5", b"DIM a = (-5)",));
        assert!(same_parsed(b"DIM a = -5 + 1", b"DIM a = (-5) + 1",));
        assert!(same_parsed(
            b"DIM a = 6 * -2 + 16 / 2",
            b"DIM a = (6 * (-2)) + (16 / 2)",
        ));
        assert!(same_parsed(
            b"DIM a = -6 + -2 - -8",
            b"DIM a = ((-6) + (-2)) - (-8)",
        ));
        assert!(same_parsed(
            b"DIM a = -6 + -2 - -8",
            b"DIM a = (((-6) + (-2)) - (-8))",
        ));
        assert!(same_parsed(
            b"DIM a = -6 * -2 / -8",
            b"DIM a = ((-6) * (-2)) / (-8)",
        ));
        assert!(same_parsed(
            b"DIM a = (-6 * -2 / -8)",
            b"DIM a = ((-6) * (-2)) / (-8)",
        ));
        assert!(same_parsed(
            b"DIM a = NOT TRUE AND NOT FALSE",
            b"DIM a = (NOT TRUE) AND (NOT FALSE)",
        ));
        assert!(same_parsed(
            b"DIM a = TRUE AND FALSE <> TRUE AND TRUE",
            b"DIM a = (TRUE AND (FALSE <> TRUE)) AND TRUE",
        ));
        assert!(same_parsed(
            b"DIM a = 1 = 1 AND TRUE = FALSE",
            b"DIM a = (1 = 1) AND (TRUE = FALSE)",
        ));
        assert!(same_parsed(
            b"DIM a = b OR c OR d",
            b"DIM a = (b OR c) OR d",
        ));
        assert!(same_parsed(
            b"DIM a = 1.0 + COS(2.0) * 3.0",
            b"DIM a = 1.0 + (COS(2.0) * 3.0)",
        ));

        // The rest should *NOT* match
        assert!(!same_parsed(b"DIM a = 1 + TRUE", b"DIM a = 1 + 2",));
    }
}
