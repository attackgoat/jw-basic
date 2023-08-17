mod expr;
mod literal;

pub use self::{
    expr::{Bitwise, Expression, Infix, Prefix, Relation},
    literal::Literal,
};

use {
    super::token::{debug_location, Span, Token, Tokens},
    nom::{
        branch::alt,
        bytes::complete::take,
        combinator::{map, not, opt, value, verify},
        multi::{many0, many1, separated_list0, separated_list1},
        sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
        IResult,
    },
    std::{
        error::Error as StdError,
        fmt::{Debug, Display, Formatter, Result as FmtResult},
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
token!(pset_op, Pset);
token!(preset_op, Preset);
token!(tset_op, Tset);

// Reserved Words
token!(cbool_token, ConvertBoolean);
token!(cbyte_token, ConvertByte);
token!(cfloat_token, ConvertFloat);
token!(cint_token, ConvertInteger);
token!(cstr_token, ConvertString);
token!(abs_token, Abs);
token!(sin_token, Sin);
token!(call_token, Call);
token!(case_token, Case);
token!(cos_token, Cos);
token!(cls_token, ClearScreen);
token!(color_token, Color);
token!(dim_token, Dimension);
token!(do_token, Do);
token!(else_token, Else);
token!(end_token, End);
token!(exit_token, Exit);
token!(for_token, For);
token!(function_token, Function);
token!(get_token, Get);
token!(goto_token, Goto);
token!(if_token, If);
token!(is_token, Is);
token!(key_down_token, KeyDown);
token!(line_token, Line);
token!(locate_token, Locate);
token!(loop_token, Loop);
token!(mod_token, Mod);
token!(next_token, Next);
token!(palette_token, Palette);
token!(peek_token, Peek);
token!(poke_token, Poke);
token!(print_token, Print);
token!(put_token, Put);
token!(rect_token, Rectangle);
token!(rnd_token, Rnd);
token!(select_token, Select);
token!(step_token, Step);
token!(sub_token, Sub);
token!(then_token, Then);
token!(timer_token, Timer);
token!(to_token, To);
token!(until_token, Until);
token!(wend_token, Wend);
token!(while_token, While);
token!(yield_token, Yield);

// Punctuations
token!(colon_punc, Colon);
token!(comma_punc, Comma);
token!(end_of_line_punc, EndOfLine);
token!(l_paren_punc, LeftParenthesis);
token!(l_sq_bracket_punc, LeftSquareBracket);
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

fn parse_coord(tokens: Tokens) -> IResult<Tokens, (Expression, Expression)> {
    delimited(
        l_paren_punc,
        separated_pair(Expression::parse, comma_punc, Expression::parse),
        r_paren_punc,
    )(tokens)
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Exit<'a>(Span<'a>);

impl<'a> Exit<'a> {
    pub fn location(self) -> Span<'a> {
        self.0
    }
}

impl<'a> Debug for Exit<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        debug_location(f, self.0)
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Identifier<'a> {
    pub location: Span<'a>,
    pub name: &'a str,
}

impl<'a> Identifier<'a> {
    fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(map(id_token, |token| token.identifier().unwrap()), |name| {
            Self {
                location: tokens.location(),
                name,
            }
        })(tokens)
    }
}

impl<'a> Debug for Identifier<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!("Identifier `{}` ", self.name))?;

        debug_location(f, self.location)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Case<'a> {
    RangeFull(Expression<'a>, Expression<'a>),
    Relation(Relation<'a>, Expression<'a>),
}

impl<'a> Case<'a> {
    fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        alt((
            map(
                separated_pair(Expression::parse, to_token, Expression::parse),
                |(start, end)| Self::RangeFull(start, end),
            ),
            preceded(
                is_token,
                map(
                    pair(Relation::parse, Expression::parse),
                    |(relation, expr)| Self::Relation(relation, expr),
                ),
            ),
            map(Expression::parse, |expr: Expression<'_>| {
                Self::Relation(Relation::Equal(tokens.location()), expr)
            }),
        ))(tokens)
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Label<'a> {
    Name(Identifier<'a>),
    Number(u32, Span<'a>),
}

impl<'a> Label<'a> {
    pub fn is_name(self) -> bool {
        matches!(self, Self::Name(..))
    }

    pub fn is_number(self) -> bool {
        matches!(self, Self::Number(..))
    }

    pub fn location(self) -> Span<'a> {
        match self {
            Self::Name(Identifier { location, .. }) | Self::Number(_, location) => location,
        }
    }

    fn parse(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        alt((
            map(Identifier::parse, Self::Name),
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
            Self::Name(Identifier { name, .. }) => f.write_str(name),
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PutAction {
    And,
    Or,
    Pset,
    Preset,
    Tset,
    Xor,
}

impl PutAction {
    fn parse(tokens: Tokens) -> IResult<Tokens, Self> {
        alt((
            map(and_op, |_| Self::And),
            map(or_op, |_| Self::Or),
            map(pset_op, |_| Self::Pset),
            map(preset_op, |_| Self::Preset),
            map(tset_op, |_| Self::Tset),
            map(xor_op, |_| Self::Xor),
        ))(tokens)
    }
}

impl Display for PutAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(match self {
            Self::And => "AND",
            Self::Or => "OR",
            Self::Pset => "PSET",
            Self::Preset => "PRESET",
            Self::Tset => "TSET",
            Self::Xor => "XOR",
        })
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Syntax<'a> {
    Assign(Variable<'a>, Option<Vec<Expression<'a>>>, Expression<'a>),
    Call(Identifier<'a>, Vec<Expression<'a>>),
    Color(Expression<'a>, Option<Expression<'a>>),
    ClearScreen,
    Dimension(
        Vec<(
            Variable<'a>,
            Option<Vec<SubscriptRange<'a>>>,
            Option<Expression<'a>>,
        )>,
    ),
    End,
    ExitDo(Exit<'a>),
    ExitFor(Exit<'a>),
    ExitFunction(Exit<'a>),
    ExitSub(Exit<'a>),
    ExitWhile(Exit<'a>),
    For(
        Variable<'a>,
        Expression<'a>,
        Expression<'a>,
        Option<Expression<'a>>,
        Ast<'a>,
    ),
    Function(Variable<'a>, Vec<Variable<'a>>, Ast<'a>),
    Get(
        (Expression<'a>, Expression<'a>),
        (Expression<'a>, Expression<'a>),
        Variable<'a>,
        Option<Expression<'a>>,
    ),
    Goto(Label<'a>),
    If {
        tests: Vec<(Expression<'a>, Ast<'a>)>,
        default: Ast<'a>,
    },
    Label(Label<'a>),
    Line(
        Option<(Expression<'a>, Expression<'a>)>,
        (Expression<'a>, Expression<'a>),
        Expression<'a>,
    ),
    Locate(Expression<'a>, Option<Expression<'a>>),
    Loop {
        test: Option<(bool, bool, Expression<'a>)>,
        body_ast: Ast<'a>,
    },
    Palette(
        Expression<'a>,
        Expression<'a>,
        Expression<'a>,
        Expression<'a>,
    ),
    Poke(Expression<'a>, Expression<'a>),
    Print(Vec<Print<'a>>),
    Pset(Expression<'a>, Expression<'a>, Expression<'a>),
    Put(
        (Expression<'a>, Expression<'a>),
        (Expression<'a>, Expression<'a>),
        Variable<'a>,
        Option<Expression<'a>>,
        Option<PutAction>,
    ),
    Rectangle(
        Option<(Expression<'a>, Expression<'a>)>,
        (Expression<'a>, Expression<'a>),
        Expression<'a>,
        Option<Expression<'a>>,
    ),
    Select {
        test_expr: Expression<'a>,
        test_cases: Vec<(Vec<Case<'a>>, Ast<'a>)>,
        default_ast: Ast<'a>,
    },
    Sub(Identifier<'a>, Vec<Variable<'a>>, Ast<'a>),
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
                Self::parse_function,
                Self::parse_get,
                Self::parse_goto,
                Self::parse_if,
                Self::parse_line,
                Self::parse_locate,
                Self::parse_palette,
                Self::parse_poke,
                Self::parse_print,
                Self::parse_put,
                Self::parse_rect,
                Self::parse_sub,
                Self::parse_while,
                Self::parse_yield,
                Self::parse_label,
                alt((
                    Self::parse_call,
                    Self::parse_end,
                    Self::parse_exit,
                    Self::parse_loop,
                    Self::parse_pset,
                    Self::parse_select,
                )),
            )),
            many0(end_of_line_punc),
        ))(tokens)
    }

    fn parse_assign(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            terminated(
                tuple((
                    Variable::parse,
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

    fn parse_call(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            terminated(
                alt((
                    preceded(
                        call_token,
                        pair(
                            Identifier::parse,
                            map(
                                opt(delimited(
                                    l_paren_punc,
                                    separated_list0(comma_punc, Expression::parse),
                                    r_paren_punc,
                                )),
                                Option::unwrap_or_default,
                            ),
                        ),
                    ),
                    pair(
                        Identifier::parse,
                        separated_list0(comma_punc, Expression::parse),
                    ),
                )),
                opt(end_of_line_punc),
            ),
            |(sub, args)| Self::Call(sub, args),
        )(tokens)
    }

    fn parse_cls(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(cls_token, end_of_line_punc), |_| {
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
        fn subscript(tokens: Tokens) -> IResult<Tokens, SubscriptRange> {
            pair(
                opt(terminated(Expression::parse, to_token)),
                Expression::parse,
            )(tokens)
        }

        fn subscripts(tokens: Tokens) -> IResult<Tokens, Vec<SubscriptRange>> {
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
                        Variable::parse,
                        opt(subscripts),
                        opt(preceded(eq_op, Expression::parse)),
                    )),
                ),
                opt(end_of_line_punc),
            ),
            Self::Dimension,
        )(tokens)
    }

    fn parse_end(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(terminated(end_token, end_of_line_punc), |_| Self::End)(tokens)
    }

    fn parse_exit(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        preceded(
            exit_token,
            alt((
                map(do_token, |_| Self::ExitDo(Exit(tokens.location()))),
                map(for_token, |_| Self::ExitFor(Exit(tokens.location()))),
                map(function_token, |_| {
                    Self::ExitFunction(Exit(tokens.location()))
                }),
                map(sub_token, |_| Self::ExitSub(Exit(tokens.location()))),
                map(while_token, |_| Self::ExitWhile(Exit(tokens.location()))),
            )),
        )(tokens)
    }

    fn parse_for(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        let (tokens, (_, var, _, start, _, end, step, _)) = tuple((
            for_token,
            Variable::parse,
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
            opt(verify(Variable::parse, |next_var| {
                next_var.name == var.name && next_var.ty == var.ty
            })),
            opt(end_of_line_punc),
        ))(tokens)?;

        Ok((tokens, Self::For(var, start, end, step, ast)))
    }

    fn parse_function(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                function_token,
                separated_pair(
                    tuple((
                        Variable::parse,
                        map(
                            opt(delimited(
                                l_paren_punc,
                                separated_list0(comma_punc, Variable::parse),
                                r_paren_punc,
                            )),
                            Option::unwrap_or_default,
                        ),
                    )),
                    many1(end_of_line_punc),
                    Self::parse,
                ),
                tuple((end_token, function_token, opt(end_of_line_punc))),
            ),
            |((var, args), body)| Self::Function(var, args, body),
        )(tokens)
    }

    fn parse_get(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                get_token,
                tuple((
                    delimited(
                        l_paren_punc,
                        separated_pair(Expression::parse, comma_punc, Expression::parse),
                        r_paren_punc,
                    ),
                    preceded(
                        sub_op,
                        delimited(
                            l_paren_punc,
                            separated_pair(Expression::parse, comma_punc, Expression::parse),
                            r_paren_punc,
                        ),
                    ),
                    preceded(comma_punc, Variable::parse),
                    opt(delimited(l_paren_punc, Expression::parse, r_paren_punc)),
                )),
                opt(end_of_line_punc),
            ),
            |(from_exprs, to_exprs, var, var_index_expr)| {
                Self::Get(from_exprs, to_exprs, var, var_index_expr)
            },
        )(tokens)
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
                map(
                    opt(preceded(else_then, Self::parse)),
                    Option::unwrap_or_default,
                ),
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
            alt((
                verify(terminated(Label::parse, colon_punc), |label| {
                    label.is_name()
                }),
                verify(Label::parse, |label| label.is_number()),
            )),
            Self::Label,
        )(tokens)
    }

    fn parse_locate(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                locate_token,
                pair(
                    Expression::parse,
                    opt(preceded(comma_punc, Expression::parse)),
                ),
                end_of_line_punc,
            ),
            |(row, col)| Self::Locate(row, col),
        )(tokens)
    }

    fn parse_line(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            tuple((
                line_token,
                opt(terminated(parse_coord, sub_op)),
                parse_coord,
                comma_punc,
                Expression::parse,
            )),
            |(_, from_coord, to_coord, _, color)| Self::Line(from_coord, to_coord, color),
        )(tokens)
    }

    fn parse_loop(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        fn parse_until_or_while(tokens: Tokens) -> IResult<Tokens, (bool, Expression)> {
            pair(
                alt((map(until_token, |_| false), map(while_token, |_| true))),
                Expression::parse,
            )(tokens)
        }

        fn map_test_cond(
            test_before: bool,
            until_or_while: Option<(bool, Expression)>,
        ) -> Option<(bool, bool, Expression)> {
            until_or_while.map(|(test_while, test_expr)| (test_before, test_while, test_expr))
        }

        map(
            tuple((
                do_token,
                alt((
                    map(
                        tuple((
                            opt(parse_until_or_while),
                            preceded(end_of_line_punc, Self::parse),
                            loop_token,
                            not(parse_until_or_while),
                        )),
                        |(until_or_while, body_ast, ..)| {
                            (map_test_cond(true, until_or_while), body_ast)
                        },
                    ),
                    map(
                        tuple((
                            preceded(end_of_line_punc, Self::parse),
                            loop_token,
                            opt(parse_until_or_while),
                        )),
                        |(body_ast, _, until_or_while)| {
                            (map_test_cond(false, until_or_while), body_ast)
                        },
                    ),
                )),
                end_of_line_punc,
            )),
            |(_, (test, body_ast), ..)| Self::Loop { test, body_ast },
        )(tokens)
    }

    fn parse_palette(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                palette_token,
                tuple((
                    Expression::parse,
                    comma_punc,
                    Expression::parse,
                    comma_punc,
                    Expression::parse,
                    comma_punc,
                    Expression::parse,
                )),
                opt(end_of_line_punc),
            ),
            |(color_index, _, r, _, g, _, b)| Self::Palette(color_index, r, g, b),
        )(tokens)
    }

    fn parse_poke(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            preceded(
                poke_token,
                separated_pair(Expression::parse, comma_punc, Expression::parse),
            ),
            |(address, val)| Self::Poke(address, val),
        )(tokens)
    }

    fn parse_print(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(print_token, many0(Print::parse), opt(end_of_line_punc)),
            Self::Print,
        )(tokens)
    }

    fn parse_pset(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                pset_op,
                separated_pair(
                    delimited(
                        l_paren_punc,
                        separated_pair(Expression::parse, comma_punc, Expression::parse),
                        r_paren_punc,
                    ),
                    comma_punc,
                    Expression::parse,
                ),
                opt(end_of_line_punc),
            ),
            |((x, y), color)| Self::Pset(x, y, color),
        )(tokens)
    }

    fn parse_put(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                put_token,
                tuple((
                    delimited(
                        l_paren_punc,
                        separated_pair(Expression::parse, comma_punc, Expression::parse),
                        r_paren_punc,
                    ),
                    preceded(
                        comma_punc,
                        delimited(
                            l_paren_punc,
                            separated_pair(Expression::parse, comma_punc, Expression::parse),
                            r_paren_punc,
                        ),
                    ),
                    preceded(comma_punc, Variable::parse),
                    opt(delimited(l_paren_punc, Expression::parse, r_paren_punc)),
                    opt(preceded(comma_punc, PutAction::parse)),
                )),
                opt(end_of_line_punc),
            ),
            |(from_exprs, size_exprs, var, var_index_expr, action)| {
                Self::Put(from_exprs, size_exprs, var, var_index_expr, action)
            },
        )(tokens)
    }

    fn parse_rect(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            tuple((
                rect_token,
                opt(terminated(parse_coord, sub_op)),
                parse_coord,
                comma_punc,
                Expression::parse,
                opt(preceded(comma_punc, Expression::parse)),
            )),
            |(_, from_coord, to_coord, _, color, is_filled)| {
                Self::Rectangle(from_coord, to_coord, color, is_filled)
            },
        )(tokens)
    }

    fn parse_select(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                pair(select_token, case_token),
                separated_pair(
                    Expression::parse,
                    many1(end_of_line_punc),
                    pair(
                        many0(separated_pair(
                            preceded(case_token, separated_list1(comma_punc, Case::parse)),
                            many1(alt((colon_punc, end_of_line_punc))),
                            Self::parse,
                        )),
                        map(
                            opt(preceded(
                                tuple((
                                    case_token,
                                    else_token,
                                    many1(alt((colon_punc, end_of_line_punc))),
                                )),
                                Self::parse,
                            )),
                            Option::unwrap_or_default,
                        ),
                    ),
                ),
                tuple((end_token, select_token)),
            ),
            |(test_expr, (test_cases, default_ast))| Self::Select {
                test_expr,
                test_cases,
                default_ast,
            },
        )(tokens)
    }

    fn parse_sub(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            delimited(
                sub_token,
                separated_pair(
                    tuple((
                        Identifier::parse,
                        map(
                            opt(delimited(
                                l_paren_punc,
                                separated_list0(comma_punc, Variable::parse),
                                r_paren_punc,
                            )),
                            Option::unwrap_or_default,
                        ),
                    )),
                    many1(end_of_line_punc),
                    Self::parse,
                ),
                tuple((end_token, sub_token, opt(end_of_line_punc))),
            ),
            |((sub, args), body)| Self::Sub(sub, args, body),
        )(tokens)
    }

    fn parse_while(tokens: Tokens<'a>) -> IResult<Tokens<'a>, Self> {
        map(
            tuple((
                while_token,
                Expression::parse,
                end_of_line_punc,
                Self::parse,
                wend_token,
                end_of_line_punc,
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
            "line {}, column {}: {}\n{}\n{}^-- Here",
            self.line_number,
            self.column_number,
            self.reason,
            self.line,
            " ".repeat(self.column_number.max(1) - 1)
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
    pub fn is_numeric(self) -> bool {
        matches!(self, Type::Byte | Type::Float | Type::Integer)
    }

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

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Variable<'a> {
    pub location: Span<'a>,
    pub name: &'a str,
    pub ty: Option<Type>,
}

impl<'a> Variable<'a> {
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

impl<'a> Debug for Variable<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_fmt(format_args!("Variable `{}` ", self.name))?;

        if let Some(ty) = self.ty {
            <dyn Debug>::fmt(&ty, f)?;
            f.write_str(" ")?;
        }

        debug_location(f, self.location)
    }
}

#[cfg(test)]
mod tests {
    use {super::*, crate::tests::span};

    /// Asserts the AST defined by two syntaxes is equal; does not compare the original source code
    /// location of each tag.
    fn same_syntax(lhs: &Syntax, rhs: &Syntax) -> bool {
        fn compare_infix(lhs: Infix, rhs: Infix) -> bool {
            match lhs {
                Infix::Relation(lhs) => {
                    matches!(rhs, Infix::Relation(rhs) if compare_relation(lhs, rhs))
                }
                lhs @ _ => lhs == rhs,
            }
        }

        fn compare_relation(lhs: Relation, rhs: Relation) -> bool {
            match lhs {
                Relation::Equal(_) => matches!(rhs, Relation::Equal(_)),
                Relation::NotEqual(_) => matches!(rhs, Relation::NotEqual(_)),
                Relation::GreaterThanEqual(_) => matches!(rhs, Relation::GreaterThanEqual(_)),
                Relation::LessThanEqual(_) => matches!(rhs, Relation::LessThanEqual(_)),
                Relation::GreaterThan(_) => matches!(rhs, Relation::GreaterThan(_)),
                Relation::LessThan(_) => matches!(rhs, Relation::LessThan(_)),
            }
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
                    matches!(rhs, Expression::Variable(rhs) if compare_var(*lhs, *rhs))
                }
                Expression::Infix(lhs_op, lhs1, lhs2, _) => {
                    matches!(rhs, Expression::Infix(rhs_op, rhs1, rhs2, _) if compare_infix(*lhs_op, *rhs_op) && compare_expr(lhs1, rhs1) && compare_expr(lhs2, rhs2))
                }
                Expression::Prefix(lhs_op, lhs, _) => {
                    matches!(rhs, Expression::Prefix(rhs_op, rhs, _) if lhs_op == rhs_op && compare_expr(lhs, rhs))
                }
                Expression::Function(lhs_var, lhs, _) => {
                    matches!(rhs, Expression::Function(rhs_var, rhs, _) if compare_var(*lhs_var, *rhs_var) && compare_vec(lhs, rhs, compare_expr))
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
                Expression::Peek(lhs_ty, lhs, _) => {
                    matches!(rhs, Expression::Peek(rhs_ty, rhs, _) if compare_opt(lhs_ty, rhs_ty, Type::eq) && compare_expr(lhs, rhs))
                }
                Expression::KeyDown(lhs, _) => {
                    matches!(rhs, Expression::KeyDown(rhs, _) if compare_expr(lhs, rhs))
                }
                Expression::Random(_) => matches!(rhs, Expression::Random(_)),
                Expression::Timer(_) => matches!(rhs, Expression::Timer(_)),
            }
        }

        fn compare_subscript_range(lhs: &SubscriptRange, rhs: &SubscriptRange) -> bool {
            compare_opt(&lhs.0, &rhs.0, compare_expr) && compare_expr(&lhs.1, &rhs.1)
        }

        fn compare_dim(
            lhs: &(Variable, Option<Vec<SubscriptRange>>, Option<Expression>),
            rhs: &(Variable, Option<Vec<SubscriptRange>>, Option<Expression>),
        ) -> bool {
            compare_var(lhs.0, rhs.0)
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

        fn compare_var(lhs: Variable, rhs: Variable) -> bool {
            lhs.name == rhs.name && lhs.ty == rhs.ty
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
            Variable {
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
            Variable {
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
            Variable {
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
            Variable {
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
            Variable {
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
            Variable {
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
                Variable {
                    location: span(4, 1, input),
                    name: "var1",
                    ty: None,
                },
                None,
                Some(Expression::Literal(Literal::Integer(5, span(11, 1, input)))),
            ),
            (
                Variable {
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
            Variable {
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
