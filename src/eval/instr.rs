use {
    crate::{
        syntax::{
            Bitwise, Case, Exit, Expression, Identifier, Infix, Label, Literal, Prefix, Print,
            PutAction, Relation, Syntax, SyntaxError, Type, Variable,
        },
        token::{location_string, Token, Tokens},
    },
    log::{debug, error},
    nom::{error::Error, Err},
    std::{collections::HashMap, ops::Range},
};

pub type Address = usize;

#[derive(Clone)]
struct FunctionScope<'a> {
    ty: Type,
    args: &'a [Variable<'a>],
    global_vars: HashMap<&'a str, (Type, Address)>,
    body_ast: &'a [Syntax<'a>],
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Instruction {
    AddBytes(Address, Address, Address),
    AddFloats(Address, Address, Address),
    AddIntegers(Address, Address, Address),
    AddStrings(Address, Address, Address),

    SubtractBytes(Address, Address, Address),
    SubtractFloats(Address, Address, Address),
    SubtractIntegers(Address, Address, Address),

    MultiplyBytes(Address, Address, Address),
    MultiplyFloats(Address, Address, Address),
    MultiplyIntegers(Address, Address, Address),

    DivideBytes(Address, Address, Address),
    DivideFloats(Address, Address, Address),
    DivideIntegers(Address, Address, Address),

    ModulusBytes(Address, Address, Address),
    ModulusFloats(Address, Address, Address),
    ModulusIntegers(Address, Address, Address),

    NotBoolean(Address, Address),
    AndBooleans(Address, Address, Address),
    OrBooleans(Address, Address, Address),
    XorBooleans(Address, Address, Address),

    NotByte(Address, Address),
    AndBytes(Address, Address, Address),
    OrBytes(Address, Address, Address),
    XorBytes(Address, Address, Address),

    NotInteger(Address, Address),
    AndIntegers(Address, Address, Address),
    OrIntegers(Address, Address, Address),
    XorIntegers(Address, Address, Address),

    EqualBooleans(Address, Address, Address),
    EqualBytes(Address, Address, Address),
    EqualFloats(Address, Address, Address),
    EqualIntegers(Address, Address, Address),
    EqualStrings(Address, Address, Address),

    NotEqualBooleans(Address, Address, Address),
    NotEqualBytes(Address, Address, Address),
    NotEqualFloats(Address, Address, Address),
    NotEqualIntegers(Address, Address, Address),
    NotEqualStrings(Address, Address, Address),

    GreaterThanEqualBytes(Address, Address, Address),
    GreaterThanEqualFloats(Address, Address, Address),
    GreaterThanEqualIntegers(Address, Address, Address),

    GreaterThanBytes(Address, Address, Address),
    GreaterThanFloats(Address, Address, Address),
    GreaterThanIntegers(Address, Address, Address),

    ConvertBooleanToByte(Address, Address),
    ConvertBooleanToFloat(Address, Address),
    ConvertBooleanToInteger(Address, Address),
    ConvertBooleanToString(Address, Address),

    ConvertByteToBoolean(Address, Address),
    ConvertByteToFloat(Address, Address),
    ConvertByteToInteger(Address, Address),
    ConvertByteToString(Address, Address),

    ConvertFloatToBoolean(Address, Address),
    ConvertFloatToByte(Address, Address),
    ConvertFloatToInteger(Address, Address),
    ConvertFloatToString(Address, Address),

    ConvertIntegerToBoolean(Address, Address),
    ConvertIntegerToByte(Address, Address),
    ConvertIntegerToFloat(Address, Address),
    ConvertIntegerToString(Address, Address),

    ConvertStringToBoolean(Address, Address),
    ConvertStringToByte(Address, Address),
    ConvertStringToFloat(Address, Address),
    ConvertStringToInteger(Address, Address),

    AbsFloat(Address, Address),
    AbsInteger(Address, Address),
    Cos(Address, Address),
    Sin(Address, Address),
    ClearScreen,
    Color(Address, Address),
    KeyDown(Address, Address),
    Line(Address, Address, Address, Address, Address),
    Locate(Address, Address),
    Palette(Address, Address, Address, Address),
    PrintString(Address),
    Random(Address),
    Rectangle(Address, Address, Address, Address, Address, Address),
    SetPixel(Address, Address, Address),
    Timer(Address),

    GetGraphic(Address, Address, Address, Address, Address, Address),
    PutGraphicAnd(Address, Address, Address, Address, Address, Address),
    PutGraphicOr(Address, Address, Address, Address, Address, Address),
    PutGraphicPset(Address, Address, Address, Address, Address, Address),
    PutGraphicPreset(Address, Address, Address, Address, Address, Address),
    PutGraphicTset(Address, Address, Address, Address, Address, Address),
    PutGraphicXor(Address, Address, Address, Address, Address, Address),

    PeekBoolean(Address, Address),
    PeekByte(Address, Address),
    PeekFloat(Address, Address),
    PeekInteger(Address, Address),
    PeekString(Address, Address),

    PokeBoolean(Address, Address),
    PokeByte(Address, Address),
    PokeFloat(Address, Address),
    PokeInteger(Address, Address),
    PokeString(Address, Address),

    Branch(Address, usize),
    BranchNot(Address, usize),
    End,
    Jump(usize),
    Yield,

    Copy(Address, Address),
    ReadBooleans(Address, Box<[Address]>, Address),
    ReadBytes(Address, Box<[Address]>, Address),
    ReadFloats(Address, Box<[Address]>, Address),
    ReadIntegers(Address, Box<[Address]>, Address),
    ReadStrings(Address, Box<[Address]>, Address),
    WriteBooleans(Address, Box<[Address]>, Address),
    WriteBytes(Address, Box<[Address]>, Address),
    WriteFloats(Address, Box<[Address]>, Address),
    WriteIntegers(Address, Box<[Address]>, Address),
    WriteStrings(Address, Box<[Address]>, Address),

    DimensionBooleans(Box<[Range<Address>]>, Address),
    DimensionBytes(Box<[Range<Address>]>, Address),
    DimensionFloats(Box<[Range<Address>]>, Address),
    DimensionIntegers(Box<[Range<Address>]>, Address),
    DimensionStrings(Box<[Range<Address>]>, Address),

    WriteBoolean(bool, Address),
    WriteByte(u8, Address),
    WriteFloat(f32, Address),
    WriteInteger(i32, Address),
    WriteString(String, Address),
}

impl Instruction {
    pub fn compile(source_code: &[u8]) -> Result<Vec<Self>, SyntaxError> {
        // Lex the source code into tokens - this only checks for symbols being identifiable
        let (remaining_tokens, mut tokens) = match Token::lex(source_code) {
            Err(Err::Error(Error { input, code: _ }) | Err::Failure(Error { input, code: _ })) => {
                Err(SyntaxError::from_location(
                    input,
                    Token::ascii_str(input.get_line_beginning()).unwrap_or_default(),
                ))
            }
            Err(Err::Incomplete(_)) => unreachable!(),
            Ok(res) => Ok(res),
        }?;

        // Artificially add a final end-of-line in case the file ends on the last line (This makes
        // parsing simpler because we can always expect an end-of-line). We could do this in the
        // lexer but it is specific to this code path only and we don't want to test the artificial
        // data in those tests. This is tested in the integration tests. See headless.rs
        if let Some(last_token) = tokens.last() {
            if !matches!(last_token, Token::EndOfLine(_)) {
                tokens.push(Token::EndOfLine(last_token.location()));
            }
        }

        debug!("Tokens:");
        debug!("{:#?}", tokens);

        if !remaining_tokens.is_empty() {
            error!("Unparsed tokens");

            return Err(SyntaxError::from_location(
                remaining_tokens,
                "Unparsed tokens",
            ));
        }

        // Parse the tokens into syntax by applying the rules of grammar - syntax may be correct
        // however at this stage the program may not be valid
        let (remaining_syntax, syntax) = match Syntax::parse(Tokens::new(&tokens)) {
            Err(Err::Error(Error { input, code: _ }) | Err::Failure(Error { input, code: _ })) => {
                Err(SyntaxError::from_location(
                    input.location(),
                    Token::ascii_str(input.location().get_line_beginning()).unwrap_or_default(),
                ))
            }
            Err(Err::Incomplete(_)) => unreachable!(),
            Ok(res) => Ok(res),
        }?;

        debug!("Syntax:");
        debug!("{:#?}", syntax);

        if !remaining_syntax.is_empty() {
            error!("Unparsed syntax");

            return Err(SyntaxError::from_location(
                remaining_syntax.location(),
                "Unparsed syntax",
            ));
        }

        // Finally, compile the syntax into executable instructions
        let instrs = Self::compile_scope(
            &syntax,
            0,
            0,
            &mut HashMap::default(),
            &mut HashMap::default(),
            HashMap::default(),
        )?;

        let instrs = instrs
            .into_iter()
            .map(ScopeInstruction::instruction)
            .collect::<Result<Vec<Instruction>, SyntaxError>>()?;

        debug!("Instructions:");

        for (index, instr) in instrs.iter().enumerate() {
            debug!("{index} {:?}", instr);
        }

        Ok(instrs)
    }

    fn compile_expression<'a>(
        address: Address,
        expr: &Expression<'a>,
        program_offset: usize,
        program: &mut Vec<ScopeInstruction<'a>>,
        fns: &HashMap<&'a str, FunctionScope<'a>>,
        subs: &HashMap<&'a str, SubScope<'a>>,
        vars: &HashMap<&str, (Type, Address)>,
    ) -> Result<(Type, Address), SyntaxError> {
        Ok(match expr {
            Expression::Literal(lit) => {
                let (ty, instr) = match *lit {
                    Literal::Boolean(val, _) => (Type::Boolean, Self::WriteBoolean(val, address)),
                    Literal::Byte(val, _) => (Type::Byte, Self::WriteByte(val, address)),
                    Literal::Float(val, _) => (Type::Float, Self::WriteFloat(val, address)),
                    Literal::Integer(val, _) => (Type::Integer, Self::WriteInteger(val, address)),
                    Literal::String(val, _) => {
                        (Type::String, Self::WriteString(val.to_owned(), address))
                    }
                };

                program.push(instr.into());

                (ty, address)
            }
            Expression::Tuple(_exprs, _) => {
                todo!();
            }
            Expression::Variable(var) => vars.get(var.name).copied().ok_or_else(|| {
                SyntaxError::from_location(
                    var.location,
                    format!("Undefined variable `{}`", var.name),
                )
            })?,
            Expression::Function(var, arg_exprs, location) => {
                if let Some((var_ty, var_address)) = vars.get(var.name).copied() {
                    let mut index_address = address + 1;

                    // This is an array access - all args must be integers
                    let index_addresses = arg_exprs
                        .iter()
                        .map(|index_expr| {
                            let (index_expr_ty, index_expr_address) = Self::compile_expression(
                                index_address,
                                index_expr,
                                program_offset,
                                program,
                                fns,
                                subs,
                                vars,
                            )?;

                            assert!(index_expr_address <= index_address);

                            if index_expr_ty != Type::Integer {
                                return Err(SyntaxError::from_location(
                                    index_expr.location(),
                                    "Index type must be integer",
                                ));
                            }

                            if index_expr_address == index_address {
                                index_address += 1;
                            }

                            Ok(index_expr_address)
                        })
                        .collect::<Result<Box<_>, SyntaxError>>()?;

                    match var_ty {
                        Type::Boolean => {
                            program.push(
                                Self::ReadBooleans(var_address, index_addresses, address).into(),
                            );
                        }
                        Type::Byte => {
                            program.push(
                                Self::ReadBytes(var_address, index_addresses, address).into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::ReadFloats(var_address, index_addresses, address).into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::ReadIntegers(var_address, index_addresses, address).into(),
                            );
                        }
                        Type::String => {
                            program.push(
                                Self::ReadStrings(var_address, index_addresses, address).into(),
                            );
                        }
                    }

                    (var_ty, address)
                } else if let Some(scope) = fns.get(var.name) {
                    if arg_exprs.len() != scope.args.len() {
                        return Err(SyntaxError::from_location(
                            *location,
                            format!(
                                "`{}` expects {} arguments but {} were provided.",
                                var.name,
                                scope.args.len(),
                                arg_exprs.len()
                            ),
                        ));
                    }

                    program.push(
                        match scope.ty {
                            Type::Boolean => Self::WriteBoolean(false, address),
                            Type::Byte => Self::WriteByte(0, address),
                            Type::Float => Self::WriteFloat(0.0, address),
                            Type::Integer => Self::WriteInteger(0, address),
                            Type::String => Self::WriteString("".to_string(), address),
                        }
                        .into(),
                    );

                    let mut fns = fns.clone();
                    fns.remove(var.name);

                    let mut local_vars = scope.global_vars.clone();
                    local_vars.insert(var.name, (scope.ty, address));

                    let mut next_address = address + 1;
                    for (arg, expr) in scope.args.iter().zip(arg_exprs) {
                        assert!(arg.ty.is_some());

                        let arg_ty = arg.ty.unwrap();
                        let (expr_ty, expr_address) = Self::compile_expression(
                            next_address,
                            expr,
                            program_offset,
                            program,
                            &fns,
                            subs,
                            vars,
                        )?;

                        assert!(expr_address <= next_address);

                        if arg_ty != expr_ty {
                            return Err(SyntaxError::from_location(
                                    arg.location,
                                    format!("`{}` expects a {} expression for argument `{}` but a {} was provided.", var.name, arg_ty, arg.name, expr_ty),
                                ));
                        }

                        local_vars.insert(arg.name, (expr_ty, expr_address));

                        if expr_address == next_address {
                            next_address += 1;
                        }
                    }

                    let mut subs = subs.clone();
                    let mut body = Self::compile_scope(
                        scope.body_ast,
                        program_offset + program.len(),
                        next_address,
                        &mut fns,
                        &mut subs,
                        local_vars,
                    )?;
                    let body_len = body.len();

                    for instr in &mut body {
                        if matches!(instr, ScopeInstruction::ExitFunction(_)) {
                            *instr = Self::Jump(program_offset + program.len() + body_len).into();
                        }
                    }

                    program.append(&mut body);

                    (scope.ty, address)
                } else {
                    return Err(SyntaxError::from_location(
                        *location,
                        format!("`{}` is undefined.", var.name),
                    ));
                }
            }
            Expression::ConvertBoolean(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                let address = match expr_ty {
                    Type::Boolean => expr_address,
                    Type::Byte => {
                        program.push(Self::ConvertByteToBoolean(expr_address, address).into());
                        address
                    }
                    Type::Float => {
                        program.push(Self::ConvertFloatToBoolean(expr_address, address).into());
                        address
                    }
                    Type::Integer => {
                        program.push(Self::ConvertIntegerToBoolean(expr_address, address).into());
                        address
                    }
                    Type::String => {
                        program.push(Self::ConvertStringToBoolean(expr_address, address).into());
                        address
                    }
                };

                (Type::Boolean, address)
            }
            Expression::ConvertByte(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                let address = match expr_ty {
                    Type::Boolean => {
                        program.push(Self::ConvertBooleanToByte(expr_address, address).into());
                        address
                    }
                    Type::Byte => expr_address,
                    Type::Float => {
                        program.push(Self::ConvertFloatToByte(expr_address, address).into());
                        address
                    }
                    Type::Integer => {
                        program.push(Self::ConvertIntegerToByte(expr_address, address).into());
                        address
                    }
                    Type::String => {
                        program.push(Self::ConvertStringToByte(expr_address, address).into());
                        address
                    }
                };

                (Type::Byte, address)
            }
            Expression::ConvertFloat(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                let address = match expr_ty {
                    Type::Boolean => {
                        program.push(Self::ConvertBooleanToFloat(expr_address, address).into());
                        address
                    }
                    Type::Byte => {
                        program.push(Self::ConvertByteToFloat(expr_address, address).into());
                        address
                    }
                    Type::Float => expr_address,
                    Type::Integer => {
                        program.push(Self::ConvertIntegerToFloat(expr_address, address).into());
                        address
                    }
                    Type::String => {
                        program.push(Self::ConvertStringToFloat(expr_address, address).into());
                        address
                    }
                };

                (Type::Float, address)
            }
            Expression::ConvertInteger(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                let address = match expr_ty {
                    Type::Boolean => {
                        program.push(Self::ConvertBooleanToInteger(expr_address, address).into());
                        address
                    }
                    Type::Byte => {
                        program.push(Self::ConvertByteToInteger(expr_address, address).into());
                        address
                    }
                    Type::Float => {
                        program.push(Self::ConvertFloatToInteger(expr_address, address).into());
                        address
                    }
                    Type::Integer => expr_address,
                    Type::String => {
                        program.push(Self::ConvertStringToInteger(expr_address, address).into());
                        address
                    }
                };

                (Type::Integer, address)
            }
            Expression::ConvertString(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                let address = match expr_ty {
                    Type::Boolean => {
                        program.push(Self::ConvertBooleanToString(expr_address, address).into());
                        address
                    }
                    Type::Byte => {
                        program.push(Self::ConvertByteToString(expr_address, address).into());
                        address
                    }
                    Type::Float => {
                        program.push(Self::ConvertFloatToString(expr_address, address).into());
                        address
                    }
                    Type::Integer => {
                        program.push(Self::ConvertIntegerToString(expr_address, address).into());
                        address
                    }
                    Type::String => expr_address,
                };

                (Type::String, address)
            }
            Expression::Abs(ty, expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                if !matches!(expr_ty, Type::Float | Type::Integer) {
                    return Err(SyntaxError::from_location(
                        expr.location(),
                        format!("{expr_ty} should be a signed number"),
                    ));
                }

                if let &Some(ty) = ty {
                    if ty != expr_ty {
                        return Err(SyntaxError::from_location(
                            expr.location(),
                            format!("Cannot return a {ty} using a {expr_ty} expressions"),
                        ));
                    }
                }

                program.push(match expr_ty {
                    Type::Float => Self::AbsFloat(expr_address, address).into(),
                    Type::Integer => Self::AbsInteger(expr_address, address).into(),
                    _ => unreachable!(),
                });

                (expr_ty, address)
            }
            Expression::Cos(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                if expr_ty != Type::Float {
                    return Err(SyntaxError::from_location(
                        expr.location(),
                        format!("{expr_ty} should be Float"),
                    ));
                }

                program.push(Self::Cos(expr_address, address).into());

                (Type::Float, address)
            }
            Expression::Sin(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                if expr_ty != Type::Float {
                    return Err(SyntaxError::from_location(
                        expr.location(),
                        format!("{expr_ty} should be Float"),
                    ));
                }

                program.push(Self::Sin(expr_address, address).into());

                (Type::Float, address)
            }
            Expression::Peek(ty, expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                if expr_ty != Type::Integer {
                    return Err(SyntaxError::from_location(
                        expr.location(),
                        format!("{expr_ty} should be Integer"),
                    ));
                }

                let ty = ty.unwrap_or(Type::Byte);

                program.push(
                    match ty {
                        Type::Boolean => Self::PeekBoolean(expr_address, address),
                        Type::Byte => Self::PeekByte(expr_address, address),
                        Type::Float => Self::PeekFloat(expr_address, address),
                        Type::Integer => Self::PeekInteger(expr_address, address),
                        Type::String => Self::PeekString(expr_address, address),
                    }
                    .into(),
                );

                (ty, address)
            }
            Expression::KeyDown(expr, _) => {
                let (expr_ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                if expr_ty != Type::Byte {
                    return Err(SyntaxError::from_location(
                        expr.location(),
                        format!("{expr_ty} should be Byte"),
                    ));
                }

                program.push(Self::KeyDown(expr_address, address).into());

                (Type::Boolean, address)
            }
            Expression::Random(_) => {
                program.push(Self::Random(address).into());
                (Type::Float, address)
            }
            Expression::Timer(_) => {
                program.push(Self::Timer(address).into());
                (Type::Integer, address)
            }
            Expression::Prefix(prefix, expr, _) => {
                let (ty, expr_address) = Self::compile_expression(
                    address,
                    expr,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(expr_address <= address);

                match prefix {
                    Prefix::Minus => {
                        match ty {
                            Type::Float => {
                                program.push(Self::WriteFloat(-1.0, address).into());
                                program.push(
                                    Self::MultiplyFloats(address, expr_address, address).into(),
                                );
                            }
                            Type::Integer => {
                                program.push(Self::WriteInteger(-1, address).into());
                                program.push(
                                    Self::MultiplyIntegers(address, expr_address, address).into(),
                                );
                            }
                            ty => {
                                return Err(SyntaxError::from_location(
                                    expr.location(),
                                    format!("Invalid type {ty}: must be signed number"),
                                ));
                            }
                        }

                        (ty, address)
                    }
                    Prefix::Negate => {
                        match ty {
                            Type::Boolean => {
                                program.push(Self::NotBoolean(expr_address, address).into());
                            }
                            Type::Byte => {
                                program.push(Self::NotByte(expr_address, address).into());
                            }
                            Type::Integer => {
                                program.push(Self::NotInteger(expr_address, address).into());
                            }
                            ty => {
                                return Err(SyntaxError::from_location(
                                    expr.location(),
                                    format!("Invalid type {ty}: must be fixed point number"),
                                ));
                            }
                        }

                        (ty, address)
                    }
                }
            }
            Expression::Infix(infix, lhs, rhs, _) => {
                let (lhs_expr_ty, lhs_expr_address) = Self::compile_expression(
                    address,
                    lhs,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(lhs_expr_address <= address);

                let rhs_address = if lhs_expr_address == address {
                    address + 1
                } else {
                    address
                };

                let (rhs_expr_ty, rhs_expr_address) = Self::compile_expression(
                    rhs_address,
                    rhs,
                    program_offset,
                    program,
                    fns,
                    subs,
                    vars,
                )?;

                assert!(rhs_expr_address <= rhs_address);

                if lhs_expr_ty != rhs_expr_ty {
                    return Err(SyntaxError::from_location(
                        expr.location(),
                        format!("{lhs_expr_ty} incompatible with {rhs_expr_ty}"),
                    ));
                }

                match infix {
                    Infix::Add => match lhs_expr_ty {
                        Type::Byte => {
                            program.push(
                                Self::AddBytes(lhs_expr_address, rhs_expr_address, address).into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::AddFloats(lhs_expr_address, rhs_expr_address, address).into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::AddIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::String => {
                            program.push(
                                Self::AddStrings(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot add {ty} values"),
                            ));
                        }
                    },
                    Infix::Subtract => match lhs_expr_ty {
                        Type::Byte => {
                            program.push(
                                Self::SubtractBytes(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::SubtractFloats(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::SubtractIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot subtract {ty} values"),
                            ));
                        }
                    },
                    Infix::Divide => match lhs_expr_ty {
                        Type::Byte => {
                            program.push(
                                Self::DivideBytes(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::DivideFloats(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::DivideIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot divide {ty} values"),
                            ));
                        }
                    },
                    Infix::Multiply => match lhs_expr_ty {
                        Type::Byte => {
                            program.push(
                                Self::MultiplyBytes(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::MultiplyFloats(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::MultiplyIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot multiply {ty} values"),
                            ));
                        }
                    },
                    Infix::Modulus => {
                        program.push(
                            match lhs_expr_ty {
                                Type::Byte => Self::ModulusBytes,
                                Type::Float => Self::ModulusFloats,
                                Type::Integer => Self::ModulusIntegers,
                                ty => {
                                    return Err(SyntaxError::from_location(
                                        expr.location(),
                                        format!("Cannot modulus {ty} values"),
                                    ));
                                }
                            }(
                                lhs_expr_address, rhs_expr_address, address
                            )
                            .into(),
                        );
                    }
                    Infix::Bitwise(Bitwise::And) => match lhs_expr_ty {
                        Type::Boolean => {
                            program.push(
                                Self::AndBooleans(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Byte => {
                            program.push(
                                Self::AndBytes(lhs_expr_address, rhs_expr_address, address).into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::AndIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot AND {ty} values"),
                            ));
                        }
                    },
                    Infix::Bitwise(Bitwise::Not) => match lhs_expr_ty {
                        Type::Boolean => {
                            // Leaves out rhs to match qbasic
                            program.push(Self::NotBoolean(lhs_expr_address, address).into());
                        }
                        Type::Byte => {
                            // Leaves out rhs to match qbasic
                            program.push(Self::NotByte(lhs_expr_address, address).into());
                        }
                        Type::Integer => {
                            // Leaves out rhs to match qbasic
                            program.push(Self::NotInteger(lhs_expr_address, address).into());
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot AND {ty} values"),
                            ));
                        }
                    },
                    Infix::Bitwise(Bitwise::Or) => match lhs_expr_ty {
                        Type::Boolean => {
                            program.push(
                                Self::OrBooleans(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Byte => {
                            program.push(
                                Self::OrBytes(lhs_expr_address, rhs_expr_address, address).into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::OrIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot OR {ty} values"),
                            ));
                        }
                    },
                    Infix::Bitwise(Bitwise::Xor) => match lhs_expr_ty {
                        Type::Boolean => {
                            program.push(
                                Self::XorBooleans(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Byte => {
                            program.push(
                                Self::XorBytes(lhs_expr_address, rhs_expr_address, address).into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::XorIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot XOR {ty} values"),
                            ));
                        }
                    },
                    Infix::Relation(Relation::Equal(_)) => match lhs_expr_ty {
                        Type::Boolean => {
                            program.push(
                                Self::EqualBooleans(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Byte => {
                            program.push(
                                Self::EqualBytes(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::EqualFloats(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::EqualIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::String => {
                            program.push(
                                Self::EqualStrings(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                    },
                    Infix::Relation(Relation::NotEqual(_)) => match lhs_expr_ty {
                        Type::Boolean => {
                            program.push(
                                Self::NotEqualBooleans(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Byte => {
                            program.push(
                                Self::NotEqualBytes(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::NotEqualFloats(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::NotEqualIntegers(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::String => {
                            program.push(
                                Self::NotEqualStrings(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                    },
                    Infix::Relation(Relation::GreaterThanEqual(_)) => match lhs_expr_ty {
                        Type::Byte => {
                            program.push(
                                Self::GreaterThanEqualBytes(
                                    lhs_expr_address,
                                    rhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::GreaterThanEqualFloats(
                                    lhs_expr_address,
                                    rhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::GreaterThanEqualIntegers(
                                    lhs_expr_address,
                                    rhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot compare {ty} values using greater-than-equal"),
                            ));
                        }
                    },
                    Infix::Relation(Relation::GreaterThan(_)) => match lhs_expr_ty {
                        Type::Byte => {
                            program.push(
                                Self::GreaterThanBytes(lhs_expr_address, rhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Float => {
                            program.push(
                                Self::GreaterThanFloats(
                                    lhs_expr_address,
                                    rhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        Type::Integer => {
                            program.push(
                                Self::GreaterThanIntegers(
                                    lhs_expr_address,
                                    rhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot compare {ty} values using greater-than-equal"),
                            ));
                        }
                    },
                    Infix::Relation(Relation::LessThanEqual(_)) => match lhs_expr_ty {
                        Type::Byte => {
                            // Notice lhs and rhs switched!
                            program.push(
                                Self::GreaterThanEqualBytes(
                                    rhs_expr_address,
                                    lhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        Type::Float => {
                            // Notice lhs and rhs switched!
                            program.push(
                                Self::GreaterThanEqualFloats(
                                    rhs_expr_address,
                                    lhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        Type::Integer => {
                            // Notice lhs and rhs switched!
                            program.push(
                                Self::GreaterThanEqualIntegers(
                                    rhs_expr_address,
                                    lhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot compare {ty} values using less-than-equal"),
                            ));
                        }
                    },
                    Infix::Relation(Relation::LessThan(_)) => match lhs_expr_ty {
                        Type::Byte => {
                            // Notice lhs and rhs switched!
                            program.push(
                                Self::GreaterThanBytes(rhs_expr_address, lhs_expr_address, address)
                                    .into(),
                            );
                        }
                        Type::Float => {
                            // Notice lhs and rhs switched!
                            program.push(
                                Self::GreaterThanFloats(
                                    rhs_expr_address,
                                    lhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        Type::Integer => {
                            // Notice lhs and rhs switched!
                            program.push(
                                Self::GreaterThanIntegers(
                                    rhs_expr_address,
                                    lhs_expr_address,
                                    address,
                                )
                                .into(),
                            );
                        }
                        ty => {
                            return Err(SyntaxError::from_location(
                                expr.location(),
                                format!("Cannot compare {ty} values using less-than"),
                            ));
                        }
                    },
                }

                let ty = if let Infix::Relation(_) = infix {
                    Type::Boolean
                } else {
                    lhs_expr_ty
                };

                (ty, address)
            }
        })
    }

    fn compile_scope<'a>(
        syntax: &'a [Syntax],
        program_offset: usize,
        mut address: Address,
        fns: &mut HashMap<&'a str, FunctionScope<'a>>,
        subs: &mut HashMap<&'a str, SubScope<'a>>,
        mut vars: HashMap<&'a str, (Type, Address)>,
    ) -> Result<Vec<ScopeInstruction<'a>>, SyntaxError> {
        #[derive(Clone, Copy, Eq, Hash, PartialEq)]
        pub enum UnlocatedLabel<'a> {
            Name(&'a str),
            Number(u32),
        }

        impl<'a> From<Label<'a>> for UnlocatedLabel<'a> {
            fn from(label: Label<'a>) -> Self {
                match label {
                    Label::Name(Identifier { name, .. }) => UnlocatedLabel::Name(name),
                    Label::Number(number, _) => UnlocatedLabel::Number(number),
                }
            }
        }

        let mut program = vec![];

        let mut labels = HashMap::new();

        for syntax in syntax {
            match syntax {
                Syntax::Assign(var, index_exprs, expr) => {
                    let (expr_ty, expr_address) = Self::compile_expression(
                        address,
                        expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(
                        expr_address <= address,
                        "Invalid allocation: {}",
                        location_string(expr.location())
                    );

                    if let Some(var_ty) = var.ty {
                        if expr_ty != var_ty {
                            return Err(SyntaxError::from_location(
                                var.location,
                                format!(
                                    "variable `{}{}` ({}) cannot store value of {} expression",
                                    var.name,
                                    var_ty.symbol(),
                                    var_ty.to_string().to_lowercase(),
                                    expr_ty.to_string().to_lowercase()
                                ),
                            ));
                        }
                    }

                    let (var_ty, var_address) =
                        *vars.entry(var.name).or_insert_with(|| (expr_ty, address));

                    if var_ty != expr_ty {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!(
                                "variable `{}{}` ({}) previously assigned {} type",
                                var.name,
                                var_ty.symbol(),
                                var_ty.to_string().to_lowercase(),
                                expr_ty.to_string().to_lowercase()
                            ),
                        ));
                    }

                    if expr_address == address {
                        address += 1;
                    }

                    if let Some(index_exprs) = index_exprs {
                        let mut index_address = address;
                        let index_addresses = index_exprs
                            .iter()
                            .map(|index_expr| {
                                let (index_expr_ty, index_expr_address) = Self::compile_expression(
                                    index_address,
                                    index_expr,
                                    program_offset,
                                    &mut program,
                                    fns,
                                    subs,
                                    &vars,
                                )?;

                                assert!(index_expr_address <= index_address);

                                if index_expr_ty != Type::Integer {
                                    return Err(SyntaxError::from_location(
                                        index_expr.location(),
                                        "Index type must be integer",
                                    ));
                                }

                                if index_expr_address == index_address {
                                    index_address += 1;
                                }

                                Ok(index_expr_address)
                            })
                            .collect::<Result<Box<_>, SyntaxError>>()?;

                        program.push(
                            match var_ty {
                                Type::Boolean => Instruction::WriteBooleans(
                                    expr_address,
                                    index_addresses,
                                    var_address,
                                ),
                                Type::Byte => Instruction::WriteBytes(
                                    expr_address,
                                    index_addresses,
                                    var_address,
                                ),
                                Type::Float => Instruction::WriteFloats(
                                    expr_address,
                                    index_addresses,
                                    var_address,
                                ),
                                Type::Integer => Instruction::WriteIntegers(
                                    expr_address,
                                    index_addresses,
                                    var_address,
                                ),
                                Type::String => Instruction::WriteStrings(
                                    expr_address,
                                    index_addresses,
                                    var_address,
                                ),
                            }
                            .into(),
                        );
                    } else if var_address != expr_address {
                        program.push(Instruction::Copy(expr_address, var_address).into());
                    }
                }
                Syntax::ClearScreen => {
                    program.push(Instruction::ClearScreen.into());
                }
                Syntax::Call(sub, arg_exprs) => {
                    let scope = subs.get(sub.name).ok_or(SyntaxError::from_location(
                        sub.location,
                        format!("`{}` is undefined.", sub.name),
                    ))?;

                    if arg_exprs.len() != scope.args.len() {
                        return Err(SyntaxError::from_location(
                            sub.location,
                            format!(
                                "`{}` expects {} arguments but {} were provided.",
                                sub.name,
                                scope.args.len(),
                                arg_exprs.len()
                            ),
                        ));
                    }

                    let mut subs = subs.clone();
                    subs.remove(sub.name);

                    let mut address = address;
                    let mut local_vars = scope.global_vars.clone();
                    for (arg, expr) in scope.args.iter().zip(arg_exprs) {
                        assert!(arg.ty.is_some());

                        let arg_ty = arg.ty.unwrap();
                        let (expr_ty, expr_address) = Self::compile_expression(
                            address,
                            expr,
                            program_offset,
                            &mut program,
                            fns,
                            &subs,
                            &vars,
                        )?;

                        assert!(expr_address <= address);

                        if arg_ty != expr_ty {
                            return Err(SyntaxError::from_location(
                                    arg.location,
                                    format!("`{}` expects a {} expression for argument `{}` but a {} was provided.", sub.name, arg_ty, arg.name, expr_ty),
                                ));
                        }

                        local_vars.insert(arg.name, (expr_ty, expr_address));

                        if expr_address == address {
                            address += 1;
                        }
                    }

                    let body_offset = program_offset + program.len();
                    let mut body = Self::compile_scope(
                        scope.body_ast,
                        body_offset,
                        address,
                        fns,
                        &mut subs,
                        local_vars,
                    )?;
                    let body_len = body.len();

                    for instr in &mut body {
                        if matches!(instr, ScopeInstruction::ExitSub(_)) {
                            *instr = Self::Jump(body_offset + body_len).into();
                        }
                    }

                    program.append(&mut body);
                }
                Syntax::Color(foreground_expr, background_expr) => {
                    let (foreground_expr_ty, mut foreground_expr_address) =
                        Self::compile_expression(
                            address,
                            foreground_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                    assert!(foreground_expr_address <= address);

                    match foreground_expr_ty {
                        Type::Integer => {
                            if foreground_expr_address != address {
                                program.push(
                                    Instruction::Copy(foreground_expr_address, address).into(),
                                );
                                foreground_expr_address = address;
                            }

                            program
                                .push(Instruction::ConvertIntegerToByte(address, address).into());
                        }
                        ty if ty != Type::Byte => {
                            return Err(SyntaxError::from_location(
                                foreground_expr.location(),
                                format!(
                                    "Expected byte or integer, found {}",
                                    ty.to_string().to_lowercase()
                                ),
                            ));
                        }
                        _ => (),
                    }

                    let mut background_address = if foreground_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    if let Some(background_expr) = background_expr {
                        let (background_expr_ty, background_expr_address) =
                            Self::compile_expression(
                                background_address,
                                background_expr,
                                program_offset,
                                &mut program,
                                fns,
                                subs,
                                &vars,
                            )?;

                        assert!(background_expr_address <= background_address);

                        match background_expr_ty {
                            Type::Integer => {
                                if background_expr_address != background_address {
                                    program.push(
                                        Instruction::Copy(
                                            background_expr_address,
                                            background_address,
                                        )
                                        .into(),
                                    );
                                }

                                program.push(
                                    Instruction::ConvertIntegerToByte(
                                        background_address,
                                        background_address,
                                    )
                                    .into(),
                                );
                            }
                            ty if ty != Type::Byte => {
                                return Err(SyntaxError::from_location(
                                    foreground_expr.location(),
                                    format!(
                                        "Expected byte or integer, found {}",
                                        ty.to_string().to_lowercase()
                                    ),
                                ));
                            }
                            _ => background_address = background_expr_address,
                        }
                    } else {
                        program.push(Instruction::WriteByte(0xFF, background_address).into());
                    }

                    program.push(
                        Instruction::Color(foreground_expr_address, background_address).into(),
                    );
                }
                Syntax::Dimension(dims) => {
                    for (dim, subscripts, expr) in dims {
                        let subscipt_addresses = subscripts
                            .as_ref()
                            .map(|subscripts| {
                                subscripts
                                    .iter()
                                    .map(|(start_expr, end_expr)| {
                                        let start_expr_address =
                                            if let Some(start_expr) = start_expr {
                                                let (start_expr_ty, start_expr_address) =
                                                    Self::compile_expression(
                                                        address,
                                                        start_expr,
                                                        program_offset,
                                                        &mut program,
                                                        fns,
                                                        subs,
                                                        &vars,
                                                    )?;

                                                assert!(start_expr_address <= address);

                                                if start_expr_ty != Type::Integer {
                                                    return Err(SyntaxError::from_location(
                                                        start_expr.location(),
                                                        "Subscript type must be integer",
                                                    ));
                                                }

                                                start_expr_address
                                            } else {
                                                program.push(Self::WriteInteger(0, address).into());
                                                address
                                            };

                                        if start_expr_address == address {
                                            address += 1;
                                        }

                                        let (end_expr_ty, end_expr_address) =
                                            Self::compile_expression(
                                                address,
                                                end_expr,
                                                program_offset,
                                                &mut program,
                                                fns,
                                                subs,
                                                &vars,
                                            )?;

                                        assert!(end_expr_address <= address);

                                        if end_expr_ty != Type::Integer {
                                            return Err(SyntaxError::from_location(
                                                end_expr.location(),
                                                "Subscript type must be integer",
                                            ));
                                        }

                                        if end_expr_address == address {
                                            address += 1;
                                        }

                                        Ok(start_expr_address..end_expr_address)
                                    })
                                    .collect::<Result<Box<_>, SyntaxError>>()
                            })
                            .transpose()?;

                        let ty = if let Some(expr) = expr {
                            let (expr_ty, expr_address) = Self::compile_expression(
                                address,
                                expr,
                                program_offset,
                                &mut program,
                                fns,
                                subs,
                                &vars,
                            )?;

                            assert!(expr_address <= address);

                            if expr_address < address {
                                program.push(Instruction::Copy(expr_address, address).into());
                            }

                            if let Some(dim_ty) = dim.ty {
                                if dim_ty != expr_ty {
                                    return Err(SyntaxError::from_location(dim.location, format!("Invalid dimension; variable is {dim_ty} but expression is {expr_ty}.")));
                                }
                            }

                            expr_ty
                        } else {
                            dim.ty.unwrap_or(Type::Integer)
                        };

                        if expr.is_none() {
                            if let Some(subscript_addresses) = subscipt_addresses {
                                program.push(
                                    match ty {
                                        Type::Boolean => {
                                            Self::DimensionBooleans(subscript_addresses, address)
                                        }
                                        Type::Byte => {
                                            Self::DimensionBytes(subscript_addresses, address)
                                        }
                                        Type::Float => {
                                            Self::DimensionFloats(subscript_addresses, address)
                                        }
                                        Type::Integer => {
                                            Self::DimensionIntegers(subscript_addresses, address)
                                        }
                                        Type::String => {
                                            Self::DimensionStrings(subscript_addresses, address)
                                        }
                                    }
                                    .into(),
                                )
                            } else {
                                program.push(
                                    match ty {
                                        Type::Boolean => Self::WriteBoolean(false, address),
                                        Type::Byte => Self::WriteByte(0, address),
                                        Type::Float => Self::WriteFloat(0.0, address),
                                        Type::Integer => Self::WriteInteger(0, address),
                                        Type::String => Self::WriteString("".to_owned(), address),
                                    }
                                    .into(),
                                );
                            }
                        }

                        if vars.insert(dim.name, (ty, address)).is_some() {
                            return Err(SyntaxError::from_location(
                                dim.location,
                                format!("Cannot redefine variable `{}`.", dim.name),
                            ));
                        }

                        address += 1;
                    }
                }
                Syntax::End => program.push(Self::End.into()),
                &Syntax::ExitDo(exit) => program.push(ScopeInstruction::ExitDo(exit)),
                &Syntax::ExitFor(exit) => program.push(ScopeInstruction::ExitFor(exit)),
                &Syntax::ExitFunction(exit) => program.push(ScopeInstruction::ExitFunction(exit)),
                &Syntax::ExitSub(exit) => program.push(ScopeInstruction::ExitSub(exit)),
                &Syntax::ExitWhile(exit) => program.push(ScopeInstruction::ExitWhile(exit)),
                Syntax::For(var, start_expr, end_expr, step_expr, body_ast) => {
                    let mut address = address;
                    let (start_expr_ty, start_expr_address) = Self::compile_expression(
                        address,
                        start_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(start_expr_address <= address);

                    if !start_expr_ty.is_numeric() {
                        return Err(SyntaxError::from_location(
                            start_expr.location(),
                            format!(
                                "Unexpected {} type, must be numeric",
                                start_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if start_expr_address == address {
                        address += 1;
                    }

                    let (end_expr_ty, end_expr_address) = Self::compile_expression(
                        address,
                        end_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(end_expr_address <= address);

                    if end_expr_ty != start_expr_ty {
                        return Err(SyntaxError::from_location(
                            end_expr.location(),
                            format!(
                                "Unexpected {} type, must be {}",
                                end_expr_ty.to_string().to_ascii_lowercase(),
                                start_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if end_expr_address == address {
                        address += 1;
                    }

                    let step_expr_address = if let Some(step_expr) = step_expr {
                        let (step_expr_ty, step_expr_address) = Self::compile_expression(
                            address,
                            step_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                        assert!(step_expr_address <= address);

                        if step_expr_ty != start_expr_ty {
                            return Err(SyntaxError::from_location(
                                step_expr.location(),
                                format!(
                                    "Unexpected {} type, must be {}",
                                    end_expr_ty.to_string().to_ascii_lowercase(),
                                    start_expr_ty.to_string().to_ascii_lowercase()
                                ),
                            ));
                        }

                        step_expr_address
                    } else {
                        program.push(
                            match start_expr_ty {
                                Type::Byte => Self::WriteByte(1, address),
                                Type::Float => Self::WriteFloat(1.0, address),
                                Type::Integer => Self::WriteInteger(1, address),
                                _ => unreachable!(),
                            }
                            .into(),
                        );

                        address
                    };

                    if step_expr_address == address {
                        address += 1;
                    }

                    let mut vars = vars.clone();
                    if vars
                        .insert(var.name, (start_expr_ty, start_expr_address))
                        .is_some()
                    {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!("Variable `{}` already defined", var.name),
                        ));
                    }

                    let body_offset = program.len() + program_offset;
                    let mut body = Self::compile_scope(
                        body_ast,
                        body_offset,
                        address,
                        fns,
                        subs,
                        vars.clone(),
                    )?;
                    let body_len = body.len();

                    for instr in &mut body {
                        if matches!(instr, ScopeInstruction::ExitFor(_)) {
                            *instr = Self::Jump(body_offset + body_len + 9).into();
                        }
                    }

                    program.append(&mut body);

                    match start_expr_ty {
                        Type::Byte => {
                            program.push(
                                Self::AddBytes(
                                    start_expr_address,
                                    step_expr_address,
                                    start_expr_address,
                                )
                                .into(),
                            );
                            program.push(Self::WriteByte(0, address).into());
                            program.push(
                                Self::GreaterThanBytes(step_expr_address, address, address).into(),
                            );
                            program.push(
                                Self::BranchNot(address, program_offset + program.len() + 4).into(),
                            );
                            program.push(
                                Self::GreaterThanBytes(
                                    start_expr_address,
                                    end_expr_address,
                                    address,
                                )
                                .into(),
                            );
                            program.push(Self::BranchNot(address, body_offset).into());
                            program.push(Self::Jump(program_offset + program.len() + 3).into());
                            program.push(
                                Self::GreaterThanBytes(
                                    end_expr_address,
                                    start_expr_address,
                                    address,
                                )
                                .into(),
                            );
                            program.push(Self::BranchNot(address, body_offset).into());
                        }
                        Type::Float => {
                            program.push(
                                Self::AddFloats(
                                    start_expr_address,
                                    step_expr_address,
                                    start_expr_address,
                                )
                                .into(),
                            );
                            program.push(Self::WriteFloat(0.0, address).into());
                            program.push(
                                Self::GreaterThanFloats(step_expr_address, address, address).into(),
                            );
                            program.push(
                                Self::BranchNot(address, program_offset + program.len() + 4).into(),
                            );
                            program.push(
                                Self::GreaterThanFloats(
                                    start_expr_address,
                                    end_expr_address,
                                    address,
                                )
                                .into(),
                            );
                            program.push(Self::BranchNot(address, body_offset).into());
                            program.push(Self::Jump(program_offset + program.len() + 3).into());
                            program.push(
                                Self::GreaterThanFloats(
                                    end_expr_address,
                                    start_expr_address,
                                    address,
                                )
                                .into(),
                            );
                            program.push(Self::BranchNot(address, body_offset).into());
                        }
                        Type::Integer => {
                            program.push(
                                Self::AddIntegers(
                                    start_expr_address,
                                    step_expr_address,
                                    start_expr_address,
                                )
                                .into(),
                            );
                            program.push(Self::WriteInteger(0, address).into());
                            program.push(
                                Self::GreaterThanIntegers(step_expr_address, address, address)
                                    .into(),
                            );
                            program.push(
                                Self::BranchNot(address, program_offset + program.len() + 4).into(),
                            );
                            program.push(
                                Self::GreaterThanIntegers(
                                    start_expr_address,
                                    end_expr_address,
                                    address,
                                )
                                .into(),
                            );
                            program.push(Self::BranchNot(address, body_offset).into());
                            program.push(Self::Jump(program_offset + program.len() + 3).into());
                            program.push(
                                Self::GreaterThanIntegers(
                                    end_expr_address,
                                    start_expr_address,
                                    address,
                                )
                                .into(),
                            );
                            program.push(Self::BranchNot(address, body_offset).into());
                        }
                        _ => unreachable!(),
                    }
                }
                Syntax::Function(var, args, body_ast) => {
                    if program_offset != 0 {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!("Cannot define function `{}` in this scope.", var.name),
                        ));
                    }

                    if vars.contains_key(var.name) {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!("Cannot redefine variable `{}` as function.", var.name),
                        ));
                    }

                    if var.ty.is_none() {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!("Function `{}` must declare a return type.", var.name),
                        ));
                    }

                    for arg in args {
                        if arg.ty.is_none() {
                            return Err(SyntaxError::from_location(
                                arg.location,
                                format!("Function argument `{}` must declare a type.", arg.name),
                            ));
                        }
                    }

                    if fns
                        .insert(
                            var.name,
                            FunctionScope {
                                ty: var.ty.unwrap(),
                                args,
                                global_vars: vars.clone(),
                                body_ast,
                            },
                        )
                        .is_some()
                    {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!("Cannot redefine function `{}`.", var.name),
                        ));
                    }
                }
                Syntax::Get(
                    (from_x_expr, from_y_expr),
                    (to_x_expr, to_y_expr),
                    var,
                    var_index_expr,
                ) => {
                    let (from_x_expr_ty, from_x_expr_address) = Self::compile_expression(
                        address,
                        from_x_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(from_x_expr_address <= address);

                    if from_x_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            from_x_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                from_x_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if from_x_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let (from_y_expr_ty, from_y_expr_address) = Self::compile_expression(
                        address,
                        from_y_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(from_y_expr_address <= address);

                    if from_y_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            from_y_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                from_y_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if from_y_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let (to_x_expr_ty, to_x_expr_address) = Self::compile_expression(
                        address,
                        to_x_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(to_x_expr_address <= address);

                    if to_x_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            to_x_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                to_x_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if to_x_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let (to_y_expr_ty, to_y_expr_address) = Self::compile_expression(
                        address,
                        to_y_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(to_y_expr_address <= address);

                    if to_y_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            to_y_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                to_y_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if to_y_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let var_index_expr_address = if let Some(var_index_expr) = var_index_expr {
                        let (var_index_expr_ty, var_index_expr_address) = Self::compile_expression(
                            address,
                            var_index_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                        assert!(var_index_expr_address <= address);

                        if var_index_expr_ty != Type::Integer {
                            return Err(SyntaxError::from_location(
                                var_index_expr.location(),
                                format!(
                                    "Unexpected {} type, must be integer",
                                    var_index_expr_ty.to_string().to_ascii_lowercase()
                                ),
                            ));
                        }

                        var_index_expr_address
                    } else {
                        program.push(Self::WriteInteger(0, address).into());

                        address
                    };

                    let (var_ty, var_address) = vars
                        .get(var.name)
                        .ok_or_else(|| {
                            SyntaxError::from_location(
                                var.location,
                                format!("`{}` is undefined.", var.name),
                            )
                        })
                        .copied()?;

                    if var_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!(
                                "Unexpected {} type, must be byte",
                                var_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    program.push(
                        Self::GetGraphic(
                            from_x_expr_address,
                            from_y_expr_address,
                            to_x_expr_address,
                            to_y_expr_address,
                            var_address,
                            var_index_expr_address,
                        )
                        .into(),
                    );
                }
                &Syntax::Goto(label) => {
                    program.push(label.into());
                }
                Syntax::If { tests, default } => {
                    let mut jumps_to_fix = vec![];
                    let mut branch_to_fix = None;

                    for (test_expr, test_body) in tests {
                        let (test_expr_ty, test_expr_address) = Self::compile_expression(
                            address,
                            test_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                        assert!(test_expr_address <= address);

                        if test_expr_ty != Type::Boolean {
                            return Err(SyntaxError::from_location(
                                test_expr.location(),
                                "Expected a boolean",
                            ));
                        }

                        let test_body = Self::compile_scope(
                            test_body,
                            program.len() + program_offset + 1,
                            address,
                            fns,
                            subs,
                            vars.clone(),
                        )?;

                        branch_to_fix = Some((
                            program.len(),
                            test_expr_address,
                            program.len() + program_offset + test_body.len() + 1,
                        ));

                        program.push(
                            Instruction::BranchNot(
                                test_expr_address,
                                program.len() + program_offset + test_body.len() + 2,
                            )
                            .into(),
                        );
                        program.extend(test_body.into_iter().map(Into::into));

                        jumps_to_fix.push(program.len());
                        program.push(Instruction::Jump(0).into());
                    }

                    if !default.is_empty() {
                        let default_body = Self::compile_scope(
                            default,
                            program.len() + program_offset,
                            address,
                            fns,
                            subs,
                            vars.clone(),
                        )?;

                        program.extend(default_body.into_iter().map(Into::into));
                    } else {
                        if let Some((branch, expr, index)) = branch_to_fix {
                            program[branch] = Instruction::BranchNot(expr, index).into();
                        }

                        if let Some(jump) = jumps_to_fix.last().copied() {
                            if jump == program.len() - 1 {
                                jumps_to_fix.pop().unwrap();
                                program.pop().unwrap();
                            }
                        }
                    }

                    for jmp in jumps_to_fix {
                        assert!(matches!(
                            program[jmp],
                            ScopeInstruction::Instruction(Instruction::Jump(_))
                        ));

                        program[jmp] = Instruction::Jump(program.len() + program_offset).into();
                    }
                }
                &Syntax::Label(label) => {
                    let program_index = program.len() + program_offset;
                    let unlocated_label: UnlocatedLabel = label.into();

                    if labels.insert(unlocated_label, program_index).is_some() {
                        return Err(SyntaxError::from_location(
                            label.location(),
                            format!("Duplicate label `{}`", label),
                        ));
                    }
                }
                Syntax::Locate(row_expr, col_expr) => {
                    let mut address = address;

                    let (row_expr_ty, row_expr_address) = Self::compile_expression(
                        address,
                        row_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(row_expr_address <= address);

                    if row_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            row_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                row_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if row_expr_address == address {
                        address += 1;
                    }

                    let col_expr_address = if let Some(col_expr) = col_expr {
                        let (col_expr_ty, col_expr_address) = Self::compile_expression(
                            address,
                            col_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                        assert!(col_expr_address <= address);

                        if col_expr_ty != Type::Integer {
                            return Err(SyntaxError::from_location(
                                col_expr.location(),
                                format!(
                                    "Unexpected {} type, must be integer",
                                    col_expr_ty.to_string().to_ascii_lowercase()
                                ),
                            ));
                        }

                        col_expr_address
                    } else {
                        program.push(Self::WriteInteger(0, address).into());

                        address
                    };

                    program.push(Self::Locate(col_expr_address, row_expr_address).into());
                }
                Syntax::Loop { test, body_ast } => {
                    if let Some((true, test_while, test_expr)) = test {
                        let test_offset = program_offset + program.len();
                        let (test_expr_ty, test_expr_address) = Self::compile_expression(
                            address,
                            test_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                        assert!(test_expr_address <= address);

                        if test_expr_ty != Type::Boolean {
                            return Err(SyntaxError::from_location(
                                test_expr.location(),
                                format!(
                                    "Unexpected {} type, must be boolean",
                                    test_expr_ty.to_string().to_ascii_lowercase()
                                ),
                            ));
                        }

                        let body_offset = program_offset + program.len() + 1;
                        let mut body = Self::compile_scope(
                            body_ast,
                            body_offset,
                            address,
                            fns,
                            subs,
                            vars.clone(),
                        )?;
                        let body_len = body.len();

                        for instr in &mut body {
                            if matches!(instr, ScopeInstruction::ExitDo(_)) {
                                *instr = Self::Jump(body_offset + body_len + 1).into();
                            }
                        }

                        if *test_while {
                            program.push(
                                Self::BranchNot(test_expr_address, body_offset + body.len() + 1)
                                    .into(),
                            );
                        } else {
                            program.push(
                                Self::Branch(test_expr_address, body_offset + body.len() + 1)
                                    .into(),
                            );
                        }

                        program.append(&mut body);
                        program.push(Self::Jump(test_offset).into());
                    } else if let Some((false, test_while, test_expr)) = test {
                        let body_offset = program_offset + program.len();
                        let mut body = Self::compile_scope(
                            body_ast,
                            body_offset,
                            address,
                            fns,
                            subs,
                            vars.clone(),
                        )?;
                        let body_len = body.len();
                        let program_len = program.len();

                        program.append(&mut body);

                        let (test_expr_ty, test_expr_address) = Self::compile_expression(
                            address,
                            test_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;
                        let test_offset = program_offset + program.len();

                        assert!(test_expr_address <= address);

                        if test_expr_ty != Type::Boolean {
                            return Err(SyntaxError::from_location(
                                test_expr.location(),
                                format!(
                                    "Unexpected {} type, must be boolean",
                                    test_expr_ty.to_string().to_ascii_lowercase()
                                ),
                            ));
                        }

                        for instr in &mut program[program_len..program_len + body_len] {
                            if matches!(instr, ScopeInstruction::ExitDo(_)) {
                                *instr = Self::Jump(test_offset + 1).into();
                            }
                        }

                        if *test_while {
                            program.push(Self::Branch(test_expr_address, body_offset).into());
                        } else {
                            program.push(Self::BranchNot(test_expr_address, body_offset).into());
                        }
                    } else {
                        let body_offset = program_offset + program.len();
                        let mut body = Self::compile_scope(
                            body_ast,
                            body_offset,
                            address,
                            fns,
                            subs,
                            vars.clone(),
                        )?;
                        let body_len = body.len();

                        for instr in &mut body {
                            if matches!(instr, ScopeInstruction::ExitDo(_)) {
                                *instr = Self::Jump(body_offset + body_len + 1).into();
                            }
                        }

                        program.append(&mut body);
                        program.push(Self::Jump(body_offset).into());
                    }
                }
                Syntax::Line(from_exprs, (to_x_expr, to_y_expr), color_expr) => {
                    let mut address = address;

                    let (from_x_expr_address, from_y_expr_address) =
                        if let Some((from_x_expr, from_y_expr)) = from_exprs {
                            let (from_x_expr_ty, from_x_expr_address) = Self::compile_expression(
                                address,
                                from_x_expr,
                                program_offset,
                                &mut program,
                                fns,
                                subs,
                                &vars,
                            )?;

                            assert!(from_x_expr_address <= address);

                            if from_x_expr_ty != Type::Integer {
                                return Err(SyntaxError::from_location(
                                    from_x_expr.location(),
                                    format!(
                                        "Unexpected {} type, must be integer",
                                        from_x_expr_ty.to_string().to_ascii_lowercase()
                                    ),
                                ));
                            }

                            if from_x_expr_address == address {
                                address += 1;
                            }

                            let (from_y_expr_ty, from_y_expr_address) = Self::compile_expression(
                                address,
                                from_y_expr,
                                program_offset,
                                &mut program,
                                fns,
                                subs,
                                &vars,
                            )?;

                            assert!(from_y_expr_address <= address);

                            if from_y_expr_ty != Type::Integer {
                                return Err(SyntaxError::from_location(
                                    from_y_expr.location(),
                                    format!(
                                        "Unexpected {} type, must be integer",
                                        from_y_expr_ty.to_string().to_ascii_lowercase()
                                    ),
                                ));
                            }

                            if from_y_expr_address == address {
                                address += 1;
                            }

                            (from_x_expr_address, from_y_expr_address)
                        } else {
                            let from_x_expr_address = address;
                            let from_y_expr_address = address + 1;
                            address += 2;

                            program.push(Self::WriteInteger(0, from_x_expr_address).into());
                            program.push(Self::WriteInteger(0, from_y_expr_address).into());

                            (from_x_expr_address, from_y_expr_address)
                        };

                    let (to_x_expr_ty, to_x_expr_address) = Self::compile_expression(
                        address,
                        to_x_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(to_x_expr_address <= address);

                    if to_x_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            to_x_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                to_x_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if to_x_expr_address == address {
                        address += 1;
                    }

                    let (to_y_expr_ty, to_y_expr_address) = Self::compile_expression(
                        address,
                        to_y_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(to_y_expr_address <= address);

                    if to_y_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            to_y_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                to_y_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if to_y_expr_address == address {
                        address += 1;
                    }

                    let (color_expr_ty, color_expr_address) = Self::compile_expression(
                        address,
                        color_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(color_expr_address <= address);

                    if color_expr_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            color_expr.location(),
                            format!(
                                "Unexpected {} type, must be byte",
                                color_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    program.push(
                        Self::Line(
                            from_x_expr_address,
                            from_y_expr_address,
                            to_x_expr_address,
                            to_y_expr_address,
                            color_expr_address,
                        )
                        .into(),
                    );
                }
                Syntax::Palette(color_index_expr, r_expr, g_expr, b_expr) => {
                    let mut address = address;

                    let (color_index_expr_ty, color_index_expr_address) = Self::compile_expression(
                        address,
                        color_index_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(color_index_expr_address <= address);

                    if color_index_expr_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            color_index_expr.location(),
                            format!(
                                "Unexpected {} type, must be byte",
                                color_index_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if color_index_expr_address == address {
                        address += 1;
                    }

                    let (r_expr_ty, r_expr_address) = Self::compile_expression(
                        address,
                        r_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(r_expr_address <= address);

                    if r_expr_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            r_expr.location(),
                            format!(
                                "Unexpected {} type, must be byte",
                                r_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if r_expr_address == address {
                        address += 1;
                    }

                    let (g_expr_ty, g_expr_address) = Self::compile_expression(
                        address,
                        g_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(g_expr_address <= address);

                    if g_expr_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            g_expr.location(),
                            format!(
                                "Unexpected {} type, must be byte",
                                g_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if g_expr_address == address {
                        address += 1;
                    }

                    let (b_expr_ty, b_expr_address) = Self::compile_expression(
                        address,
                        b_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(b_expr_address <= address);

                    if b_expr_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            b_expr.location(),
                            format!(
                                "Unexpected {} type, must be byte",
                                b_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    program.push(
                        Self::Palette(
                            color_index_expr_address,
                            r_expr_address,
                            g_expr_address,
                            b_expr_address,
                        )
                        .into(),
                    );
                }
                Syntax::Poke(addr_expr, val_expr) => {
                    let mut address = address;

                    let (addr_expr_ty, addr_expr_address) = Self::compile_expression(
                        address,
                        addr_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(addr_expr_address <= address);

                    if addr_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            addr_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                addr_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if addr_expr_address == address {
                        address += 1;
                    }

                    let (val_expr_ty, val_expr_address) = Self::compile_expression(
                        address,
                        val_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(val_expr_address <= address);

                    program.push(
                        match val_expr_ty {
                            Type::Boolean => Self::PokeBoolean(addr_expr_address, val_expr_address),
                            Type::Byte => Self::PokeByte(addr_expr_address, val_expr_address),
                            Type::Float => Self::PokeFloat(addr_expr_address, val_expr_address),
                            Type::Integer => Self::PokeInteger(addr_expr_address, val_expr_address),
                            Type::String => Self::PokeString(addr_expr_address, val_expr_address),
                        }
                        .into(),
                    );
                }
                Syntax::Print(prints) => {
                    program.push(Instruction::WriteString("".to_owned(), address).into());
                    let str_address = address;
                    address += 1;

                    for print in prints {
                        match print {
                            Print::Expression(expr) => {
                                let (expr_ty, expr_address) = Self::compile_expression(
                                    address,
                                    expr,
                                    program_offset,
                                    &mut program,
                                    fns,
                                    subs,
                                    &vars,
                                )?;

                                assert!(expr_address <= address);

                                match expr_ty {
                                    Type::Boolean => {
                                        program.push(
                                            Instruction::ConvertBooleanToString(
                                                expr_address,
                                                address,
                                            )
                                            .into(),
                                        );
                                        program.push(
                                            Instruction::AddStrings(
                                                str_address,
                                                address,
                                                str_address,
                                            )
                                            .into(),
                                        );
                                    }
                                    Type::Byte => {
                                        program.push(
                                            Instruction::ConvertByteToString(expr_address, address)
                                                .into(),
                                        );
                                        program.push(
                                            Instruction::AddStrings(
                                                str_address,
                                                address,
                                                str_address,
                                            )
                                            .into(),
                                        );
                                    }
                                    Type::Float => {
                                        program.push(
                                            Instruction::ConvertFloatToString(
                                                expr_address,
                                                address,
                                            )
                                            .into(),
                                        );
                                        program.push(
                                            Instruction::AddStrings(
                                                str_address,
                                                address,
                                                str_address,
                                            )
                                            .into(),
                                        );
                                    }
                                    Type::Integer => {
                                        program.push(
                                            Instruction::ConvertIntegerToString(
                                                expr_address,
                                                address,
                                            )
                                            .into(),
                                        );
                                        program.push(
                                            Instruction::AddStrings(
                                                str_address,
                                                address,
                                                str_address,
                                            )
                                            .into(),
                                        );
                                    }
                                    Type::String => {
                                        program.push(
                                            Instruction::AddStrings(
                                                str_address,
                                                expr_address,
                                                str_address,
                                            )
                                            .into(),
                                        );
                                    }
                                }
                            }
                            Print::Tab => {
                                program
                                    .push(Instruction::WriteString(" ".to_owned(), address).into());
                                program.push(
                                    Instruction::AddStrings(str_address, address, str_address)
                                        .into(),
                                );
                            }
                        }
                    }

                    program.push(Instruction::PrintString(str_address).into());

                    address = str_address;
                }
                Syntax::Pset(x_expr, y_expr, color_expr) => {
                    let mut address = address;
                    let (x_expr_ty, x_expr_address) = Self::compile_expression(
                        address,
                        x_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(x_expr_address <= address);

                    if x_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            x_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                x_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if x_expr_address == address {
                        address += 1;
                    }

                    let (y_expr_ty, y_expr_address) = Self::compile_expression(
                        address,
                        y_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(y_expr_address <= address);

                    if y_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            y_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                y_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if y_expr_address == address {
                        address += 1;
                    }

                    let (color_expr_ty, color_expr_address) = Self::compile_expression(
                        address,
                        color_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(color_expr_address <= address);

                    if color_expr_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            color_expr.location(),
                            format!(
                                "Unexpected {} type, must be byte",
                                color_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    program.push(
                        Self::SetPixel(x_expr_address, y_expr_address, color_expr_address).into(),
                    );
                }
                Syntax::Put(
                    (x_expr, y_expr),
                    (width_expr, height_expr),
                    var,
                    var_index_expr,
                    action,
                ) => {
                    let (x_expr_ty, x_expr_address) = Self::compile_expression(
                        address,
                        x_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(x_expr_address <= address);

                    if x_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            x_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                x_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if x_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let (y_expr_ty, y_expr_address) = Self::compile_expression(
                        address,
                        y_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(y_expr_address <= address);

                    if y_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            y_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                y_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if y_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let (width_expr_ty, width_expr_address) = Self::compile_expression(
                        address,
                        width_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(width_expr_address <= address);

                    if width_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            width_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                width_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if width_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let (height_expr_ty, height_expr_address) = Self::compile_expression(
                        address,
                        height_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(height_expr_address <= address);

                    if height_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            height_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                height_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let address = if height_expr_address == address {
                        address + 1
                    } else {
                        address
                    };

                    let var_index_expr_address = if let Some(var_index_expr) = var_index_expr {
                        let (var_index_expr_ty, var_index_expr_address) = Self::compile_expression(
                            address,
                            var_index_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                        assert!(var_index_expr_address <= address);

                        if var_index_expr_ty != Type::Integer {
                            return Err(SyntaxError::from_location(
                                var_index_expr.location(),
                                format!(
                                    "Unexpected {} type, must be integer",
                                    var_index_expr_ty.to_string().to_ascii_lowercase()
                                ),
                            ));
                        }

                        var_index_expr_address
                    } else {
                        program.push(Self::WriteInteger(0, address).into());

                        address
                    };

                    let (var_ty, var_address) = vars
                        .get(var.name)
                        .ok_or_else(|| {
                            SyntaxError::from_location(
                                var.location,
                                format!("`{}` is undefined.", var.name),
                            )
                        })
                        .copied()?;

                    if var_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            var.location,
                            format!(
                                "Unexpected {} type, must be byte",
                                var_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    program.push(
                        match action.unwrap_or(PutAction::Tset) {
                            PutAction::And => Self::PutGraphicAnd(
                                x_expr_address,
                                y_expr_address,
                                width_expr_address,
                                height_expr_address,
                                var_address,
                                var_index_expr_address,
                            ),
                            PutAction::Or => Self::PutGraphicOr(
                                x_expr_address,
                                y_expr_address,
                                width_expr_address,
                                height_expr_address,
                                var_address,
                                var_index_expr_address,
                            ),
                            PutAction::Pset => Self::PutGraphicPset(
                                x_expr_address,
                                y_expr_address,
                                width_expr_address,
                                height_expr_address,
                                var_address,
                                var_index_expr_address,
                            ),
                            PutAction::Preset => Self::PutGraphicPreset(
                                x_expr_address,
                                y_expr_address,
                                width_expr_address,
                                height_expr_address,
                                var_address,
                                var_index_expr_address,
                            ),
                            PutAction::Tset => Self::PutGraphicTset(
                                x_expr_address,
                                y_expr_address,
                                width_expr_address,
                                height_expr_address,
                                var_address,
                                var_index_expr_address,
                            ),
                            PutAction::Xor => Self::PutGraphicXor(
                                x_expr_address,
                                y_expr_address,
                                width_expr_address,
                                height_expr_address,
                                var_address,
                                var_index_expr_address,
                            ),
                        }
                        .into(),
                    );
                }
                Syntax::Rectangle(
                    from_exprs,
                    (to_x_expr, to_y_expr),
                    color_expr,
                    is_filled_expr,
                ) => {
                    let mut address = address;

                    let (from_x_expr_address, from_y_expr_address) =
                        if let Some((from_x_expr, from_y_expr)) = from_exprs {
                            let (from_x_expr_ty, from_x_expr_address) = Self::compile_expression(
                                address,
                                from_x_expr,
                                program_offset,
                                &mut program,
                                fns,
                                subs,
                                &vars,
                            )?;

                            assert!(from_x_expr_address <= address);

                            if from_x_expr_ty != Type::Integer {
                                return Err(SyntaxError::from_location(
                                    from_x_expr.location(),
                                    format!(
                                        "Unexpected {} type, must be integer",
                                        from_x_expr_ty.to_string().to_ascii_lowercase()
                                    ),
                                ));
                            }

                            if from_x_expr_address == address {
                                address += 1;
                            }

                            let (from_y_expr_ty, from_y_expr_address) = Self::compile_expression(
                                address,
                                from_y_expr,
                                program_offset,
                                &mut program,
                                fns,
                                subs,
                                &vars,
                            )?;

                            assert!(from_y_expr_address <= address);

                            if from_y_expr_ty != Type::Integer {
                                return Err(SyntaxError::from_location(
                                    from_y_expr.location(),
                                    format!(
                                        "Unexpected {} type, must be integer",
                                        from_y_expr_ty.to_string().to_ascii_lowercase()
                                    ),
                                ));
                            }

                            if from_y_expr_address == address {
                                address += 1;
                            }

                            (from_x_expr_address, from_y_expr_address)
                        } else {
                            let from_x_expr_address = address;
                            let from_y_expr_address = address + 1;
                            address += 2;

                            program.push(Self::WriteInteger(0, from_x_expr_address).into());
                            program.push(Self::WriteInteger(0, from_y_expr_address).into());

                            (from_x_expr_address, from_y_expr_address)
                        };

                    let (to_x_expr_ty, to_x_expr_address) = Self::compile_expression(
                        address,
                        to_x_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(to_x_expr_address <= address);

                    if to_x_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            to_x_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                to_x_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if to_x_expr_address == address {
                        address += 1;
                    }

                    let (to_y_expr_ty, to_y_expr_address) = Self::compile_expression(
                        address,
                        to_y_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(to_y_expr_address <= address);

                    if to_y_expr_ty != Type::Integer {
                        return Err(SyntaxError::from_location(
                            to_y_expr.location(),
                            format!(
                                "Unexpected {} type, must be integer",
                                to_y_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if to_y_expr_address == address {
                        address += 1;
                    }

                    let (color_expr_ty, color_expr_address) = Self::compile_expression(
                        address,
                        color_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(color_expr_address <= address);

                    if color_expr_ty != Type::Byte {
                        return Err(SyntaxError::from_location(
                            color_expr.location(),
                            format!(
                                "Unexpected {} type, must be byte",
                                color_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    if color_expr_address == address {
                        address += 1;
                    }

                    let is_filled_expr_address = if let Some(is_filled_expr) = is_filled_expr {
                        let (is_filled_expr_ty, is_filled_expr_address) = Self::compile_expression(
                            address,
                            is_filled_expr,
                            program_offset,
                            &mut program,
                            fns,
                            subs,
                            &vars,
                        )?;

                        assert!(is_filled_expr_address <= address);

                        if is_filled_expr_ty != Type::Boolean {
                            return Err(SyntaxError::from_location(
                                is_filled_expr.location(),
                                format!(
                                    "Unexpected {} type, must be boolean",
                                    is_filled_expr_ty.to_string().to_ascii_lowercase()
                                ),
                            ));
                        }

                        is_filled_expr_address
                    } else {
                        program.push(Self::WriteBoolean(false, address).into());

                        address
                    };

                    program.push(
                        Self::Rectangle(
                            from_x_expr_address,
                            from_y_expr_address,
                            to_x_expr_address,
                            to_y_expr_address,
                            color_expr_address,
                            is_filled_expr_address,
                        )
                        .into(),
                    );
                }
                Syntax::Select {
                    test_expr,
                    test_cases,
                    default_ast,
                } => {
                    let mut address = address;
                    let (test_expr_ty, test_expr_address) = Self::compile_expression(
                        address,
                        test_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(test_expr_address <= address);

                    if test_expr_address == address {
                        address += 1;
                    }

                    let mut jumps =
                        Vec::with_capacity(test_cases.len() + !default_ast.is_empty() as usize);
                    for (test_cases, test_case_ast) in test_cases {
                        for test_case in test_cases {
                            match test_case {
                                Case::RangeFull(lower_bound_expr, upper_bound_expr) => {
                                    let (lower_bound_expr_ty, lower_bound_expr_address) =
                                        Self::compile_expression(
                                            address,
                                            lower_bound_expr,
                                            program_offset,
                                            &mut program,
                                            fns,
                                            subs,
                                            &vars,
                                        )?;

                                    assert!(lower_bound_expr_address <= address);

                                    if lower_bound_expr_ty != test_expr_ty {
                                        return Err(SyntaxError::from_location(
                                            lower_bound_expr.location(),
                                            format!(
                                                "Unexpected {} type, must be {}",
                                                lower_bound_expr_ty
                                                    .to_string()
                                                    .to_ascii_lowercase(),
                                                test_expr_ty.to_string().to_ascii_lowercase()
                                            ),
                                        ));
                                    } else if !lower_bound_expr_ty.is_numeric() {
                                        return Err(SyntaxError::from_location(
                                            lower_bound_expr.location(),
                                            format!(
                                                "Unexpected {} type, must be numeric",
                                                lower_bound_expr_ty
                                                    .to_string()
                                                    .to_ascii_lowercase()
                                            ),
                                        ));
                                    }

                                    if lower_bound_expr_address == address {
                                        address += 1;
                                    }

                                    let (upper_bound_expr_ty, upper_bound_expr_address) =
                                        Self::compile_expression(
                                            address,
                                            upper_bound_expr,
                                            program_offset,
                                            &mut program,
                                            fns,
                                            subs,
                                            &vars,
                                        )?;

                                    assert!(upper_bound_expr_address <= address);

                                    if upper_bound_expr_ty != test_expr_ty {
                                        return Err(SyntaxError::from_location(
                                            upper_bound_expr.location(),
                                            format!(
                                                "Unexpected {} type, must be {}",
                                                upper_bound_expr_ty
                                                    .to_string()
                                                    .to_ascii_lowercase(),
                                                test_expr_ty.to_string().to_ascii_lowercase()
                                            ),
                                        ));
                                    } else if !upper_bound_expr_ty.is_numeric() {
                                        return Err(SyntaxError::from_location(
                                            upper_bound_expr.location(),
                                            format!(
                                                "Unexpected {} type, must be numeric",
                                                upper_bound_expr_ty
                                                    .to_string()
                                                    .to_ascii_lowercase()
                                            ),
                                        ));
                                    }

                                    if upper_bound_expr_address == address {
                                        address += 1;
                                    }

                                    program.push(
                                        match test_expr_ty {
                                            Type::Byte => Self::GreaterThanBytes,
                                            Type::Float => Self::GreaterThanFloats,
                                            Type::Integer => Self::GreaterThanIntegers,
                                            _ => unreachable!(),
                                        }(
                                            lower_bound_expr_address, test_expr_address, address
                                        )
                                        .into(),
                                    );
                                    program.push(
                                        match test_expr_ty {
                                            Type::Byte => Self::GreaterThanBytes,
                                            Type::Float => Self::GreaterThanFloats,
                                            Type::Integer => Self::GreaterThanIntegers,
                                            _ => unreachable!(),
                                        }(
                                            test_expr_address, upper_bound_expr_address, address + 1
                                        )
                                        .into(),
                                    );
                                    program.push(
                                        Self::OrBooleans(address, address + 1, address).into(),
                                    );
                                }
                                Case::Relation(relation, relation_expr) => {
                                    if !test_expr_ty.is_numeric()
                                        && !matches!(
                                            relation,
                                            Relation::Equal(_) | Relation::NotEqual(_)
                                        )
                                    {
                                        return Err(SyntaxError::from_location(
                                            relation.location(),
                                            format!(
                                                "Unexpected {} relation, must be = or <> when type is {}",
                                                relation,test_expr_ty
                                                    .to_string()
                                                    .to_ascii_lowercase()
                                            ),
                                        ));
                                    }

                                    let (relation_expr_ty, relation_expr_address) =
                                        Self::compile_expression(
                                            address,
                                            relation_expr,
                                            program_offset,
                                            &mut program,
                                            fns,
                                            subs,
                                            &vars,
                                        )?;

                                    assert!(relation_expr_address <= address);

                                    if relation_expr_ty != test_expr_ty {
                                        return Err(SyntaxError::from_location(
                                            relation_expr.location(),
                                            format!(
                                                "Unexpected {} type, must be {}",
                                                relation_expr_ty.to_string().to_ascii_lowercase(),
                                                test_expr_ty.to_string().to_ascii_lowercase()
                                            ),
                                        ));
                                    }

                                    if relation_expr_address == address {
                                        address += 1;
                                    }

                                    // Swap the order of test/relation expressions if we're doing >
                                    // or >= because we don't have < or <= instructions.
                                    let (test_expr_address, relation_expr_address) = if matches!(
                                        relation,
                                        Relation::GreaterThan(_) | Relation::GreaterThanEqual(_)
                                    ) {
                                        (relation_expr_address, test_expr_address)
                                    } else {
                                        (test_expr_address, relation_expr_address)
                                    };

                                    program.push(
                                        match (relation, test_expr_ty) {
                                            (Relation::Equal(_), Type::Boolean) => {
                                                Self::NotEqualBooleans
                                            }
                                            (Relation::Equal(_), Type::Byte) => Self::NotEqualBytes,
                                            (Relation::Equal(_), Type::Float) => {
                                                Self::NotEqualFloats
                                            }
                                            (Relation::Equal(_), Type::Integer) => {
                                                Self::NotEqualIntegers
                                            }
                                            (Relation::Equal(_), Type::String) => {
                                                Self::NotEqualStrings
                                            }
                                            (
                                                Relation::GreaterThan(_) | Relation::LessThan(_),
                                                Type::Byte,
                                            ) => Self::GreaterThanEqualBytes,
                                            (
                                                Relation::GreaterThan(_) | Relation::LessThan(_),
                                                Type::Float,
                                            ) => Self::GreaterThanEqualFloats,
                                            (
                                                Relation::GreaterThan(_) | Relation::LessThan(_),
                                                Type::Integer,
                                            ) => Self::GreaterThanEqualIntegers,
                                            (
                                                Relation::GreaterThanEqual(_)
                                                | Relation::LessThanEqual(_),
                                                Type::Byte,
                                            ) => Self::GreaterThanBytes,
                                            (
                                                Relation::GreaterThanEqual(_)
                                                | Relation::LessThanEqual(_),
                                                Type::Float,
                                            ) => Self::GreaterThanFloats,
                                            (
                                                Relation::GreaterThanEqual(_)
                                                | Relation::LessThanEqual(_),
                                                Type::Integer,
                                            ) => Self::GreaterThanIntegers,
                                            (Relation::NotEqual(_), Type::Boolean) => {
                                                Self::EqualBooleans
                                            }
                                            (Relation::NotEqual(_), Type::Byte) => Self::EqualBytes,
                                            (Relation::NotEqual(_), Type::Float) => {
                                                Self::EqualFloats
                                            }
                                            (Relation::NotEqual(_), Type::Integer) => {
                                                Self::EqualIntegers
                                            }
                                            (Relation::NotEqual(_), Type::String) => {
                                                Self::EqualStrings
                                            }
                                            _ => unreachable!(),
                                        }(
                                            test_expr_address, relation_expr_address, address
                                        )
                                        .into(),
                                    );
                                }
                            }

                            let mut test_case_ast = Self::compile_scope(
                                test_case_ast,
                                program_offset + program.len() + 1,
                                address,
                                fns,
                                subs,
                                vars.clone(),
                            )?;

                            program.push(
                                Self::Branch(
                                    address,
                                    program_offset + program.len() + test_case_ast.len() + 2,
                                )
                                .into(),
                            );
                            program.append(&mut test_case_ast);
                            jumps.push(program.len());
                            program.push(Self::Jump(0).into());
                        }
                    }

                    program.append(&mut Self::compile_scope(
                        default_ast,
                        program_offset + program.len(),
                        address,
                        fns,
                        subs,
                        vars.clone(),
                    )?);

                    for jump in jumps {
                        program[jump] = Self::Jump(program_offset + program.len()).into();
                    }
                }
                Syntax::Sub(id, args, body_ast) => {
                    if program_offset != 0 {
                        return Err(SyntaxError::from_location(
                            id.location,
                            format!("Cannot define sub `{}` in this scope.", id.name),
                        ));
                    }

                    if vars.contains_key(id.name) {
                        return Err(SyntaxError::from_location(
                            id.location,
                            format!("Cannot redefine variable `{}` as sub.", id.name),
                        ));
                    }

                    for arg in args {
                        if arg.ty.is_none() {
                            return Err(SyntaxError::from_location(
                                arg.location,
                                format!("Sub argument `{}` must declare a type.", arg.name),
                            ));
                        }
                    }

                    if subs
                        .insert(
                            id.name,
                            SubScope {
                                args,
                                global_vars: vars.clone(),
                                body_ast,
                            },
                        )
                        .is_some()
                    {
                        return Err(SyntaxError::from_location(
                            id.location,
                            format!("Cannot redefine sub `{}`.", id.name),
                        ));
                    }
                }
                Syntax::While {
                    test_expr,
                    body_ast,
                } => {
                    let test_offset = program.len() + program_offset;
                    let (test_expr_ty, test_expr_address) = Self::compile_expression(
                        address,
                        test_expr,
                        program_offset,
                        &mut program,
                        fns,
                        subs,
                        &vars,
                    )?;

                    assert!(test_expr_address <= address);

                    if test_expr_ty != Type::Boolean {
                        return Err(SyntaxError::from_location(
                            test_expr.location(),
                            format!(
                                "Unexpected {} type, must be boolean",
                                test_expr_ty.to_string().to_ascii_lowercase()
                            ),
                        ));
                    }

                    let body_offset = program.len() + program_offset + 1;
                    let mut body = Self::compile_scope(
                        body_ast,
                        body_offset,
                        address,
                        fns,
                        subs,
                        vars.clone(),
                    )?;
                    let body_len = body.len();

                    for instr in &mut body {
                        if matches!(instr, ScopeInstruction::ExitWhile(_)) {
                            *instr = Self::Jump(body_offset + body_len + 1).into();
                        }
                    }

                    program.push(
                        Self::BranchNot(test_expr_address, body_offset + body.len() + 1).into(),
                    );
                    program.append(&mut body);
                    program.push(Self::Jump(test_offset).into());
                }
                Syntax::Yield => {
                    program.push(Instruction::Yield.into());
                }
            }
        }

        for instr in &mut program {
            if let ScopeInstruction::Label(label) = instr {
                if let Some(program_index) = labels.get(&(*label).into()).copied() {
                    *instr = Instruction::Jump(program_index).into();
                }
            }
        }

        Ok(program)
    }
}

enum ScopeInstruction<'a> {
    ExitDo(Exit<'a>),
    ExitFor(Exit<'a>),
    ExitFunction(Exit<'a>),
    ExitSub(Exit<'a>),
    ExitWhile(Exit<'a>),
    Instruction(Instruction),
    Label(Label<'a>),
}

impl<'a> ScopeInstruction<'a> {
    fn instruction(self) -> Result<Instruction, SyntaxError> {
        match self {
            Self::Instruction(instr) => Ok(instr),
            Self::Label(label) => Err(SyntaxError::from_location(
                label.location(),
                format!("Undefined Label `{label}`"),
            )),
            ScopeInstruction::ExitDo(exit)
            | ScopeInstruction::ExitFor(exit)
            | ScopeInstruction::ExitFunction(exit)
            | ScopeInstruction::ExitSub(exit)
            | ScopeInstruction::ExitWhile(exit) => Err(SyntaxError::from_location(
                exit.location(),
                "Unexpected exit",
            )),
        }
    }
}

impl<'a> From<Instruction> for ScopeInstruction<'a> {
    fn from(instr: Instruction) -> Self {
        Self::Instruction(instr)
    }
}

impl<'a> From<Label<'a>> for ScopeInstruction<'a> {
    fn from(instr: Label<'a>) -> Self {
        Self::Label(instr)
    }
}

#[derive(Clone)]
struct SubScope<'a> {
    args: &'a [Variable<'a>],
    global_vars: HashMap<&'a str, (Type, Address)>,
    body_ast: &'a [Syntax<'a>],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_assign_add() {
        let input = br#"
            DIM a%
            a = 1
            a = a + 1
        "#;

        // TODO: Run a pass on the instructions which removes copies and writes if possible, the
        // below instructions can be simplified!
        let expected = vec![
            Instruction::WriteInteger(0, 0),
            Instruction::WriteInteger(1, 1),
            Instruction::Copy(1, 0),
            Instruction::WriteInteger(1, 2),
            Instruction::AddIntegers(0, 2, 2),
            Instruction::Copy(2, 0),
        ];

        let result = Instruction::compile(input).unwrap();

        assert_eq!(expected, result);
    }
}
