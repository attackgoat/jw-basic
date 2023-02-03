# JW-Basic

[![LoC](https://tokei.rs/b1/github/attackgoat/jw-basic?category=code)](https://github.com/attackgoat/jw-basic)

A toy language that is somewhat like QBasic. 

_Features:_

- Graphics: 160x96 (255 colors)
- Text: 32x16 (4x5 font)
- Character set: ASCII (32-127)
- Keyboard input

_TODO:_

Some features are not fully implemented!

- Arrays: 100% but tuples/assign-by-tuple 75%
- Graphics: interpreter 100% but instructions 75%
- Keyboard code: 10%
- Documentation: lacking!
- Bugs: lots!

_Language demonstration:_

```
' This is a comment! Types you may use:
'  ?: Boolean
'  @: Unsigned byte
'  %: Signed 32-bit integer
'  !: Single-precision float
'  $: String

DIM myVar$ = "Variable initialization is optional"
DIM answer% = 42, question? = TRUE
DIM latitude!

' Type specifiers are not required:
latitude = 100.0

IF question THEN
    latitude! = -100.0!
END IF

CLS
PRINT "Hello, there!", myVar

' A diagonal red line
LINE (0, 0) - (159, 95), 4@

' Some colorful boxes
FOR c@ = 5 TO 95
    RECTANGLE (c - 5, c) - (159, c + 1), c
NEXT FOR

```

_See full specification below_

## Example Programs

This repository contains several example programs. Access the helpfile with `--help`:

```
A BASIC language interpreter. Does not conform to existing standards. Mostly a toy.

Usage: jw-basic [OPTIONS] <PATH>

Arguments:
  <PATH>
          File to load and interpret (.bas format)

Options:
  -d, --debug
          Display debugging information
          
          NOTE: Set `RUST_LOG=debug` environment variable to display output

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Hello, world!

[examples/hello_world.bas](examples/hello_world.bas)

```
cargo run examples/hello_world.bas
```

<image alt="Preview" src=".github/img/hello_world.png" height=364 width=535>

### Moose3D

```
cargo run examples/moose3d.bas
```

## Language Specifications

```
CLS

    CLS clears the screen of text and graphics. Color 0 is used.


COLOR foreground@[, background@]

    COLOR sets text and graphics colors.

    foreground: The color of characters and any lines or rectangles which do not specify a color.
    backround:  The backgroud of text characters or the fill-color of rectangles which do not
                specify a color.

    Examples:

        COLOR 4, 14 ' Red on yellow, danger!


DIM [SHARED] variable[type][(subscripts)] [= value] [, variable[type][(subscripts)]] [= value] ...

    DIM declares variables and arrays.

    SHARED:     Specifies that variables are shared with all SUB or FUNCTION procedures in the
                module.
    variable:   The name of a variable.
    type:       Type of data which may be stored:
                    ? (boolean)
                    @ (byte)
                    % (integer)
                    ! (float)
                    $ (string)
    subscripts: [lower% TO] upper% [, [lower% TO] upper%] ...
                lower: Lower bound of the array. The default bound is zero.
                upper: Upper bound of the array. Inclusive.

    Examples:

        DIM SHARED name$, myMatrix(2, 2) = [[1e-15!, 2!], [3.14!, 4.2!]]
        DIM total% = 5


PRINT [expression][; expression][, expression]

    PRINT displays text using the current foreground and background colors at the current cursor
    location.

    expression: Any expression.
    semicolon:  Prints the following expression with zero additional spaces.
    comma:      Prints the following expression with one additional space.

    Examples:

        PRINT "Hello " + name$ + ". Nice to meet you!", "Welcome to day ", dayOfWeek%; "!"
```

## Credits

Raycasting example inspired by [Lode's Computer Graphics Tutorial](https://lodev.org/cgtutor/raycasting.html)