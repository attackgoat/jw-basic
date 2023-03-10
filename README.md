<image alt="logo" src=".github/img/logo.png" width=100%>

# JW-Basic

[![LoC](https://tokei.rs/b1/github/attackgoat/jw-basic?category=code)](https://github.com/attackgoat/jw-basic)

A toy language that is somewhat like QBasic. 

_Features:_

- Graphics: 160x96 (255 colors & transparent)
- Text: 32x16 (4x5 font)
- Character set: ASCII (32-127)
- Keyboard input
- Multidimensional arrays

_Design:_

- Parses tokens using [`nom`](https://github.com/rust-bakery/nom) and
  [`nom_locate`](https://github.com/fflorent/nom_locate)
- Syntax & expression tree parsed with from tokens
- Assembly-like instructions emitted from syntax
- Instructions executed using a register-based virtual machine
- Graphics & text output using [`Screen-13`](https://github.com/attackgoat/screen-13)
- Operates as a library or command-line program

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
LINE (0, 13) - (159, 21), 4@

' Some colorful boxes
FOR c = 25 TO 95 STEP 3
    RECTANGLE (c - 5, c - 3) - (159, c), CBYTE(c), TRUE
NEXT

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

<image alt="Preview" src=".github/img/hello_world.png" height=439 width=660>

### Raycast

[examples/raycast.bas](examples/raycast.bas)

```
cargo run examples/raycast.bas
```

<image alt="Preview" src=".github/img/raycast.png" height=439 width=660>

## Language Specifications

```
ABS[! | %](expr) | COS[!](expr) | SIN[!](expr)

    Math functions.

    expr: Any expression.

    Examples:

        ABS(-1.0) + COS(4.5) ' Radians of course!


CLS

    CLS clears the screen of text and graphics. Color 0 is used.


COLOR foreground@[, background@]

    COLOR sets text and graphics colors.

    foreground: The color of characters and any lines or rectangles which do not specify a color.
    backround:  The backgroud of text characters or the fill-color of rectangles which do not
                specify a color.

    Examples:

        COLOR 4, 14 ' Red on yellow, danger!


CBOOLEAN[?](expr) | CBYTE[@](expr) | CFLOAT[!](expr) | CINT[%](expr) | CSTR[$](expr)

    Converts an expression to another type.

    expr: Any expression.

    Examples:

        CSTR(1.0)


DIM var[type][(subscripts)] [= value] [, var[type][(subscripts)]] [= value] ...

    DIM declares variables and arrays. Variables may also be simply assigned without DIM.

    var:        The name of a variable.
    type:       Type of data which may be stored:
                    ? (boolean)
                    @ (byte)
                    ! (float)
                    % (integer)
                    $ (string)
    subscripts: [lower% TO] upper% [, [lower% TO] upper%] ...
                lower: Lower bound of the array. The default bound is zero.
                upper: Upper bound of the array. Inclusive.
    value:      Any expression. Not supported with arrays yet.

    Examples:

        DIM name$, myMatrix!(2, -2 TO 2)
        DIM total% = 5
        myMatrix(2, -2) = 10.0


FOR var = start TO end [STEP step]
    [..]
NEXT [var]

    Loop where `var` is incremented (or decremented) from `start` to `end` in `step` increments.

    var:   A byte, float, or integer variable defined for the body of the FOR..NEXT statement.
    start: Any expression evaluated to become the initial value of `var`.
    end:   Any expression evaluated to become the inclusive final value of `var`.
    step:  Any expression evaluated to be added to `var` for each iteration.

    Examples:

        FOR temperature = 96.0 to 104.5 STEP 0.1
            PRINT temperature
        NEXT


GOTO [label | line number]

    Jumps directly to a given labelled or numbered line. Fun at parties.

    Examples:

        Again:
        PRINT "Dance!"
        GOTO Again


IF expr THEN
    [..]
[ELSE IF expr THEN]
    [..]
[ELSE THEN]
    [..]
END IF

    Branching logic tree.

    expr: Any expression which evaluates to a boolean.


KEYDOWN[@](expr)

    Returns TRUE when a given key is pressed.

    expr: Any expression which evaluates to a byte, see source code for the keys which have
          been setup.


LINE [(x0, y0) -] (x1, y1), color

    Draws a line between two points.

    x0, y0, x1, y1: Any expression which evaluates to an integer.
    color:          Any expression which evaluates to a byte.


LOCATE row[, col]

    Moves the text output location of the following PRINT statements.


PALETTE color, r, g, b

    Changes the currently active palette allowing for colorful animation without re-drawing the
    screen.

    color:   Any expression which evaluates to a byte in the 0-254 range. 255 (&hFF@) is
             transparent.
    r, g, b: Any expression which evaluates to a byte.


PRINT [expr][; expr][, expr]

    PRINT displays text using the current foreground and background colors at the current cursor
    location.

    expr:       Any expression.
    semicolon:  Prints the following expression with zero additional spaces.
    comma:      Prints the following expression with one additional space.

    Examples:

        PRINT "Hello " + name$ + ". Nice to meet you!", "Welcome to day ", dayOfWeek%; "!"


RECTANGLE [(x0, y0) -] (x1, y1), color[, filled]

    Draws a rectangle between two points.

    x0, y0, x1, y1: Any expression which evaluates to an integer.
    color:          Any expression which evaluates to a byte.
    filled:         Any expression which evaluates to a boolean.


TIMER[%]()

    Returns the number of microseconds since the program began execution.


WHILE expr
    [..]
WEND

    Loop which begins if `expr` is TRUE and continues until it is FALSE.

    expr: Any expression which evaluates to a boolean.


YIELD

    Pause execution of a program until the next update of the interpreter. Without calling this
    execution will continue until the final statement is executed.
```

## Tests

In addition to the test programs, there are unit and integration tests of the language. When
something goes wrong *you should* receive an error indicating the line and column number which
caused the issue.

Running the tests:

```
cargo test
```


## Credits

This project was designed completely for fun and to learn how a language might be developed. I hope
you find something useful that you can bring to your projects, just like I was able to find sources
of inspiration for _JW-Basic_.

- [QBasic cafe](https://www.qbasic.net/en/reference/qb11/overview.htm) used for reference
  documentation
- Parsing code inspired by [`monkey-rust`](https://github.com/Rydgel/monkey-rust)
- Raycasting example inspired by
  [Lode's Computer Graphics Tutorial](https://lodev.org/cgtutor/raycasting.html)

Feel free to submit PRs if you would like to enhance this code or fill out remaining features.
