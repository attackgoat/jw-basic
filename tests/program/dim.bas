DIM bool1 = TRUE
DIM bool2? = TRUE XOR FALSE

DIM byte1 = &hFF@
DIM byte2@ = 255@

DIM int1 = 42
DIM int2% = &H0000002A

DIM float1 = 42.0
DIM float2! = 42e0

DIM string1 = "Test"
DIM string2$ = "Te" + "st"

' Should print TRUE
PRINT (bool1 = bool2) AND (byte1 = byte2) AND (int1 = int2) AND (float1 = float2) AND (string1 = string2)

DIM var1%, var2$, var3$ = "!"

REM Should print "0  0"
PRINT var1, var2, var1

' Should print "00"
PRINT var1; var1

' Should print "Hello!"
PRINT "Hello" + var3

DIM varA = "A", varAB = varA + "B", varCAB = "C" + varAB + "!"
PRINT varCAB

DIM myArray1()
dim myArray2(123)
DIM myArray3(-123 TO 123)
dim myArray4(-123 to 123, 456)
DIM myArray5(-123 TO 123, -456 to 456)

dim myArray6%()
DIM myArray7%(123)
dim myArray8%(-123 to 123)
DIM myArray9%(-123 TO 123, 456)
dim myArray0%(-123 to 123, -456 TO 456)

dim myArrayA%(-123 to -99, -456 TO -99)

myArray9(-56, 66) = 22
myArray9(-56, 67) = 0
myArray9  (-56, 67) = 1

PRINT myArray9(-56, 67), myArray9(-56, 66), myArray9(-55, 62)

DIM myArrayB(-1 to 1)
myArrayB(-1) = 1
myArrayB(0) = 2
myArrayB(1) = 3

PRINT myArrayB(-1), myArrayB(0), myArrayB(1)