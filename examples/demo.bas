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