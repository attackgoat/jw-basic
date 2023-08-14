
RECTANGLE (0, 0) - (159, 95), 4@, TRUE
RECTANGLE (1, 1) - (158, 94), 54@, TRUE

FOR Y = 0 TO 17
    LINE (2, Y * 5 + 2) - (48, Y + 2), BYTE(Y + 32)
    LINE (2, Y * 5 + 3) - (48, Y + 2), BYTE(Y + 32)
    LINE (2, Y * 5 + 4) - (48, Y + 2), BYTE(Y + 32)
    LINE (2, Y * 5 + 5) - (48, Y + 2), BYTE(Y + 32)
    LINE (2, Y * 5 + 6) - (48, Y + 2), BYTE(Y + 32)
NEXT

DIM myRectangle@(3)
GET (0, 0) - (1, 1), myRectangle
PUT (157, 93), (2, 2), myRectangle, AND
PUT (97, 2), (2, 2), myRectangle

DIM colorfulRectangle@(46 * 90)
GET (2, 2) - (47, 91), colorfulRectangle
PUT (50, 2), (46, 90), colorfulRectangle

x = 2
width = 4
DIM verticalSlice@(96 * width)
GET (x, 0) - (x + width - 1, 95), verticalSlice
PUT (100, 0), (width, 96), verticalSlice, AND
PUT (100 + width + 1, 0), (width, 96), verticalSlice, OR
PUT (100 + width * 2 + 2, 0), (width, 96), verticalSlice, PSET
PUT (100 + width * 3 + 3, 0), (width, 96), verticalSlice, PRESET
PUT (100 + width * 4 + 4, 0), (width, 96), verticalSlice, XOR

bigBorderWidth = 120
bigBorderHeight = 50
DIM bigBorderImage@(0 TO (bigBorderWidth * bigBorderHeight) - 1)
FOR by = 0 TO bigBorderHeight - 1
    FOR bx = 0 to bigBorderWidth - 1
        bigBorderImage(by * bigBorderWidth + bx) = 255@
    NEXT
NEXT

bigBorderSize = 2
FOR by = 0 TO bigBorderHeight - 1
    FOR bx = 0 to bigBorderWidth - 1
        IF bx < bigBorderSize OR bx >= bigBorderWidth - bigBorderSize OR by < bigBorderSize OR by >= bigBorderHeight - bigBorderSize THEN
            bigBorderImage(by * bigBorderWidth + bx) = 4@
        END IF
    NEXT
NEXT

PUT (160 / 2 - bigBorderWidth / 2 - 1, 96 / 2 - bigBorderHeight / 2 - 1), (bigBorderWidth, bigBorderHeight), bigBorderImage, TSET


DIM fullScreen@(0 TO (160 * 96) - 1)
GET (0, 0) - (159, 95), fullScreen

YIELD
CLS

PUT (0, 0), (160, 96), fullScreen
