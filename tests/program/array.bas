DIM bunch(3 TO 4, 6 TO 8, 45 TO 46, 77 TO 77)

bunch(3, 6, 45, 77) = 90
bunch(3, 7, 45, 77) = 91
bunch(3, 8, 45, 77) = 92
bunch(4, 6, 45, 77) = 93
bunch(4, 7, 45, 77) = 94
bunch(4, 8, 45, 77) = 95
bunch(3, 6, 46, 77) = 96
bunch(3, 7, 46, 77) = 97
bunch(3, 8, 46, 77) = 98
bunch(4, 6, 46, 77) = 99
bunch(4, 7, 46, 77) = 100
bunch(4, 8, 46, 77) = 101

PRINT bunch(3, 6, 45, 77)
PRINT bunch(3, 7, 45, 77)
PRINT bunch(3, 8, 45, 77)
PRINT bunch(4, 6, 45, 77)
PRINT bunch(4, 7, 45, 77)
PRINT bunch(4, 8, 45, 77)
PRINT bunch(3, 6, 46, 77)
PRINT bunch(3, 7, 46, 77)
PRINT bunch(3, 8, 46, 77)
PRINT bunch(4, 6, 46, 77)
PRINT bunch(4, 7, 46, 77)
PRINT bunch(4, 8, 46, 77)

DIM myBooleans?(3)
myBooleans(0) = TRUE

DIM myBytes@(3)
myBytes(0) = &haa@

DIM myFloats!(3)
myFloats(0) = 4.2

DIM myStrings$(3)
myStrings(0) = ":)"

PRINT myBooleans(0),myBooleans(1),myBytes(0),myBytes(1),myFloats(0),myFloats(1),myStrings(0),myStrings(1),"!"