SUB mySub
    PRINT "A"
END SUB

SUB mySub2()
    PRINT "B"
END SUB

SUB mySub3 ( )
    mySub
    CALL mySub
    mySub2
END SUB

SUB mySub4 (letter$)
    PRINT letter
END SUB

SUB mySub5(lhs$, rhs$)
    PRINT lhs, rhs$
END SUB

mySub3
mySub4 "C"
CALL mySub4 ("D")
mySub5 "<", ">"

PRINT "OK"
