FUNCTION func1?
END FUNCTION

FUNCTION func2?
    func2 = TRUE
END FUNCTION

FUNCTION func3?
    func3 = func2()
END FUNCTION

FUNCTION func4%(x%, y%)
    IF x > y THEN
        func4 = x
    ELSE THEN
        func4 = y
    END IF
END FUNCTION

globalVar = "Hello, world"

FUNCTION func5$
    func5 = globalVar
END FUNCTION

PRINT func1()
PRINT func2()
PRINT func3()
PRINT func4(2,5)
PRINT func4(899,123)
PRINT func5()
PRINT "OK1"

FUNCTION areEqual?(lhs$, rhs$)
    IF lhs = rhs THEN
        areEqual = TRUE
    ELSE THEN
        areEqual = FALSE
    END IF
END FUNCTION

myGlobal = 0

FUNCTION changeGlobal%
    myGlobal = myGlobal + 1
    changeGlobal = myGlobal
END FUNCTION

PRINT areEqual("Apples", "Oranges")
PRINT changeGlobal()
PRINT "OK2"