SUB TESTA(value%)
    SELECT CASE value
        CASE 0: PRINT value, "is", 0
        CASE 1: PRINT value, "is", 1
        CASE 2, 3: PRINT value, "is", "2 or 3"
        CASE 4 TO 5: PRINT value, "is", "4 to 5"
        CASE IS < 7: PRINT value, "is", "< 7"
        CASE IS <= 8: PRINT value, "is", "<= 8"
        CASE IS > 13::::: PRINT value, "is", "> 13"
        CASE IS >= 12:
        ::::
        PRINT value, "is", ">= 12"
        CASE IS <> 10: PRINT value, "is", "<> 10"
        CASE IS = 10: PRINT value, "is", "10"
        CASE ELSE: PRINT "Error"
    END SELECT
END SUB

SUB TESTB(value$)
    SELECT CASE value
        CASE "bing":
        case "bang"

        CASE "boom"::

        CASE "foo"
            PRINT value; "-bar"

        CASE "foo": PRINT "Error"

        CASE ELSE
            PRINT value; "-buz"

    END SELECT
END SUB

SUB TESTC(value%)
    SELECT CASE value
        CASE 9, 11 TO 13, 15: PRINT value
        CASE 16 TO 16, IS = 17: PRINT "B"
        CASE IS > 1, IS < -1
            PRINT "X"
        CASE ELSE
            PRINT "Y"
    END SELECT
END SUB

FOR i = 0 TO 15
    TESTA i
NEXT

YIELD
CLS

TESTB "bing"
TESTB "bang"
TESTB "boom"
TESTB "foo"
TESTB "baz"

YIELD
CLS

TESTC 9
TESTC 10
TESTC 11
TESTC 12
TESTC 13
TESTC 14
TESTC 15
TESTC 16
TESTC 17

YIELD
CLS

FOR i = -2 TO 2
    TESTC i
NEXT