total = 0

FOR i = 1 TO 1000
    total = total + 1
    EXIT FOR
    total = 0
NEXT i

PRINT total

DO
    total = total + 1
    EXIT DO
    total = 0
LOOP

PRINT total

DO UNTIL FALSE
    total = total + 1
    EXIT DO
    total = 0
LOOP

PRINT total

DO WHILE TRUE
    total = total + 1
    EXIT DO
    total = 0
LOOP

PRINT total

DO
    total = total + 1
    EXIT DO
    total = 0
LOOP UNTIL FALSE

PRINT total

DO
    total = total + 1
    EXIT DO
    total = 0
LOOP WHILE TRUE

PRINT total

SUB TEST
    total = total + 1
    EXIT SUB
    total = 0
END SUB

TEST

PRINT total

FUNCTION TEST%
    TEST% = total + 1
    EXIT FUNCTION
    TEST% = 0
END FUNCTION

PRINT TEST%()