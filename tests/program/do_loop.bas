i = 0

DO WHILE i < 2
    i = i + 1
    PRINT i
LOOP

PRINT

DO UNTIL i = 0
    i = i - 1
    PRINT i
LOOP

PRINT

DO
    i = i + 1
    PRINT i
LOOP WHILE i < 2

PRINT

DO
    i = i - 1
    PRINT i
LOOP UNTIL i = 0

PRINT

DO
    i = i + 1
    PRINT i

    IF i = 2 THEN
        END
    END IF
LOOP