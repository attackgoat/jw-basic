PRINT PEEK(0)
PRINT PEEK?(0)
PRINT PEEK@(0)
PRINT PEEK!(0)
PRINT PEEK%(0)
PRINT PEEK$(0)
PRINT "OK1"

YIELD
CLS

POKE 0, TRUE
PRINT PEEK(0)
PRINT PEEK?(0)
PRINT PEEK@(0)
PRINT PEEK!(0)
PRINT PEEK%(0)
PRINT PEEK$(0)
PRINT "OK2"

YIELD
CLS

POKE 16, 32767
PRINT PEEK(16)
PRINT PEEK?(16)
PRINT PEEK@(16)
PRINT PEEK!(16)
PRINT PEEK%(16)
PRINT PEEK$(16)
PRINT "OK3"

YIELD
CLS

POKE 20, 42@
PRINT PEEK(20)
PRINT PEEK?(20)
PRINT PEEK@(20)
PRINT PEEK!(20)
PRINT PEEK%(20)
PRINT PEEK$(20)
PRINT "OK4"

YIELD
CLS

POKE 20, 42.0
PRINT PEEK(20)
PRINT PEEK?(20)
PRINT PEEK@(20)
PRINT PEEK!(20)
PRINT PEEK%(20)
PRINT PEEK$(20)
PRINT "OK5"

YIELD
CLS

POKE 20, "Hello there"
PRINT PEEK(20)
PRINT PEEK?(20)
PRINT PEEK@(20)
PRINT PEEK!(20)
PRINT PEEK%(20)
PRINT PEEK$(20)
PRINT "OK6"

YIELD
CLS

POKE 24, "   here"
PRINT PEEK$(20)
PRINT "OK7"