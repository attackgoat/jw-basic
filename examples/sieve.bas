REM See: https://en.wikipedia.org/wiki/Byte_Sieve
REM Eratosthenes Sieve Prime Number Program in BASIC
1 SIZE = 8190
2 DIM FLAGS(8191)
3 PRINT "Only 1 iteration"
5 COUNT = 0
6 FOR I = 0 TO SIZE
7 FLAGS (I) = 1
8 NEXT I
9 FOR I = 0 TO SIZE
10 IF FLAGS (I) = 0 THEN
    GOTO 18
   END IF
11 PRIME = I+I + 3
12 K = I + PRIME
13 IF K > SIZE THEN
    GOTO 17
   END IF
14 FLAGS (K) = 0
15 K = K + PRIME
16 GOTO 13
17 COUNT = COUNT + 1
18 NEXT I
19 PRINT COUNT," PRIMES"