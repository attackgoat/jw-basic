REM See: https://en.wikipedia.org/wiki/Creative_Computing_Benchmark
10 ' Ahl's Simple Benchmark
20 FOR N=1 TO 100: A=N
30 FOR I=1 TO 10
40 A=SQR(A): R=R+RND(1)
50 NEXT I
60 FOR I=1 TO 10
70 A=A^2: R=R+RND(1)
80 NEXT I
90 S=S+A: NEXT N
100 PRINT ABS(1010-S/5)
110 PRINT ABS(1000-R)