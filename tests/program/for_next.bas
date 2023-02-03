FOR i = 0 TO 3 STEP 1
    print i
NEXT

color 4

FOR i = 0 TO 3 STEP 2
    print i
NEXT

color 1

FOR i = 4 TO 6
    print i
NEXT

color 2



a = 0



FOR j = 10 TO 6 step -1

    for k = 2 To 4
       FOR l = 9 TO 13 STep 2
           a = a + j + k + l
       NexT l
        
    NEXT
print a, j

NEXT j

color 6

x = 4.2
for z = 1.0 to 9.2 step 2.33
    x = x + z
next

print x
print "OK"