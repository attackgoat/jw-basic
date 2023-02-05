' Copyright (c) 2004-2019, Lode Vandevenne
' 
' All rights reserved.
' 
' Redistribution and use in source and binary forms, with or without modification, are permitted
' provided that the following conditions are met:
' 
'     * Redistributions of source code must retain the above copyright notice, this list of
'       conditions and the following disclaimer.
'     * Redistributions in binary form must reproduce the above copyright notice, this list of
'       conditions and the following disclaimer in the documentation and/or other materials provided
'       with the distribution.
' 
' THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
' "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
' LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
' A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
' CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
' EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
' PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
' PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
' LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
' NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
' SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

KeyCodeLeft = 1@
KeyCodeRight = 2@
KeyCodeUp = 3@
KeyCodeDown = 4@
MapWidth = 24
MapHeight = 24
ScreenWidth = 160
ScreenHeight = 96

COLOR 13@

DIM map@(MapWidth - 1, MapHeight - 1)

FOR y = 0 TO MapHeight - 1
        map(0, y) = 1@
        map(MapWidth - 1, y) = 1@
NEXT
FOR x = 0 to MapWidth - 1
        map(x, 0) = 1@
        map(x, MapHeight - 1) = 1@
NEXT

FOR y = 4 TO 8
    FOR x = 6 TO 10
        map(x, y) = 2@
    NEXT
NEXT
FOR y = 5 TO 7
    FOR x = 7 TO 9
        map(x, y) = 0@
    NEXT
NEXT

map(8, 8) = 0@

map(15, 4) = 3@
map(17, 4) = 3@
map(19, 4) = 3@
map(15, 6) = 3@
map(19, 6) = 3@
map(15, 8) = 3@
map(17, 8) = 3@
map(19, 8) = 3@

FOR y = 16 TO 22
        map(1, y) = 4@
        map(8, y) = 4@
NEXT
FOR x = 1 to 8
        map(x, 16) = 4@
        map(x, 22) = 4@
NEXT

FOR y = 16 TO 20
        map(3, y) = 4@
        map(8, y) = 4@
NEXT
FOR x = 3 to 8
        map(x, 20) = 4@
NEXT

map(3, 18) = 0@
map(8, 21) = 0@

map(6, 18) = 5@

' This is the map we made:
' [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,2,2,2,2,2,0,0,0,0,3,0,3,0,3,0,0,0,1],
' [1,0,0,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,2,0,0,0,2,0,0,0,0,3,0,0,0,3,0,0,0,1],
' [1,0,0,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,2,2,0,2,2,0,0,0,0,3,0,3,0,3,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,4,0,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,4,0,0,0,0,5,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,4,0,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,4,0,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
' [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

posX = 22.0
posY = 12.0
dirX = -1.0
dirY = 0.0
planeX = 0.0
planeY = 0.66

WHILE TRUE
    CLS
    startTime = TIMER()

    RECTANGLE (0, 0) - (ScreenWidth - 1, ScreenHeight / 2), 24@, TRUE
    RECTANGLE (0, ScreenHeight / 2 + 1) - (ScreenWidth - 1, ScreenHeight - 1), 20@, TRUE

    FOR x = 0 TO ScreenWidth - 1
        cameraX = 2.0 * CFLOAT(x) / CFLOAT(ScreenWidth) - 1.0

        rayDirX = dirX + planeX * cameraX
        rayDirY = dirY + planeY * cameraX

        DIM deltaDistX!
        IF rayDirX = 0.0 THEN
            deltaDistX = 1e30
        ELSE THEN
            deltaDistX = ABS(1.0 / rayDirX)
        END IF

        DIM deltaDistY!
        IF rayDirY = 0.0 THEN
            deltaDistY = 1e30
        ELSE THEN
            deltaDistY = ABS(1.0 / rayDirY)
        END IF

        mapX = CINT(posX)
        mapY = CINT(posY)

        DIM stepX, stepY, sideDistX!, sideDistY!
        IF rayDirX < 0.0 THEN
            stepX = -1
            sideDistX = (posX - CFLOAT(mapX)) * deltaDistX
        ELSE THEN
            stepX = 1
            sideDistX = (CFLOAT(mapX) + 1.0 - posX) * deltaDistX
        END IF

        IF rayDirY < 0.0 THEN
            stepY = -1
            sideDistY = (posY - CFLOAT(mapY)) * deltaDistY
        ELSE THEN
            stepY = 1
            sideDistY = (CFLOAT(mapY) + 1.0 - posY) * deltaDistY
        END IF

        DIM side?, hitColor@
        hit? = FALSE

        WHILE NOT hit
            ' jump to next map square, either in x-direction, or in y-direction
            IF sideDistX < sideDistY THEN
                sideDistX = sideDistX + deltaDistX
                mapX = mapX + stepX
                side = FALSE
            ELSE THEN
                sideDistY = sideDistY + deltaDistY
                mapY = mapY + stepY
                side = TRUE
            END IF

            ' Check if ray has hit a wall
            hitColor = map(mapX, mapY)
            IF hitColor > 0@ THEN
                hit = TRUE
            END IF
        WEND

        DIM perpWallDist!

        IF NOT side THEN
            perpWallDist = sideDistX - deltaDistX
        ELSE THEN
            perpWallDist = sideDistY - deltaDistY
        END IF

        lineHeight = CINT(CFLOAT(ScreenHeight) / perpWallDist)
        drawStart = -lineHeight / 2 + ScreenHeight / 2

        IF drawStart < 0 THEN
            drawStart = 0
        END IF

        drawEnd = lineHeight / 2 + ScreenHeight / 2

        IF drawEnd >= ScreenHeight THEN
            drawEnd = ScreenHeight - 1
        END IF

        IF side THEN
           hitColor = hitColor + 8@
        END IF

        LINE (x, drawStart) - (x, drawEnd), hitColor
    NEXT

    RECTANGLE (0, 0) - (50, 6), &HFF@, TRUE

    endTime = TIMER()
    frameTime! = CFLOAT(endTime - startTime) / 1000000.0
    fps = CINT(1.0 / frameTime)

    LOCATE 0, 0
    PRINT "FPS:", fps

    YIELD

    moveSpeed = frameTime * 18.0
    rotSpeed = frameTime * 28.0

    IF KEYDOWN(KeyCodeUp) THEN
        newX = posX + dirX * moveSpeed
        IF map(CINT(newX), CINT(posY)) = 0@ THEN
            posX = newX
        END IF

        newY = posY + dirY * moveSpeed
        IF map(CINT(posX), CINT(newY)) = 0@ THEN
            posY = newY
        END IF
    END IF

    IF KEYDOWN(KeyCodeDown) THEN
        newX = posX - dirX * moveSpeed
        IF map(CINT(newX), CINT(posY)) = 0@ THEN
            posX = newX
        END IF

        newY = posY - dirY * moveSpeed
        IF map(CINT(posX), CINT(newY)) = 0@ THEN
            posY = newY
        END IF
    END IF

    IF KEYDOWN(KeyCodeLeft) THEN
        oldDirX = dirX
        dirX = dirX * COS(-rotSpeed) - dirY * SIN(-rotSpeed)
        dirY = oldDirX * SIN(-rotSpeed) + dirY * COS(-rotSpeed)

        oldPlaneX = planeX
        planeX = planeX * COS(-rotSpeed) - planeY * SIN(-rotSpeed)
        planeY = oldPlaneX * SIN(-rotSpeed) + planeY * COS(-rotSpeed)
    END IF

    IF KEYDOWN(KeyCodeRight) THEN
        oldDirX = dirX
        dirX = dirX * COS(rotSpeed) - dirY * SIN(rotSpeed)
        dirY = oldDirX * SIN(rotSpeed) + dirY * COS(rotSpeed)

        oldPlaneX = planeX
        planeX = planeX * COS(rotSpeed) - planeY * SIN(rotSpeed)
        planeY = oldPlaneX * SIN(rotSpeed) + planeY * COS(rotSpeed)
    END IF
WEND