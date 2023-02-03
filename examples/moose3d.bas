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

CONST KeyUp@ = 38
CONST KeyLeft@ = 37
CONST KeyRight@ = 39
CONST KeyDown@ = 40

CONST MapWidth% = 24
CONST MapHeight% = 24

CONST ScreenWidth% = 160
CONST ScreenHeight% = 96

DIM map[MapWidth, MapHeight]@ = LoadMap()
DIM posX! = 22, posY! = 12
DIM dirX! = -1, dirY! = 0
DIM planeX! = 0, planeY! = 0.66

DIM time% = TIMER%(), oldTime%

Loop:
CLS

FOR x% = 0 TO ScreenWidth - 1
    DIM cameraX! = 2 * x! / ScreenWidth! - 1 ' x-coordinate in camera space
    DIM rayDirX! = dirX + planeX * cameraX
    DIM rayDirY! = dirY + planeY * cameraX

    ' which box of the map we're in
    DIM mapX% = posX%
    DIM mapY% = posY%

    ' length of ray from current position to next x or y-side
    DIM sideDistX!, sideDistY!

    ' length of ray from one x or y-side to next x or y-side
    ' these are derived as:
    ' deltaDistX = sqrt(1 + (rayDirY * rayDirY) / (rayDirX * rayDirX))
    ' deltaDistY = sqrt(1 + (rayDirX * rayDirX) / (rayDirY * rayDirY))
    ' which can be simplified to abs(|rayDir| / rayDirX) and abs(|rayDir| / rayDirY)
    ' where |rayDir| is the length of the vector (rayDirX, rayDirY). Its length,
    ' unlike (dirX, dirY) is not 1, however this does not matter, only the
    ' ratio between deltaDistX and deltaDistY matters, due to the way the DDA
    ' stepping further below works. So the values can be computed as below.
    '  Division through zero is prevented, even though technically that's not
    '  needed in C++ with IEEE 754 floating point values.
    DIM deltaDistX!, deltaDistY!

    IF rayDirX! = 0 THEN
        deltaDistX = 1e30
    ELSE
        deltaDistX = ABS(1 / rayDirX)
    END IF
    

    ' what direction to step in x or y-direction (either +1 or -1)
    DIM stepX!, stepY!

    DIM hit? = FALSE ' was there a wall hit?
    DIM side%        ' was a NS or a EW wall hit?

    ' calculate step and initial sideDist
    IF rayDirX < 0 THEN
        stepX = -1
        sideDistX = (posX - mapX!) * deltaDistX
    ELSE
        stepX = 1;
        sideDistX = (mapX! + 1 - posX) * deltaDistX
    END IF

    IF rayDirY < 0 THEN
        stepY = -1;
        sideDistY = (posY - mapY) * deltaDistY;
    ELSE
        stepY = 1;
        sideDistY = (mapY + 1.0 - posY) * deltaDistY;
    END IF

    ' perform DDA
    WHILE !hit
        ' jump to next map square, either in x-direction, or in y-direction
        IF sideDistX < sideDistY THEN
            sideDistX += deltaDistX;
            mapX += stepX;
            side = 0;
        ELSE
            sideDistY += deltaDistY;
            mapY += stepY;
            side = 1;
        END IF

        ' Check if ray has hit a wall
        IF worldMap[mapX, mapY] > 0 THEN
            hit = 1
        END IF
    WEND

    ' Calculate distance projected on camera direction. This is the shortest distance from the point where the wall is
    ' hit to the camera plane. Euclidean to center camera point would give fisheye effect!
    ' This can be computed as (mapX - posX + (1 - stepX) / 2) / rayDirX for side == 0, or same formula with Y
    ' for size == 1, but can be simplified to the code below thanks to how sideDist and deltaDist are computed:
    ' because they were left scaled to |rayDir|. sideDist is the entire length of the ray above after the multiple
    ' steps, but we subtract deltaDist once because one step more into the wall was taken above.
    DIM perpWallDist!
      
    IF side == 0 THEN
        perpWallDist = (sideDistX - deltaDistX)
    ELSE
        perpWallDist = (sideDistY - deltaDistY)
    END IF

    ' Calculate height of line to draw on screen
    DIM lineHeight% = CINT(ScreenHeight! / perpWallDist)

    ' calculate lowest and highest pixel to fill in current stripe
    DIM drawStart% = -lineHeight / 2 + ScreenHeight / 2

    IF drawStart < 0 THEN
        drawStart = 0
    END IF

    DIM drawEnd% = lineHeight / 2 + ScreenHeight / 2

    IF drawEnd >= ScreenHeight THEN
        drawEnd = ScreenHeight - 1
    END IF

    ' choose wall color
    DIM color@
    SELECT CASE worldMap(mapX, mapY)
        CASE 1
            color = 4   ' red
        CASE 2
            color = 2   ' green
        CASE 3
            color = 1   ' blue
        CASE 4
            color = 7   ' white
        CASE ELSE
            color = 14  ' yellow
    END SELECT

    ' give x and y sides different brightness
    IF side == 1 THEN
        color = color * 2
    END IF

    ' draw the pixels of the stripe as a vertical line
    LINE (x, drawStart) - (x, drawEnd), color
NEXT FOR

' timing for input and FPS counter
oldTime = time;
time = TIMER%()

DIM frameTime! = CFLOAT!(time - oldTime) * 1000 ' frameTime is the time this frame has taken, in seconds

LOCATE 0, 0
PRINT "FPS:", CSTR$(1 / frameTime) ' FPS counter

YIELD

' speed modifiers
DIM moveSpeed! = frameTime * 5  ' the constant value is in squares/second
DIM rotSpeed! = frameTime * 3   ' the constant value is in radians/second

' move forward if no wall in front of you
IF KEY?(KeyUp) THEN
    DIM newX! = posX + dirX * moveSpeed
    IF worldMap[CINT(newX), CINT(posY)] = 0 THEN
        posX = newX
    END IF

    DIM newY! = posY + dirY * moveSpeed
    IF worldMap[CINT(posX), CINT(newY)] = 0 THEN
        posY = newY
    END IF
END IF

' move backwards if no wall behind you
IF KEY?(KeyDown) THEN
    DIM newX! = posX - dirX * moveSpeed
    IF worldMap[CINT(newX) CINT(posY)] = 0 THEN
        posX = newX
    END IF

    DIM newY! = posY - dirY * moveSpeed
    IF worldMap[CINT(posX), CINT(newY)] = 0 THEN
        posY = newY
    END IF
END IF

' rotate to the right
IF KEY?(KeyRight) THEN
    ' both camera direction and camera plane must be rotated
    DIM oldDirX! = dirX

    dirX = dirX * COS(-rotSpeed) - dirY * SIN(-rotSpeed)
    dirY = oldDirX * SIN(-rotSpeed) + dirY * COS(-rotSpeed)

    DIM oldPlaneX! = planeX

    planeX = planeX * COS(-rotSpeed) - planeY * SIN(-rotSpeed)
    planeY = oldPlaneX * SIN(-rotSpeed) + planeY * COS(-rotSpeed)
END IF

' rotate to the left
IF KEY?(KeyLeft) THEN
    ' both camera direction and camera plane must be rotated
    DIM oldDirX! = dirX

    dirX = dirX * COS(rotSpeed) - dirY * SIN(rotSpeed)
    dirY = oldDirX * SIN(rotSpeed) + dirY * COS(rotSpeed)

    DIM oldPlaneX! = planeX

    planeX = planeX * COS(rotSpeed) - planeY * SIN(rotSpeed)
    planeY = oldPlaneX * SIN(rotSpeed) + planeY * COS(rotSpeed)
END IF

GOTO Loop

FUNCTION LoadMap#(MapWidth, MapHeight) ()
    LoadMap[ 0] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    LoadMap[ 1] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[ 2] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[ 3] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[ 4] = [1,0,0,0,0,0,2,2,2,2,2,0,0,0,0,3,0,3,0,3,0,0,0,1]
    LoadMap[ 5] = [1,0,0,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[ 6] = [1,0,0,0,0,0,2,0,0,0,2,0,0,0,0,3,0,0,0,3,0,0,0,1]
    LoadMap[ 7] = [1,0,0,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[ 8] = [1,0,0,0,0,0,2,2,0,2,2,0,0,0,0,3,0,3,0,3,0,0,0,1]
    LoadMap[ 9] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[10] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[11] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[12] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[13] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[14] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[15] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[16] = [1,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[17] = [1,4,0,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[18] = [1,4,0,0,0,0,5,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[19] = [1,4,0,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[20] = [1,4,0,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[21] = [1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[22] = [1,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    LoadMap[23] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
END FUNCTION