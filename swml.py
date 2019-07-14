# -*- coding=utf-8 -*-

import pyxel
import numpy as np
import random as rd


# Define Constant Value
STEP = 0.1
STOP = 0.0
MOVE_LEFT = 0
MOVE_DOWN = 1
MOVE_UP = 2
MOVE_RIGHT = 3
NONE = 4

GAME_TITLE = 0
GAME_PLAYING = 1
GAME_RESULT = 2

class App:

    def __init__(self):
        pyxel.init(128, 160, fps=60)
        self.mainInit()
        self.__gameState = GAME_TITLE
        pyxel.load("swml.pyxel")
        pyxel.run(self.update, self.draw)

    def update(self):
        if self.__gameState == GAME_TITLE:
            self.gameTitle()
        elif self.__gameState == GAME_PLAYING:
            self.gameMain()
        elif self.__gameState == GAME_RESULT:
            self.gameResult()
        else:
            pass

    def gameTitle(self):
        if pyxel.btnp(pyxel.KEY_ENTER):
            self.__gameState = GAME_PLAYING
            self.mainInit()

    def mainInit(self):
        self.__fieldSize = 16
        self.x = 0
        self.y = 0
        self.__moveStep = 0
        self.__moveX = STOP
        self.__moveY = STOP
        self.__moveState = MOVE_RIGHT
        self.__inputKey = NONE
        self.__moveSpeed = 7
        self.__snakeBody = [[0, 0], ]
        self.__score = 0
        self.__foodPos = [rd.randint(0, self.__fieldSize-1), rd.randint(0, self.__fieldSize-1)]

    def gameMain(self):
        self.__moveStep += 1

        # ここで最終入力を記録するが、遷移は更新周期が来てから。
        if pyxel.btnp(pyxel.KEY_LEFT) and self.__moveState != MOVE_RIGHT:
            self.__inputKey = MOVE_LEFT
        elif pyxel.btnp(pyxel.KEY_RIGHT) and self.__moveState != MOVE_LEFT:
            self.__inputKey = MOVE_RIGHT
        elif pyxel.btnp(pyxel.KEY_UP) and self.__moveState != MOVE_DOWN:
            self.__inputKey = MOVE_UP
        elif pyxel.btnp(pyxel.KEY_DOWN) and self.__moveState != MOVE_UP:
            self.__inputKey = MOVE_DOWN

        if pyxel.btnp(pyxel.KEY_BACKSPACE):
            self.__gameState = GAME_TITLE

        # 更新タイミングの判定、更新する場合は更新処理。
        if (self.__moveStep // self.__moveSpeed) >= 1:
            if self.__inputKey == MOVE_LEFT:
                self.__moveState = MOVE_LEFT
            elif self.__inputKey == MOVE_RIGHT:
                self.__moveState = MOVE_RIGHT
            elif self.__inputKey == MOVE_UP:
                self.__moveState = MOVE_UP
            elif self.__inputKey == MOVE_DOWN:
                self.__moveState = MOVE_DOWN
            else:
                pass
            self.__inputKey = NONE

            if self.__moveState == MOVE_LEFT:
                self.__moveX = -1
                self.__moveY = STOP
            elif self.__moveState == MOVE_RIGHT:
                self.__moveX = 1
                self.__moveY = STOP
            elif self.__moveState == MOVE_UP:
                self.__moveX = STOP
                self.__moveY = -1
            elif self.__moveState == MOVE_DOWN:
                self.__moveX = STOP
                self.__moveY = 1

            self.__moveStep %= self.__moveSpeed
            self.x = (self.x+self.__moveX) % self.__fieldSize
            self.y = (self.y+self.__moveY) % self.__fieldSize

            if [self.x, self.y] in self.__snakeBody:
                self.__gameState = GAME_RESULT

            self.__snakeBody.append([self.x, self.y])

            if self.__snakeBody[-1] == self.__foodPos: # foodPosが2行1列に対して、bodyが1行2列なので、inでTrueにならない
                self.nextFood()
                self.__score += 1
            else:
                self.__snakeBody.pop(0)

        else:
            pass

    def gameResult(self):
        if pyxel.btnp(pyxel.KEY_R):
            self.__gameState = GAME_PLAYING
            self.mainInit()
        elif pyxel.btnp(pyxel.KEY_BACKSPACE):
            self.__gameState = GAME_TITLE

    def draw(self):
        if self.__gameState == GAME_TITLE:
            pyxel.cls(0)
            pyxel.text(pyxel.width/2-10, pyxel.height/2, "Snake", col=12)
        elif self.__gameState == GAME_PLAYING:
            pyxel.cls(1)
            pyxel.bltm(0, 0, 0, 0, 0, self.__fieldSize, self.__fieldSize)
            for tempBody in self.__snakeBody:
                pyxel.blt(tempBody[0]*8, tempBody[1]*8, 0, 0, 8, 8, 8, 0)
            pyxel.blt(self.__foodPos[0]*8,
                      self.__foodPos[1]*8, 0, 8, 0, 8, 8, 0)
            pyxel.text(0, 129, "Score: " + str(self.__score), 6)
        elif self.__gameState == GAME_RESULT:
            pyxel.cls(0)
            pyxel.text(pyxel.width/2-15, pyxel.height/2, "Score: " + str(self.__score), 6)
            pyxel.text(10, pyxel.height/2+30, "Restert: R, Title: BackSpace", 6)


    def nextFood(self):
        self.__foodPos = [rd.randint(0, self.__fieldSize-1), rd.randint(0, self.__fieldSize-1)]


if __name__ == "__main__":
    App()
