import pyxel
import numpy as np
import random as rd
import itertools as it
import pandas as pd
import os
from datetime import datetime as dt
import pickle

'''
TODO:
コード整理（特にクラス化）
追加課題：実際の操作に置き換えたときに、
'''

# 定数定義

# 動作方向
MOVE_LEFT = 0
MOVE_DOWN = 1
MOVE_UP = 2
MOVE_RIGHT = 3
NONE = 4

# ゲーム状態
GAME_TITLE = 0
GAME_PLAYING = 1
GAME_RESULT = 2
GAME_RANK = 3
GAME_SETTING = 4

# 壁サイズ（サイズ分動作領域をシフトする）
WALL_SHIFT = 8

# 移動速度
# 初期値
MOVE_SPEED_DEFAULT = 10
# 最速値
MOVE_SPEED_FAST = 3
# 速度アップ閾値
MOVE_SPEED_UP_TH = 5


# 実行ファイルディレクトリ
FILE_DIR = os.path.dirname(__file__)
RESULT_FILE_PATH = "/data/result.csv"

# Reward Setting
REWARD_GET_FOOD = 1000
REWARD_TIME = 0
REWARD_END = -100

class SnakeGameApp:

    def __init__(self):
        f = open('bestResult.txt','rb')
        self.__playData = np.array(pickle.load(f))
        f.close()
        self.__fieldSize = 16
        self.__index = -1
        pyxel.init(144, 160, fps=10)
        pyxel.load(FILE_DIR + "/AomushI.pyxres")
    
    def run(self):
        pyxel.run(self.update, self.draw)

    def update(self):
        self.__index += 1
        pass

    def draw(self):
        pyxel.cls(1)
        # 壁
        for i in range(self.__fieldSize+2):
            for j in range(self.__fieldSize+2):
                if (i == 0) or (i == self.__fieldSize+1) or (j == 0) or (j == self.__fieldSize+1):
                    pyxel.blt(i*8, j*8, 0, 8, 0, 8, 8)

        # 床
        pyxel.bltm(0+WALL_SHIFT, 0+WALL_SHIFT, 0, 0, 0,
                    self.__fieldSize, self.__fieldSize)

        # snake
        temp = np.where(self.__playData[self.__index]==1)
        for index in range(len(temp[0])):
            pyxel.blt(temp[0][index]*8+WALL_SHIFT,
                        temp[1][index]*8+WALL_SHIFT, 0, 8, 8, 8, 8, 0)
        #頭の位置も区別したほうがよさそう。
        
        temp = np.where(self.__playData[self.__index]==2)
        for index in range(len(temp[0])):
            pyxel.blt(temp[0][index]*8+WALL_SHIFT, temp[1][index]
                        * 8+WALL_SHIFT, 0, 0, 8, 8, 8, 0)

        # food
        temp = np.where(self.__playData[self.__index]==3)
        for index in range(len(temp[0])):
            pyxel.blt(temp[0][index]*8+WALL_SHIFT,
                        temp[1][index]*8+WALL_SHIFT, 0, 0, 16, 8, 8, 0)
        # pyxel.text(0, 145, "Score: " + str(self.__score), 6)
        # pyxel.text(0, 153, "Speed: " + str(11 - self.__moveSpeed), 6)


if __name__ == "__main__":
    SG = SnakeGameApp()
    SG.run()