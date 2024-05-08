import random

from games.wordle_cot.instancegenerator import WordleGameInstanceGenerator

GAME_NAME = "wordle_withclue"

if __name__ == "__main__":
    WordleGameInstanceGenerator(GAME_NAME).generate()
