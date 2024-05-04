import random

from games.wordle_nocot.instancegenerator import WordleGameInstanceGenerator

GAME_NAME = "wordle_withclue_nocot"

if __name__ == "__main__":
    WordleGameInstanceGenerator(GAME_NAME).generate()
