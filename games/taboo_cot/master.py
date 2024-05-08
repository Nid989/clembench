from typing import Dict, Tuple, List
import re
import json

import numpy as np

from backends import Model
from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemgame import get_logger
from clemgame import file_utils, string_utils

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords', quiet=True)
EN_STOPWORDS = stopwords.words('english')

EN_STEMMER = SnowballStemmer("english")

GAME_NAME = "taboo_cot"

logger = get_logger(__name__)


def convert_to_json(response: str) -> Dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        return None

class WordGuesser(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages, turn_idx):
        # mock response
        return f'GUESS: Pear'


class WordDescriber(Player):

    def __init__(self, model: Model, max_turns):
        super().__init__(model)
        self.max_turns = max_turns

    def _custom_response(self, messages, turn_idx):
        if turn_idx < self.max_turns:
            return "CLUE: This is a difficult word to describe."
        if turn_idx >= self.max_turns:
            raise Exception("We should not be here...")

   
def check_clue(utterance: str, target_word: str, related_words: List[str],
               stemmer=EN_STEMMER) -> List[Dict]:
    response_dict = convert_to_json(utterance)
    clue = response_dict['CLUE'].strip()
    clue = string_utils.remove_punctuation(clue)
    clue = clue.split(" ")
    clue_words = [clue_word for clue_word in clue if clue_word not in EN_STOPWORDS]
    clue_word_stems = [stemmer.stem(clue_word) for clue_word in clue_words]
    errors = []
    target_word_stem = stemmer.stem(target_word)
    related_word_stems = [stemmer.stem(related_word) for related_word in related_words]
        
    for clue_word, clue_word_stem in zip(clue_words, clue_word_stems):
        if target_word_stem == clue_word_stem:
            errors.append({
                "message": f"Target word '{target_word}' (stem={target_word_stem}) "
                           f"is similar to clue word '{clue_word}' (stem={clue_word_stem})",
                "type": 0
            })
        for related_word, related_word_stem in zip(related_words, related_word_stems):
            if related_word_stem == clue_word_stem:
                errors.append({
                    "message": f"Related word '{related_word}' (stem={related_word_stem}) "
                               f"is similar to clue word '{clue_word}' (stem={clue_word_stem})",
                    "type": 1
                })
    return errors


class TabooCOT(DialogueGameMaster):
    """
    This class implements a taboo game in which player A (the WordDescriber) is describing a 
    target word that player B (the WordGuesser) needs to guess. Player A cannot say or use 
    target word or related words for the `CLUE`, but can utilize them for generating `REASON` 
    (during reasoning). Morphology is checked in check_clue(). 
    """ 
    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.max_turns: int = experiment["max_turns"]
        self.describer_initial_prompt = self.experiment["describer_initial_prompt"]
        self.guesser_initial_prompt = self.experiment["guesser_initial_prompt"]

    def _on_setup(self, **game_instance):
        logger.info("_on_setup")
        self.game_instance = game_instance

        self.target_word = game_instance["target_word"]
        self.related_words = game_instance["related_word"]

        self.describer_initial_prompt = self.describer_initial_prompt.replace("$TARGET_WORD$", self.target_word)
        rel_words = f"- {self.related_words[0]}\n- {self.related_words[1]}\n- {self.related_words[2]}"
        self.describer_initial_prompt = self.describer_initial_prompt.replace("$REL_WORD$", rel_words)
        self.describer_initial_prompt = self.describer_initial_prompt.replace("$N$", str(self.max_turns))
        self.guesser_initial_prompt = self.guesser_initial_prompt.replace("$N$", str(self.max_turns))

        self.describer = WordDescriber(self.player_models[0], self.max_turns)
        self.guesser = WordGuesser(self.player_models[1])

        self.add_player(self.describer)
        self.add_player(self.guesser)

        self.invalid_response = False
        self.clue_error = None
        self.guess_word = None

    def _on_before_game(self):
        self.add_user_message(self.describer, self.describer_initial_prompt)
        self.add_user_message(self.guesser, self.guesser_initial_prompt)

    def _does_game_proceed(self):
        """
        Proceed as long as the word has not been guessed and the maximum number of 
        turns has not been reached.
        """
        if self.invalid_response:
            self.log_to_self("invalid format", "abort game")
            return False
        if self.clue_error is not None:
            error_type = self.clue_error["type"]
            if error_type == 0:
                self.log_to_self("invalid clue", "clue contains target word")
            if error_type == 1:
                self.log_to_self("invalid clue", "clue contains related word")
            return False # stop game if clue is wrong (for now)
        if self.guess_word == self.target_word:
            self.log_to_self("correct guess", self.guess_word)
            return False
        if self.guess_word is not None:
            if EN_STEMMER.stem(self.guess_word) == EN_STEMMER.stem(self.target_word):
                self.log_to_self("correct guess", self.guess_word)
                return False 
        if self.current_turn >= self.max_turns:
            self.log_to_self("max turns reached", str(self.max_turns))
            return False
        return True
    
    def _validate_JSON(self, utterance: str, valid_fields: List[str]):
        response = convert_to_json(utterance)
        if response is None:
            return False
        return all(field in response for field in valid_fields)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if player == self.guesser:
            if not self._validate_JSON(utterance, ["Let's think step by step", 'GUESS']):
                self.invalid_response = True
                return False
        if player == self.describer:
            if not self._validate_JSON(utterance, ["Let's think step by step", 'CLUE']):
                self.invalid_response = True
                return False
            errors = check_clue(utterance, self.target_word, self.related_words)
            if errors:
                self.clue_error = errors[0]
                return False
        self.log_to_self("valid format", "continue")
        return True
    
    # NOTE: I think _on_parse_response should be also validate response and not just log the "assumed" correct format data.
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        if player == self.guesser:
            utterance = convert_to_json(utterance)['GUESS']
            utterance = utterance.strip()
            utterance = utterance.lower()
            utterance = string_utils.remove_punctuation(utterance)
            self.guess_word = utterance.lower()
            self.log_to_self("guess", self.guess_word)
        if player == self.describer:
            utterance = convert_to_json(utterance)['CLUE']
            utterance = utterance.strip()
            utterance = string_utils.remove_punctuation(utterance)
            self.log_to_self("clue", utterance)
        return utterance, True
    
    def _after_add_player_response(self, player: Player, utterance: str):
        """
        Add the utterance to other player's history, 
        using the add_user_message(other_player, utterance) method.
        """
        if player == self.describer:
            utterance = utterance.strip()
            utterance = f"CLUE: {utterance}."
            self.add_user_message(self.guesser, utterance)
        if player == self.guesser:
            if self.guess_word != self.target_word:
                utterance = f"GUESS: {self.guess_word}."
                self.add_user_message(self.describer, utterance)
                
    def _on_before_turn(self, turn_idx: int):
        if turn_idx == 0:
            self.log_message_to(self.guesser, self.guesser_initial_prompt)

class TabooScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """ Episode level scores"""
        turn_scores = []
        prev_guess = None
        prev_guess_counter = 0
        prev_clue = None
        prev_clue_counter = 0
        invalid_response = False  # Note: This only takes into consideration that both players were compliant or not
        guesser_won = False
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"guess": None, "clue": None, "request_count": 1}

            for event in turn:
                action = event["action"]
                if action["type"] == "invalid format":
                    invalid_response = True
                if action["type"] == "guess":
                    turn_score["guess"] = action["content"]
                if action["type"] == "clue":
                    turn_score["clue"] = action["content"]
                if action["type"] == "correct guess":
                    guesser_won = True

            if invalid_response:
                turn_score["violated_request_count"] = 1
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["violated_request_count"] = 0
                turn_score["parsed_request_count"] = 1

            if turn_score["guess"] is not None and turn_score["guess"] == prev_guess:  # might be None, if clue is wrong
                prev_guess_counter += 1
            if turn_score["clue"] is not None and turn_score["clue"] == prev_clue:
                prev_clue_counter += 1
            self.log_turn_score(turn_idx, 'Accuracy', 1 if guesser_won else 0)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, turn_score["request_count"])
            prev_guess = turn_score["guess"]
            prev_clue = turn_score["clue"]
            turn_scores.append(turn_score)

        violated_request_count = sum([turn["violated_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)

        parsed_request_count = sum([turn["parsed_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum([turn["request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)

        self.log_episode_score(METRIC_REQUEST_SUCCESS, parsed_request_count / request_count)
        # checking the last guess (could be None) is ok,
        # b.c. the game ends only successfully, when there is a correct guess

        # Common metrics
        if invalid_response:  # whether a violation of the game rules happened (response not parsable)
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(BENCH_SCORE, np.nan)  # metric not applicable
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            if guesser_won:
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)
                self.log_episode_score(BENCH_SCORE, 100 / len(turn_scores))  # how early the guesser found the word
            else:
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
                self.log_episode_score(BENCH_SCORE, 0)  # word not found

        # Game-specific metrics
        # How often the Guesser repeated a guess
        self.log_episode_score('Repetition-Guesser', prev_guess_counter)
        # How often the Describer repeated itself
        self.log_episode_score('Repetition-Describer', prev_clue_counter)
        # this might require a side-loop between describer and GM (game should not continue with Guesser)
        # self.log_episode_score('Rule-following', ...)

class TabooCOTBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Taboo game between two agents where one has to describe a word for the other to guess."
    
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return TabooCOT(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return TabooScorer(experiment, game_instance)

def main():
    # select one experiment and instance 
    experiments = file_utils.load_json("in/instances.json", "taboo")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = TabooCOT(experiment_1, ["mock", "mock"])
    master.setup(**game_1)
    master.play()

if __name__ == "__main__":
    main()