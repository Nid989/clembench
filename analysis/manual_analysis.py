import os
import re
import itertools
import functools
import collections
import pandas as pd
from typing import Dict, List, Tuple, Union

from constants import (
    path_to_results_dir, game_level_types, path_to_outputs_dir
)
from analysis.utils import (
    load_from_json, load_from_yaml, merge_dfs_on_columns, format_cllm_pair, 
    save_to_excel, upload_to_s3
)
 
# Extract the initial turn-wise interactions b/w game-master (gm) and player 1 (player1) for Taboo game
class taboogame_init_data_formatter:
    """Testing for World Knowledge"""
    def __init__(self, cllm_name: str, level: str="1_medium_en", with_cot: bool=False):
        self.clemgame = "taboo" if not with_cot else "taboo_cot"
        self.cllm_name = cllm_name
        self.level = level
        self.path_to_model_dir = os.path.join(path_to_results_dir, format_cllm_pair(self.cllm_name))
        self._extract_init_ep_interactions_player1_all()

    def _extract_init_ep_interactions_player1_all(self):
        all_init_interactions_player1 = collections.defaultdict(list)
        # sorted episodes based on the cardinality and not `string` type
        episodes = [filename for filename in os.listdir(os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/")) if filename.startswith("episode")]
        episodes = sorted(episodes, key=lambda x: int(x.split("_")[-1]))
        for index, episode in enumerate(episodes):
            # load interactions.json file (incl. turn-wise interactions b/w gm, player1, player2)
            path_to_ep_interactions_file = os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/{episode}/interactions.json")
            if os.path.exists(path_to_ep_interactions_file):
                interactions_data = load_from_json(path_to_ep_interactions_file)
                init_interactions_player1 = self._extract_init_ep_interactions_player1(interactions_data)
                _ = [all_init_interactions_player1[key].append(value) for key, value in init_interactions_player1.items()]
        self.all_init_interactions_player1_df = pd.DataFrame(all_init_interactions_player1)
        self.all_init_interactions_player1_df[f"score_{self.cllm_name}"] = ''

    def _extract_init_ep_interactions_player1(self, ep_interactions_data: Dict):
        def extract_w_key_value_pair(turns_data: List[Dict[str, str]], key_value_pair: Tuple[str]):
            return next((index, item) for index, item in enumerate(turns_data) if item.get(key_value_pair[0]) == key_value_pair[1])
        def extract_guess_and_rel_words(content: str): # This should remain constant for normal and COT based player1 prompts
            pattern = re.compile(r'guess:(.*?)\bImportant\b', re.DOTALL)
            match = pattern.search(content)
            output = ""
            if match:
                output = match.group(0)
            assert output != "", "no match found"
            return output.replace("Important", "").strip().replace("\n\n", " ").replace("\n", " ")
        gm_to_player1 = extract_w_key_value_pair(ep_interactions_data["turns"][0], ("to", "Player 1"))[-1]
        init_response_idx, player1_to_gm = extract_w_key_value_pair(ep_interactions_data["turns"][0], ("from", "Player 1"))
        # extract the validity-check information performed by game-master (gm) based on the initial response by the player1
        validition_category = ep_interactions_data["turns"][0][init_response_idx+1]["action"]["content"].strip()
        gm_prompt_instruction = extract_guess_and_rel_words(gm_to_player1["action"]["content"])
        player1_response = player1_to_gm["action"]["content"]
        return {
            "prompt_instruction": gm_prompt_instruction,
            f"player1_response_{self.cllm_name}": player1_response,
            f"validition_category_{self.cllm_name}": validition_category
        }
    
# Extract the turn-wise interactions b/w game-master (gm) and player 1 (player1) for Wordle game (strictly `withclue`)
class wordle_withclue_data_formatter:
    """Testing for World Knowledge (initial utterance), Situation Modeling (utterances over turns)"""
    def __init__(self, cllm_name: str, level: str="1_medium_frequency_words_clue_no_critic", with_cot: bool=False):
        self.clemgame = "wordle_withclue" if not with_cot else "wordle_withclue_cot"
        self.cllm_name = cllm_name
        self.level = level
        self.path_to_model_dir = os.path.join(path_to_results_dir, format_cllm_pair(self.cllm_name))

        # extract all turn-wise conversation (utterance) b/w game-master (gm) and player 1 (either-ways)
        self._extract_ep_interactions_player1_all()

    def _extract_ep_interactions_player1_all(self):
        all_interactions_player1 = collections.defaultdict(list)
        self.all_paths_to_ep_interactions_file = []
        # sorted episodes based on the cardinality and not `string` type
        episodes = [filename for filename in os.listdir(os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/")) if filename.startswith("episode")]
        episodes = sorted(episodes, key=lambda x: int(x.split("_")[-1]))
        for index, episode in enumerate(episodes):
            # load interactions.json file (incl. turn-wise interactions b/w gm, player1)
            path_to_ep_interactions_file = os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/{episode}/interactions.json")
            self.all_paths_to_ep_interactions_file.append(path_to_ep_interactions_file)
            # print(path_to_ep_interactions_file)
            if os.path.exists(path_to_ep_interactions_file):
                interactions_data = load_from_json(path_to_ep_interactions_file)
                interactions_player1 = self._extract_ep_interactions_player1(interactions_data)
                _ = [all_interactions_player1[key].append(value) for key, value in interactions_player1.items()]
        self.all_interactions_player1_df = pd.DataFrame(all_interactions_player1)
        self.all_interactions_player1_df[f"score_{self.cllm_name}"] = ''

    def _extract_ep_interactions_player1(self, ep_interactions_data: Dict):
        def extract_w_key_value_pair(turns_data: List[Dict[str, str]], key_value_pair: Tuple[str]=None, **kwargs):
            if kwargs and key_value_pair is None:
                return next((index, item) for index, item in enumerate(turns_data) if item.get(kwargs["key_value_pair1"][0]) == kwargs["key_value_pair1"][1] and \
                            item.get(kwargs["key_value_pair2"][0]) == kwargs["key_value_pair2"][1])
            else:
                return next((index, item) for index, item in enumerate(turns_data) if item.get(key_value_pair[0]) == key_value_pair[1])
        def extract_clue(content: str):
            # prompt instruction provided by GM to Player 1 for wordlewithclue explicitly
            # contains clue at the end (last line)
            return content.strip().split("\n")[-1].strip()
        # gm_prompt_instruction (i.e. clue)
        # player1 response + gm guess-feedback (per turn)
        # target word
        target_word = None
        ep_utterances_processed = []
        total_turns = len(ep_interactions_data["turns"])
        for turn_id in range(total_turns):
            if turn_id == total_turns-1: # collect `data_for_computation` info @ final turn
                    target_word = ep_interactions_data["turns"][turn_id][-1]["action"]["data_for_computation"]["target_word"]
            ep_utterances_processed.append(f"### turn: {turn_id} ###")
            if turn_id == 0:
                # when turn_id == 0
                # extract target_word_clue + Player 1 -> GM response w.r.t to target_word_clue
                gm_to_player1 = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("to", "Player 1"))[-1]
                _, player1_to_gm = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("from", "Player 1"))
                gm_prompt_instruction = extract_clue(gm_to_player1["action"]["content"])
                player1_response = player1_to_gm["action"]["content"].strip()
                # attaching player1 response specific to intial prompt-instruction
                ep_utterances_processed.append(f"## Player 1 response ##\n{player1_response}")
            else:
                # when turn_id != 0 i.e. (1->5)
                _, gm_to_player1 = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("from", "GM"))
                _, player1_to_gm = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("from", "Player 1"))
                gm_feedback = gm_to_player1["action"]["content"].strip()
                player1_response = player1_to_gm["action"]["content"].strip()
                # attaching gm feedback to player1 response at `turn_id-1`, and player1 response to previous guess-feedback
                ep_utterances_processed.append(f"## GM feedback ##\n{gm_feedback}")
                ep_utterances_processed.append(f"## Player 1 response ##\n{player1_response}")

        return {
            "prompt_instruction": gm_prompt_instruction,
            "target_word": target_word,
            f"player1_gm_utterances_{self.cllm_name}": "\n".join(ep_utterances_processed),
        }

# Extract the interactions b/w game-master (gm) and player 1 (player1) for Reference game
class referencegame_data_formatter:
    """Testing for World Knowledge"""
    def __init__(self, cllm_name: str, level: str="0_line_grids_rows", with_cot: bool=False):
        self.clemgame = "referencegame" if not with_cot else "referencegame_cot"
        self.cllm_name = cllm_name
        self.level = level
        self.path_to_model_dir = os.path.join(path_to_results_dir, format_cllm_pair(self.cllm_name))

        # extract the utterance b/w game-master (gm) and player 1 (specifiying the grids to `player_1` and corresponding response.)
        self._extract_ep_interactions_player1_all()

    def _extract_ep_interactions_player1_all(self):
        all_interactions_player1 = collections.defaultdict(list)
        self.all_paths_to_ep_interactions_file = []
        # sorted episodes based on the cardinality and not `string` type
        episodes = [filename for filename in os.listdir(os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/")) if filename.startswith("episode")]
        episodes = sorted(episodes, key=lambda x: int(x.split("_")[-1]))
        for index, episode in enumerate(episodes):
            # load requests.json file
            # path_to_ep_requests_file = os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/{episode}/requests.json")
            path_to_ep_interactions_file = os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/{episode}/interactions.json")
            path_to_ep_instance_file = os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/{episode}/instance.json")
            if os.path.exists(path_to_ep_interactions_file):
                interactions_data = load_from_json(path_to_ep_interactions_file)
                instance_data = load_from_json(path_to_ep_instance_file)
                interactions_player1 = self._extract_ep_interactions_player1(interactions_data, instance_data)
                _ = [all_interactions_player1[key].append(value) for key, value in interactions_player1.items()]
        self.all_interactions_player1_df = pd.DataFrame(all_interactions_player1)
        # drop duplicates, maintains the grid layout for player 1 unchanged, while introducing randomness to the grids of player 2.
        self.all_interactions_player1_df.drop_duplicates(subset=["prompt_instruction"], keep="first", inplace=True)
        self.all_interactions_player1_df[f"score_{self.cllm_name}"] = ''
            
    def _extract_ep_interactions_player1(self, ep_interactions_data: Dict, ep_instance_data: Dict):
        def extract_w_key_value_pair(turns_data: List[Dict[str, str]], key_value_pair_1: Tuple[str], key_value_pair_2: Tuple[str]):
            return next((index, item) for index, item in enumerate(turns_data) if item.get(key_value_pair_1[0]) == key_value_pair_1[1] and item.get(key_value_pair_2[0]) == key_value_pair_2[1])
        def format_prompt_instruction(ep_instance_data: Dict[str, Union[int, str]]):
            player_1_target_grid = ep_instance_data["player_1_target_grid"]
            player_1_second_grid = ep_instance_data["player_1_second_grid"]
            player_1_third_grid = ep_instance_data["player_1_third_grid"]
            return "Target grid: \n\n{} \n\nDistractor grid 1: \n\n{} \n\nDistractor grid 2: \n\n{}".format(
                player_1_target_grid,
                player_1_second_grid,
                player_1_third_grid
            )

        prompt_instruction = format_prompt_instruction(ep_instance_data)
        player1_response = extract_w_key_value_pair(ep_interactions_data["turns"][0], ("from", "GM"), ("to", "GM"))[-1]["action"]["content"]
        return {
            "prompt_instruction": prompt_instruction,
            f"player1_response_{self.cllm_name}": player1_response
        }

class imagegame_data_formatter:
    """Testing for Situation Modeling"""
    def __init__(self, cllm_name: str, level: str="0_line_grids_rows", with_cot: bool=False):
        self.clemgame = "imagegame" if not with_cot else "imagegame_cot"
        self.cllm_name = cllm_name
        self.level = level
        self.path_to_model_dir = os.path.join(path_to_results_dir, format_cllm_pair(self.cllm_name))

        # extract the utterance b/w game-master (gm) and player 1 (specifiying the grids to `player_1` and corresponding response.)
        self._extract_ep_interactions_player2_all()

    def _extract_ep_interactions_player2_all(self):
        all_interactions_player2 = collections.defaultdict(list)
        # sorted episodes based on the cardinality and not `string` type
        episodes = [filename for filename in os.listdir(os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/")) if filename.startswith("episode")]
        episodes = sorted(episodes, key=lambda x: int(x.split("_")[-1]))
        for index, episode in enumerate(episodes):
            # load interactions.json file (incl. turn-wise interactions b/w gm, player1, player2)
            path_to_ep_interactions_file = os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/{episode}/interactions.json")
            path_to_ep_instance_file = os.path.join(self.path_to_model_dir, f"./{self.clemgame}/{self.level}/{episode}/instance.json")
            
            if os.path.exists(path_to_ep_interactions_file):
                interactions_data = load_from_json(path_to_ep_interactions_file)
                instance_data = load_from_json(path_to_ep_instance_file)
                interactions_player2 = self._extract_ep_interactions_player2(interactions_data, instance_data)
                _ = [all_interactions_player2[key].append(value) for key, value in interactions_player2.items()]
        self.all_interactions_player2_df = pd.DataFrame(all_interactions_player2)
        self.all_interactions_player2_df[f"score_{self.cllm_name}"] = ''
        
    def _extract_ep_interactions_player2(self, ep_interactions_data: Dict, ep_instance_data: Dict):
        def extract_w_key_value_pair(turns_data: List[Dict[str, str]], key_value_pair: Tuple[str]):
            return next((index, item) for index, item in enumerate(turns_data) if item.get(key_value_pair[0]) == key_value_pair[1])
        target_grid = ep_instance_data["target_grid"]
        ep_utterances_processed = []
        total_turns = len(ep_interactions_data["turns"])
        for turn_id in range(total_turns):
            ep_utterances_processed.append(f"### turn: {turn_id} ###")
            if turn_id == total_turns-1: # TODO: cross-check if final-turns usually have gm->player1 utterances
                continue
            if turn_id == 0:
                gm_to_player2 = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("to", "Player 2"))
                _, player2_to_gm = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("from", "Player 2"))
                # extract the first drawing instruction given by gm (game-master) to player2 
                gm_instruction = gm_to_player2[-1]["action"]["content"].split(".")[-1].strip() 
                player2_response = player2_to_gm["action"]["content"].strip()
                ep_utterances_processed.append(f"## GM instruction ##\n{gm_instruction}")
                ep_utterances_processed.append(f"## Player 2 response ##\n{player2_response}")
            else:
                _, gm_to_player2 = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("to", "Player 2"))
                _, player2_to_gm = extract_w_key_value_pair(ep_interactions_data["turns"][turn_id], ("from", "Player 2"))
                # extract the first drawing instruction given by gm (game-master) to player2 
                gm_instruction = gm_to_player2["action"]["content"].strip() 
                player2_response = player2_to_gm["action"]["content"].strip()
                ep_utterances_processed.append(f"## GM instruction ##\n{gm_instruction}")
                ep_utterances_processed.append(f"## Player 2 response ##\n{player2_response}")
        return {
            "prompt_instruction": target_grid,
            f"player2_gm_utterances_{self.cllm_name}": "\n".join(ep_utterances_processed),
        }
    
class clembench_emergence_annotation_extractor:
    def __init__(self, clemgame: str="taboo", level: str="0_high_en", models: List[str]=None, experiment_name: str=None, with_cot: bool=False, **kwargs):
        self.clemgame = clemgame
        self.level = level
        self.models = models
        self.experiment_name = experiment_name
        self.with_cot = with_cot
        self._setup()

    def _setup(self):
        # check for value errors for input attributes
        if self.clemgame not in ["taboo", "wordle_withclue", "referencegame", "imagegame"]:
            raise ValueError("The input attribute clemgame should be one of ['taboo', 'wordle_withclue', 'referencegame'], found: {}".format(self.clemgame))
        if self.level not in game_level_types[self.clemgame]:
            raise ValueError("The input attribute level should be one of {}, found: {}".format(game_level_types[self.clemgame], self.level))
    
        manual_analysis_data_formatters = {
            "taboo": taboogame_init_data_formatter,
            "wordle_withclue": wordle_withclue_data_formatter,
            "referencegame": referencegame_data_formatter,
            "imagegame": imagegame_data_formatter 
        }
        manual_analysis_data_containers = {
            "taboo": "all_init_interactions_player1_df",
            "wordle_withclue": "all_interactions_player1_df",
            "referencegame": "all_interactions_player1_df",
            "imagegame": "all_interactions_player2_df"
        }

        if self.clemgame == "wordle_withclue":
            merge_dfs_on_columns_ = functools.partial(merge_dfs_on_columns, columns=["prompt_instruction", "target_word"])
        else:
            merge_dfs_on_columns_ = functools.partial(merge_dfs_on_columns, columns=["prompt_instruction"])

        self.all_models_data_formatters = [manual_analysis_data_formatters[self.clemgame](cllm_name=model_name,
                                                                                          level=self.level,
                                                                                          with_cot=self.with_cot) 
                                           for model_name in self.models]

        self.all_models_data_containers = [getattr(self.all_models_data_formatters[index], manual_analysis_data_containers[self.clemgame])
                                           for index, model_name in enumerate(self.models)]
        self.extracted_annotations = merge_dfs_on_columns_(self.all_models_data_containers)

        self.filename = "n_manual_annotations_{}_{}_{}.xlsx".format(
            self.experiment_name,
            self.clemgame,
            self.level
        )
        save_to_excel(self.extracted_annotations, os.path.join(path_to_outputs_dir, self.filename))
        # upload to s3 
        upload_to_s3(self.filename, "im-bhavsar", delete_after_upload=True)
        
if __name__ == "__main__":
    
    ############### MAIN ###############
    config = load_from_yaml("./analysis/gated_llms_config.yaml")

    for group in config["manual_analysis"]["selected_model_groups"]:
        for competency in config["manual_analysis"]["selected_competencies"]:
            for game in competency["selected_games"]:
                for game_level in game["levels"]:
                    experiment_name = "{}_{}".format(group["group_name"], competency["competency_name"])
                    out = clembench_emergence_annotation_extractor(
                        clemgame=game["game_name"], 
                        level=game_level, 
                        models=group["group_model_names"],
                        experiment_name=experiment_name,
                        with_cot=False
                    )
    ####################################

    ############### TEST ###############
    # out = clembench_emergence_annotation_extractor(clemgame="referencegame", 
    #                                                level="2_diagonal_grids", 
    #                                                models=['claude-2.1', 'claude-3-haiku-20240307', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'], 
    #                                                experiment_name="test_experiment", 
    #                                                with_cot=False)
    # print(out.extracted_annotations)
    #################################### 