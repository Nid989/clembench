import os
import math
import warnings
import collections
import itertools
import datetime
import numpy as np
import pandas as pd
from typing import List

from constants import (
    game_level_types, path_to_results_dir, path_to_outputs_dir
)
from analysis.utils import (
    load_from_json, save_to_excel, format_cllm_pair, upload_to_s3, load_from_yaml
)

warnings.filterwarnings("ignore")

class clembenc_emergence_automatic_analysis:
    def __init__(self, clemgame):
        self.clemgame = clemgame
        self.all_exp_cllms_pairs = [item for item in os.listdir(path_to_results_dir) if os.path.isdir(os.path.join(path_to_results_dir, item))] 
        self.game_level_types = game_level_types # Dict[str, List[str]] w/ game level types (game_name: [level_types])
        self.run_for_all() # run the analysis for all cllm pairs
        
    def _get_quality_score(self,
                          episode_scores: dict):
        quality_score = None
        quality_score = episode_scores["episode scores"]["Main Score"]
        return quality_score

    def _derive_level_scores(self,
                             cllm_pair: str):
        file_w_errors = []
        level_scores = collections.defaultdict(list)
        path_to_cllm_dir = os.path.join(path_to_results_dir, cllm_pair)
        path_to_cllm_clemgame_dir = os.path.join(path_to_cllm_dir, self.clemgame)

        if not os.path.exists(path_to_cllm_clemgame_dir):
            return {"aggregate_level_scores": {}, "level_scores": {}, "file_w_errors": []}

        for level in self.game_level_types[self.clemgame]:
            path_to_cllm_clemgame_level_dir = os.path.join(path_to_cllm_clemgame_dir, level)
            for episode in os.listdir(path_to_cllm_clemgame_level_dir):
                try:
                    if not episode.startswith("episode_"):
                        continue
                    score = self._get_quality_score(
                        load_from_json(
                            f"{path_to_cllm_clemgame_level_dir}/{episode}/scores.json"
                        )
                    )
                    # TODO: add code for other scores specific to the game
                    level_scores[level].append(score)
                except Exception as e:
                    file_w_errors.append(f"{path_to_cllm_clemgame_level_dir}/{episode} {e}")
        
        aggregate_level_scores = {level: np.mean(np.nan_to_num(np.asarray(scores), nan=0)) for level, scores in level_scores.items()}
        return {
            "aggregate_level_scores": aggregate_level_scores,
            "level_scores": level_scores,
            "file_w_errors": file_w_errors
        }

    def run_for_all(self):
        self.all_exp_cllms_pairs_scores = dict()
        for cllm_pair in self.all_exp_cllms_pairs:
            self.all_exp_cllms_pairs_scores[cllm_pair] = self._derive_level_scores(cllm_pair)

class clembench_emergence_scores_extractor:
    def __init__(self, clemgame: str="taboo", models: List[str]=None, experiment_name: str=None):
        self.clemgame = clemgame 
        self.clemgame_analysis = clembenc_emergence_automatic_analysis(self.clemgame)
        self.models = models
        self.experiment_name = experiment_name
        self._setup()

    def _setup(self):
        if self.models is None:
            raise ValueError("The expected type for the input attribute `models` should be List[str], but it was found to be None.")
        if self.experiment_name is None:
            raise ValueError("The expected type for the input attribute `experiment_name` should be str, but it was found to be None.")
        self.cllm_pairs = list(map(format_cllm_pair, self.models))
        self.relevant_scores = self._extract_relevant_scores()
        self.extracted_scores = self._process_scores()
        self.filename = f"{self.clemgame}_{self.experiment_name}_scores_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
        save_to_excel(self.extracted_scores, os.path.join(path_to_outputs_dir, self.filename)) # save file locally usually on server
        # upload to s3 
        upload_to_s3(self.filename, "im-bhavsar", delete_after_upload=True)

    def _extract_relevant_scores(self):
        relevant_scores = [self.clemgame_analysis.all_exp_cllms_pairs_scores[cllm_pair] 
                           for cllm_pair in self.cllm_pairs
                           if cllm_pair in self.clemgame_analysis.all_exp_cllms_pairs_scores]
        return relevant_scores

    def _process_scores(self):
        average_level_score = lambda aggregate_level_scores: np.mean(list(aggregate_level_scores.values()))
        average_group_episode_score = lambda level_scores: np.nanmean(np.nan_to_num(np.asarray(
            list(itertools.chain.from_iterable([episode_scores for _, episode_scores in level_scores.items()]))), nan=0))

        output_scores = collections.defaultdict(list)
        for model_name, cllm_pair_scores in zip(self.models, self.relevant_scores):
            output_scores["model"].append(model_name) 
            # level-wise average scores
            for level, level_score in cllm_pair_scores["aggregate_level_scores"].items():
                output_scores[level].append(level_score)
            # average level score
            output_scores["average_level_score"].append(average_level_score(cllm_pair_scores["aggregate_level_scores"]))
            # average group episode score
            output_scores["average_group_episode_score"].append(average_group_episode_score(cllm_pair_scores["level_scores"]))

        return pd.DataFrame(output_scores)

# combine models assigned to specific model family w/ clemscores based on configuration file
def aggregate_models_w_clemscore(results_filename: str="results.csv", 
                                 config: dict=None):
    path_to_results_filename = os.path.join(path_to_results_dir, results_filename)
    results_data = pd.read_csv(path_to_results_filename)

    def groupby_w_repeat_column_headers(data: pd.DataFrame, groupby_column_name: str="model_family"):
        grouped_data = data.groupby(groupby_column_name)    
        output_rows = []
        for name, group in grouped_data:
            output_rows.append(group.columns)
            for _, row in group.iterrows():
                output_rows.append(list(row))
        output_df = pd.DataFrame(output_rows[1:], columns=output_rows[0])
        return output_df

    skipped_models = collections.defaultdict(list)
    output = collections.defaultdict(list)
    for model_family in config["selected_model_groups"]:
        if model_family["group_by_parameters"]: # if model family is grouped by `parameters`
            group_name = model_family["group_name"]
            group_model_names = model_family["group_model_names"]
            for model_name in group_model_names:
                cllm_pair_name = "{}-t0.0--{}-t0.0".format(model_name, model_name)
                filtered_data = results_data[results_data['Unnamed: 0'] == cllm_pair_name]
                if not filtered_data.empty:
                    clemscore = filtered_data['-, clemscore'].iloc[0]
                    output["model_family"].append(group_name)
                    output["model_name"].append(model_name)
                    output["cllm_pair_name"].append(cllm_pair_name)
                    output["clemscore"].append(clemscore)
                else:
                    skipped_models[group_name].append(model_name)

    output_df = pd.DataFrame(output)
    formatted_output_df = groupby_w_repeat_column_headers(output_df)
    formatted_output_df.to_excel("./aggregated_models_w_clemscore.xlsx", index=False)
    output_filename = "aggregated_models_w_clemscore.xlsx"
    save_to_excel(formatted_output_df, os.path.join(path_to_outputs_dir, 
                                                    output_filename)) # save file locally usually on server
    # upload to s3 
    upload_to_s3(output_filename, "im-bhavsar", delete_after_upload=True)

if __name__ == "__main__":
    # # Example usage
    # models = ["openchat-3.5-0106", "openchat-3.5-1210"]
    # experiment_name = "test_experiment"
    # clemgame = "privateshared_tom"
    # out = clembench_emergence_scores_extractor(clemgame, models, experiment_name)
    # print(out.extracted_scores)
    
    # combine models assigned to specific model family w/ clemscores 
    config = load_from_yaml("./analysis/clembench_runs_config.yaml")

    aggregate_models_w_clemscore(results_filename="results.csv", 
                                 config=config)

# things I need to do
# given a model family it will extract all these models and put them in a list