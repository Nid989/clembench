import re
import json
from typing import List, Union  
import pandas as pd

def load_from_json(path_to_file):
    with open(path_to_file) as f:
        data = json.load(f)
    return data

def save_to_excel(data: pd.DataFrame, path_to_file: str):
    data.to_excel(path_to_file, index=False)

def extract_model_id(cllm_pair: str):
    pattern = re.compile(r'^(.*?)(?:-t0\.0)?(?:--|$)')
    match = pattern.match(cllm_pair)
    if match:
        return match.group(1)
    else:
        return "error"

# format cllm-pair (unique) identification name using model name (cllm name)
format_cllm_pair = lambda model_name: "{}-t0.0--{}-t0.0".format(model_name, model_name)

def merge_dfs_on_columns(dfs: List[pd.DataFrame], 
                         columns: Union[str, List[str]] = ["prompt_instruction"]) -> pd.DataFrame:
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=columns, how="outer")
    return merged_df