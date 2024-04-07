import os
import re
import json
import yaml
from typing import List, Union  
import pandas as pd
import boto3
from constants import path_to_outputs_dir

def load_from_json(path_to_file):
    with open(path_to_file) as f:
        data = json.load(f)
    return data

def load_from_yaml(path_to_file: str):
    with open(path_to_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

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

# AWS S3
access_tokens = load_from_yaml("./access_tokens.yaml")
aws_access_key_id = access_tokens["aws_access_key_id"]
aws_secret_access_key = access_tokens["aws_secret_access_key"]
region="eu-central-1"

def create_s3_client():
    try:
        s3 = boto3.client('s3',
                        region_name=region,
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)
        return s3
    except Exception as e:
        print("Error creating s3 client: {}".format(e))
        return None

def upload_to_s3(filename: str, bucket_name: str="im-bhavsar", delete_after_upload: bool=False):
    file_path = os.path.join(path_to_outputs_dir, filename) # usually meant for uploading data from cluster server
    try:
        s3 = create_s3_client() # create a new s3 client
        if s3:
            with open(file_path, 'rb') as data:
                s3.put_object(Bucket=bucket_name, Key=filename, Body=data)
            print(f"File uploaded successfully to https://{bucket_name}.s3.{region}.amazonaws.com/{filename}")
            if delete_after_upload:
                os.remove(file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False
    
def download_from_s3(object_name: str, bucket_name: str="im-bhavsar"):
    local_file_path = os.path.join(path_to_outputs_dir, object_name) # usually meant for downloading data locally
    try:
        s3 = create_s3_client() # create a new s3 client
        if s3:
            response = s3.get_object(Bucket=bucket_name, Key=object_name)
            with open(local_file_path, 'wb') as f:
                f.write(response['Body'].read())
            print(f"File downloaded successfully to {local_file_path}")
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False
