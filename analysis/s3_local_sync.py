import os
from constants import path_to_outputs_dir
from utils import create_s3_client, download_from_s3

def sync_local_with_s3(bucket_name: str="im-bhavsar"):
    try:
        if not os.path.exists(path_to_outputs_dir):
            os.makedirs(path_to_outputs_dir)
        s3 = create_s3_client()
        paginator = s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in result:
                for item in result['Contents']:
                    object_key = item['Key']
                    local_file_path = os.path.join(path_to_outputs_dir, object_key)
                    if not os.path.exists(local_file_path) or \
                            s3.head_object(Bucket=bucket_name, Key=object_key)['ContentLength'] != os.path.getsize(
                                local_file_path):
                        download_from_s3(object_name=object_key, bucket_name=bucket_name)
                    else:
                        print(f"File '{path_to_outputs_dir}' is up to date. Skipping download.")
        print("Sync completed successfully.")
    except Exception as e:
        print(f"Error syncing local directory with S3: {e}")

if __name__ == "__main__":
    sync_local_with_s3()    

