
base_path = '/mnt/c/Users/MuhammadSaqib/Documents/ILP-Agent-Framework/'
def reset_env():
    import shutil
    import os
    import json

    # Define the paths
    folder_path = os.path.join(base_path, "v3_core/build/deployments")
    json_file1_path = os.path.join(base_path, "model_storage/token_pool_addresses.json")
    json_file2_path = os.path.join(base_path, "model_storage/liq_positions.json")

    # 1. Delete the folder and its contents
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # 2. Delete contents of the first JSON file
    with open(json_file1_path, 'w') as file:
        file.write("{}")

    # 3. Delete contents of the second JSON file and add {}
    with open(json_file2_path, 'w') as file:
        file.write("{}")

reset_env()