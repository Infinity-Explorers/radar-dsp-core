import os
from pathlib import Path
from huggingface_hub import login, upload_folder, upload_file

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "Automotive"
FILE_PATH = SCRIPT_DIR.parent / "data" / "metadata.parquet"

login()
# upload_folder(
#     folder_path=str(DATA_DIR),
#     repo_id="hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection",
#     repo_type="dataset"    
# )

upload_file(
    path_or_fileobj=str(FILE_PATH),
    path_in_repo="metadata.parquet",
    repo_id="hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection",
    repo_type="dataset"    
)