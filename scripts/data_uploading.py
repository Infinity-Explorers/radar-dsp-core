import os
from pathlib import Path
from huggingface_hub import login, upload_folder

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "Automotive"

login()
upload_folder(
    folder_path=str(DATA_DIR),
    repo_id="hany34/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection",
    repo_type="dataset"    
)