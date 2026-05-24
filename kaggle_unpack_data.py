from pathlib import Path
import zipfile

# Change this folder name after adding your Kaggle Dataset as an input.
# In Kaggle, check the right-side "Input" panel for the exact dataset slug.
ZIP_PATH = Path("/kaggle/input/ml-project-eeg-data/ml_project_data_kaggle.zip")
OUT_DIR = Path("/kaggle/working/data")

OUT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(OUT_DIR)

DATA_DIR = OUT_DIR / "ml_project_data"
print("Data extracted to:", DATA_DIR)
print("Top-level datasets:", [p.name for p in DATA_DIR.iterdir() if p.is_dir()])
