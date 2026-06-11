"""
Downloads the 'Real Life Violence Situations' dataset from Kaggle.
Kaggle credentials required: set KAGGLE_USERNAME and KAGGLE_KEY env vars,
or place ~/.kaggle/kaggle.json.
"""
import subprocess, sys, os

def download():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
    import kaggle
    os.makedirs("datasets/violence", exist_ok=True)
    kaggle.api.dataset_download_files(
        "mohamedmustafa/real-life-violence-situations-dataset",
        path="datasets/violence",
        unzip=True,
    )
    print("✅ Violence dataset ready at datasets/violence/")

if __name__ == "__main__":
    download()
