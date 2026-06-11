"""
Downloads the DADS (Dashcam Accident Detection) / CADP dataset.
Falls back to a smaller open dataset if the primary is unavailable.
"""
import os, subprocess, sys

def download_kaggle():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
    import kaggle
    # Primary: Road Accident Detection dataset (6 classes matching the model)
    try:
        kaggle.api.dataset_download_files(
            "ckay16/accident-detection-from-cctv-footage",
            path="datasets/accident",
            unzip=True
        )
        print("✅ Downloaded CCTV accident dataset")
        return
    except Exception as e:
        print(f"Primary dataset failed: {e}")

    # Fallback: Car accident image classification
    kaggle.api.dataset_download_files(
        "timetraveller98/road-accident-dataset-6-classes",
        path="datasets/accident",
        unzip=True
    )
    print("✅ Downloaded road accident dataset (6-class)")

if __name__ == "__main__":
    download_kaggle()
