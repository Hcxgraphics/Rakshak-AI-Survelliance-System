import os, subprocess, sys

def download_via_roboflow():
    """Downloads weapon detection dataset using Roboflow API."""
    # Install roboflow if needed
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])
    from roboflow import Roboflow

    # Public dataset — no API key required for download
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY", ""))
    project = rf.workspace("roboflow-universe-projects").project("weapons-detection-8pegg")
    dataset = project.version(1).download("yolov8", location="datasets/weapon/roboflow")
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location

def download_via_kaggle_fallback():
    """Fallback: download from Kaggle."""
    import kaggle
    kaggle.api.dataset_download_files(
        "issaisasank/guns-object-detection",
        path="datasets/weapon/kaggle",
        unzip=True
    )
    print("Downloaded guns dataset from Kaggle")

if __name__ == "__main__":
    try:
        download_via_roboflow()
    except Exception as exc:
        print(f"Roboflow failed ({exc}), trying Kaggle …")
        download_via_kaggle_fallback()
