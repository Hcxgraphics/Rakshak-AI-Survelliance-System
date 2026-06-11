"""
Downloads the DangePolice dataset used by police_training_pipeline.py.
Adjust the dataset slug if you use a different source.
"""
import os, subprocess, sys

def download_roboflow():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY", ""))
    # Replace with your actual Roboflow project slug
    project = rf.workspace("your-workspace").project("dangepolice")
    dataset = project.version(1).download(
        "coco",
        location="datasets/police/DangePolice_coco"
    )
    print(f"Dataset at: {dataset.location}")

def download_kaggle_fallback():
    """Fallback to individual Kaggle datasets merged into COCO format."""
    import kaggle
    datasets = [
        ("issaisasank/guns-object-detection",  "datasets/police/raw/guns"),
        ("rkuorei/knife-detection",             "datasets/police/raw/knives"),
        ("mohamedmustafa/police-detection",     "datasets/police/raw/police"),
    ]
    for slug, dest in datasets:
        os.makedirs(dest, exist_ok=True)
        try:
            kaggle.api.dataset_download_files(slug, path=dest, unzip=True)
            print(f"  Downloaded {slug}")
        except Exception as e:
            print(f"  Failed {slug}: {e}")
    print("\n⚠️  After download, convert to COCO format with scripts/convert_to_coco.py")

if __name__ == "__main__":
    try:
        download_roboflow()
    except Exception as exc:
        print(f"Roboflow failed ({exc}), trying Kaggle …")
        download_kaggle_fallback()
