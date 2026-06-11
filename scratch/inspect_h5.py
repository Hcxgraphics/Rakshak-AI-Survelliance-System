import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

with h5py.File("models/violence/final_model.h5", "r") as f:
    print("Keys:", list(f.keys()))
    if "model_weights" in f:
        print("Model weights keys:", list(f["model_weights"].keys()))
        f["model_weights"].visititems(print_structure)
    else:
        f.visititems(print_structure)
