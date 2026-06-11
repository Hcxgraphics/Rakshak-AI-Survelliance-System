"""
Converts raw COCO JSON so every category maps to exactly one of:
  ["NonViolence", "Violence", "guns", "knife", "police"]
All other categories are remapped or dropped.
"""
import json, pathlib, sys

CANONICAL = {
    # guns / firearms
    "gun": "guns", "pistol": "guns", "rifle": "guns", "firearm": "guns",
    "weapon": "guns",
    # knives / bladed
    "knife": "knife", "blade": "knife", "machete": "knife",
    # police
    "police": "police", "officer": "police", "cop": "police",
    # violence
    "fight": "Violence", "assault": "Violence", "attack": "Violence",
    # non-violence (fallback)
    "person": "NonViolence", "crowd": "NonViolence",
}
VALID = {"NonViolence", "Violence", "guns", "knife", "police"}

def normalize(src: pathlib.Path, dst: pathlib.Path):
    with src.open() as f:
        coco = json.load(f)

    # Remap categories
    id_map = {}
    new_cats = []
    new_id   = 1
    for cat in coco.get("categories", []):
        canonical = CANONICAL.get(cat["name"].lower())
        if canonical:
            id_map[cat["id"]] = new_id
            new_cats.append({"id": new_id, "name": canonical, "supercategory": "event"})
            new_id += 1

    coco["categories"] = new_cats

    # Remap annotations
    kept_anns = []
    for ann in coco.get("annotations", []):
        mapped = id_map.get(ann["category_id"])
        if mapped:
            ann["category_id"] = mapped
            kept_anns.append(ann)
    coco["annotations"] = kept_anns

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved normalized COCO to {dst}  ({len(kept_anns)} annotations)")

if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    normalize(src, dst)
