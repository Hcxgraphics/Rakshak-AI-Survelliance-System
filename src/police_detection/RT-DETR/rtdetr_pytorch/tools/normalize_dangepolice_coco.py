from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_NAMES = ["NonViolence", "Violence", "guns", "knife", "police"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def normalize_annotations(source_path: Path, output_path: Path) -> None:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    categories = {int(category["id"]): category["name"] for category in payload.get("categories", [])}

    remapped_annotations = []
    referenced_image_ids: set[int] = set()
    next_annotation_id = 0

    for annotation in payload.get("annotations", []):
        category_name = categories.get(int(annotation["category_id"]))
        if category_name not in CLASS_TO_INDEX:
            continue

        bbox = annotation.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        normalized = dict(annotation)
        normalized["id"] = next_annotation_id
        normalized["category_id"] = CLASS_TO_INDEX[category_name]
        normalized["bbox"] = [float(value) for value in bbox]
        normalized["area"] = float(bbox[2] * bbox[3])
        normalized["iscrowd"] = int(annotation.get("iscrowd", 0))
        remapped_annotations.append(normalized)
        referenced_image_ids.add(int(annotation["image_id"]))
        next_annotation_id += 1

    payload["annotations"] = remapped_annotations
    payload["images"] = [image for image in payload.get("images", []) if int(image["id"]) in referenced_image_ids]
    payload["categories"] = [
        {"id": index, "name": name, "supercategory": "Danger-or-Police"}
        for index, name in enumerate(CLASS_NAMES)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    annotations_root = project_root / "DangePolice_coco" / "annotations"
    parser = argparse.ArgumentParser(description="Normalize DangerPolice COCO category ids to 0..4 consistently.")
    parser.add_argument("--train-source", type=Path, default=annotations_root / "instances_train2017.json")
    parser.add_argument("--val-source", type=Path, default=annotations_root / "instances_val2017.json")
    parser.add_argument("--train-output", type=Path, default=annotations_root / "instances_train2017_normalized.json")
    parser.add_argument("--val-output", type=Path, default=annotations_root / "instances_val2017_normalized.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_annotations(args.train_source, args.train_output)
    normalize_annotations(args.val_source, args.val_output)


if __name__ == "__main__":
    main()
