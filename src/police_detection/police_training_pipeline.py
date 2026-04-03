from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
from tensorflow import keras
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

LOGGER = logging.getLogger(__name__)

CLASS_NAMES = ["NonViolence", "Violence", "guns", "knife", "police"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

# Single-label training is kept intentionally so the produced `.h5` remains
# compatible with the existing deployment inference head.
CLASS_PRIORITY = ["guns", "knife", "Violence", "police", "NonViolence"]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    class_index: int


def load_annotations(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_category_mapping(payload: dict) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for category in payload.get("categories", []):
        name = category.get("name")
        if name in CLASS_TO_INDEX:
            mapping[int(category["id"])] = name
    return mapping


def collapse_labels(label_names: Iterable[str]) -> str | None:
    unique_labels = {label for label in label_names if label in CLASS_TO_INDEX}
    for label in CLASS_PRIORITY:
        if label in unique_labels:
            return label
    return None


def build_samples(annotation_path: Path, image_root: Path) -> list[Sample]:
    payload = load_annotations(annotation_path)
    images = {int(image["id"]): image for image in payload.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = {}
    category_map = canonical_category_mapping(payload)

    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(annotation)

    samples: list[Sample] = []
    for image_id, image_meta in images.items():
        label_names = [
            category_map[annotation["category_id"]]
            for annotation in annotations_by_image.get(image_id, [])
            if annotation.get("category_id") in category_map
        ]
        label = collapse_labels(label_names)
        if label is None:
            continue

        image_path = image_root / image_meta["file_name"]
        if not image_path.exists():
            LOGGER.warning("Skipping missing image: %s", image_path)
            continue
        samples.append(Sample(image_path=image_path, class_index=CLASS_TO_INDEX[label]))

    return samples


def extract_features(samples: list[Sample], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("No samples were found for feature extraction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features: list[np.ndarray] = []
    labels: list[int] = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors = []
        for sample in batch:
            image = Image.open(sample.image_path).convert("RGB")
            tensors.append(transform(image))
            labels.append(sample.class_index)

        tensor_batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feature_maps = backbone(tensor_batch)
            pooled = F.adaptive_avg_pool2d(feature_maps, (1, 1)).flatten(1).cpu().numpy()
        features.extend(pooled)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def build_classifier(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_pipeline(
    train_annotations: Path,
    val_annotations: Path,
    train_images: Path,
    val_images: Path,
    output_model: Path,
    output_metadata: Path,
    *,
    epochs: int,
    batch_size: int,
) -> None:
    train_samples = build_samples(train_annotations, train_images)
    val_samples = build_samples(val_annotations, val_images)

    LOGGER.info("Collected %s train samples and %s val samples", len(train_samples), len(val_samples))
    LOGGER.info("Train class distribution: %s", Counter(sample.class_index for sample in train_samples))
    LOGGER.info("Val class distribution: %s", Counter(sample.class_index for sample in val_samples))

    train_features, train_labels = extract_features(train_samples, batch_size=batch_size)
    val_features, val_labels = extract_features(val_samples, batch_size=batch_size)

    train_targets = keras.utils.to_categorical(train_labels, num_classes=len(CLASS_NAMES))
    val_targets = keras.utils.to_categorical(val_labels, num_classes=len(CLASS_NAMES))

    model = build_classifier(train_features.shape[1], len(CLASS_NAMES))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(output_model), monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_features,
        train_targets,
        validation_data=(val_features, val_targets),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)

    metadata = {
        "class_names": CLASS_NAMES,
        "class_priority": CLASS_PRIORITY,
        "feature_extractor": "torchvision.mobilenet_v2.features + adaptive_avg_pool2d",
        "feature_dim": int(train_features.shape[1]),
        "train_samples": int(len(train_samples)),
        "val_samples": int(len(val_samples)),
        "epochs_trained": int(len(history.history["loss"])),
    }
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Saved classifier to %s", output_model)
    LOGGER.info("Saved metadata to %s", output_metadata)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    repo_root = project_root.parent.parent
    deploy_root = repo_root / "models" / "police"
    parser = argparse.ArgumentParser(description="Deterministic police/danger classifier training pipeline.")
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=repo_root / "datasets" / "police" / "DangePolice_coco" / "annotations" / "instances_train2017_normalized.json",
    )
    parser.add_argument(
        "--val-annotations",
        type=Path,
        default=repo_root / "datasets" / "police" / "DangePolice_coco" / "annotations" / "instances_val2017_normalized.json",
    )
    parser.add_argument(
        "--train-images",
        type=Path,
        default=repo_root / "datasets" / "police" / "DangePolice_coco" / "train2017",
    )
    parser.add_argument(
        "--val-images",
        type=Path,
        default=repo_root / "datasets" / "police" / "DangePolice_coco" / "val2017",
    )
    parser.add_argument("--output-model", type=Path, default=deploy_root / "police_or_danger.h5")
    parser.add_argument("--output-metadata", type=Path, default=deploy_root / "police_or_danger.labels.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level="INFO")
    args = parse_args()
    train_pipeline(
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        train_images=args.train_images,
        val_images=args.val_images,
        output_model=args.output_model,
        output_metadata=args.output_metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
