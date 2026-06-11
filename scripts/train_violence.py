"""
Trains a MobileNetV2-LSTM violence classifier.
Input:  (batch, 16, 64, 64, 3)   ← matches build_violence_clip() in inference.py
Output: scalar sigmoid ∈ [0,1]   ← matches _predict_violence_score() in inference.py

Run AFTER extract_violence_frames.py.
Usage:
    python scripts/train_violence.py
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

SEQ_LEN    = 16
FRAME_SIZE = 64
BATCH_SIZE = 8
EPOCHS     = 30
LR         = 1e-4

SEQUENCES_ROOT = Path("datasets/violence/sequences")
OUTPUT_PATH    = Path("models/violence/final_model.h5")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(root: Path):
    X, y = [], []
    for label_name, label_idx in [("Violence", 1), ("NonViolence", 0)]:
        folder = root / label_name
        paths  = sorted(folder.glob("*.npy"))
        print(f"  {label_name}: {len(paths)} clips")
        for p in paths:
            clip = np.load(str(p))  # (16, 64, 64, 3)
            X.append(clip)
            y.append(label_idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def make_tf_dataset(X, y, *, shuffle: bool, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model definition ─────────────────────────────────────────────────────────

def build_cnn_lstm(seq_len=SEQ_LEN, frame_size=FRAME_SIZE):
    """
    TimeDistributed MobileNetV2 feature extractor + Bidirectional LSTM.
    Accepts (batch, seq_len, frame_size, frame_size, 3).
    Returns scalar sigmoid.
    """
    # Frame-level feature extractor
    base = keras.applications.MobileNetV2(
        input_shape=(frame_size, frame_size, 3),
        include_top=False,
        pooling="avg",
        weights="imagenet",
    )
    base.trainable = False  # freeze during first phase

    inputs = keras.Input(shape=(seq_len, frame_size, frame_size, 3))
    # Apply MobileNetV2 to each frame independently
    x = keras.layers.TimeDistributed(base)(inputs)                    # (B, 16, 1280)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False))(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="violence_cnn_lstm")
    return model, base


def train(args):
    print("Loading dataset …")
    X, y = load_dataset(SEQUENCES_ROOT)
    print(f"Total clips: {len(X)}  |  Violence: {int(y.sum())}  |  Non-violence: {int((1-y).sum())}")

    # Train / val split (80/20 stratified)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split  = int(0.8 * len(indices))
    tr_idx = indices[:split]
    vl_idx = indices[split:]

    ds_train = make_tf_dataset(X[tr_idx], y[tr_idx], shuffle=True,  batch_size=args.batch_size)
    ds_val   = make_tf_dataset(X[vl_idx], y[vl_idx], shuffle=False, batch_size=args.batch_size)

    model, base = build_cnn_lstm()
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(OUTPUT_PATH), save_best_only=True,
                                        monitor="val_auc", mode="max", verbose=1),
        keras.callbacks.EarlyStopping(patience=8, monitor="val_auc", mode="max",
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=4, min_lr=1e-7),
    ]

    print("\n── Phase 1: Train with frozen backbone ──")
    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs // 2, callbacks=callbacks)

    print("\n── Phase 2: Fine-tune last 20 layers of backbone ──")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr * 0.1),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs // 2, callbacks=callbacks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_PATH))
    print(f"\n✅ Violence model saved → {OUTPUT_PATH}")
    print(f"   Input shape  : {model.input_shape}")
    print(f"   Output shape : {model.output_shape}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
