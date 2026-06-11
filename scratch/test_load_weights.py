import tensorflow as tf
from tensorflow import keras
from pathlib import Path

def build_cnn_lstm(seq_len=16, frame_size=64):
    base = keras.applications.MobileNetV2(
        input_shape=(frame_size, frame_size, 3),
        include_top=False,
        pooling="avg",
        weights=None,
    )
    inputs = keras.Input(shape=(seq_len, frame_size, frame_size, 3))
    x = keras.layers.TimeDistributed(base)(inputs)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False))(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="violence_cnn_lstm")
    return model

try:
    model = build_cnn_lstm()
    model.load_weights("models/violence/final_model.h5")
    print("Success: Loaded weights successfully!")
    model.summary()
except Exception as e:
    print("Failed to load weights:", e)
    import traceback
    traceback.print_exc()
