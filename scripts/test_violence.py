import numpy as np, tensorflow as tf, sys

model = tf.keras.models.load_model("models/violence/final_model.h5", compile=False)
print("Input shape:", model.input_shape)   # must be (None, 16, 64, 64, 3)
print("Output shape:", model.output_shape) # must be (None, 1)

# Dummy clip test
clip = np.random.rand(1, 16, 64, 64, 3).astype(np.float32)
pred = model.predict(clip, verbose=0)
print("Prediction on random clip:", pred)  # should be scalar in [0,1]
assert pred.shape in [(1,), (1, 1)], f"Unexpected output shape: {pred.shape}"
print("✅ Violence model OK")
