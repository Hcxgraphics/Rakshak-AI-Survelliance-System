import cv2
import numpy as np
import tensorflow as tf

# ✅ Load trained CNN model
model = tf.keras.models.load_model("final_model.h5")

def predict_frame(model, frame, seq_len=16):
    """Unified predictor that works for both CNN and CNN-LSTM models."""
    # Infer image size from model input shape
    if len(model.input_shape) == 5:
        # (batch, seq_len, H, W, C)
        h, w = model.input_shape[2], model.input_shape[3]
    else:
        # (batch, H, W, C)
        h, w = model.input_shape[1], model.input_shape[2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(frame_rgb, (w, h)).astype(np.float32) / 255.0

    if len(model.input_shape) == 5:
        # CNN-LSTM: (batch, seq_len, H, W, C)
        clip = np.repeat(resized[np.newaxis, np.newaxis, ...], seq_len, axis=1)
        inp  = clip  # shape (1, seq_len, H, W, C)
    else:
        # Pure CNN: (batch, H, W, C)
        inp = np.expand_dims(resized, axis=0)

    pred = model.predict(inp, verbose=0)
    # Flatten to scalar
    if pred.ndim > 1:
        pred = pred.flatten()
    return float(pred[0])

# ✅ Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Make prediction using unified helper
    prediction = predict_frame(model, frame)

    # ✅ Display result
    label = "Violence Detected!" if prediction > 0.5 else "Safe"
    color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
