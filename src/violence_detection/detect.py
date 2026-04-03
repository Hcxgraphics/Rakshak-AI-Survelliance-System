import cv2
import numpy as np
import tensorflow as tf

# ✅ Load trained CNN model
model = tf.keras.models.load_model("final_model.h5")

# ✅ Define input size
IMG_SIZE = (224, 224)

# ✅ Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Preprocess the frame
    input_frame = cv2.resize(frame, IMG_SIZE)
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
    input_frame = input_frame / 255.0  # Normalize

    # ✅ Make prediction
    prediction = model.predict(input_frame)[0][0]

    # ✅ Display result
    label = "Violence Detected!" if prediction > 0.5 else "Safe"
    color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
