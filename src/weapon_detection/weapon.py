from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained weights

# Open the webcam (0 for default webcam, change if needed)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Display results
    for r in results:
        annotated_frame = r.plot()  # Draw bounding boxes

    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
