from ultralytics import YOLO
import cv2, sys, pathlib

model = YOLO("models/weapon/best.pt")
source = sys.argv[1] if len(sys.argv) > 1 else "assets/sample_images"

results = model.predict(source=source, conf=0.25, save=True, project="test_results", name="weapon")
for r in results:
    print(r.boxes.data)
print("Results saved to test_results/weapon/")
