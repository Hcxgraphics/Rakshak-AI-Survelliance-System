
import cv2
import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from accidentModel import accident_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Feature extractor backbone (just like training)
mobilenet_feature_extractor = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))
mobilenet_feature_extractor.trainable = False


# Use MobileNetV2 as feature extractor
mobilenet_feature_extractor = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))
mobilenet_feature_extractor.trainable = False

# Police classifier model
police_model = tf.keras.models.load_model("models/policeOrDanger.h5")



# Load Models
weapon_model = YOLO("models/weapon.pt")
# police_model = tf.keras.models.load_model("models/policeOrDanger.h5")
violence_model = tf.keras.models.load_model("models/VoilenceNonVoilence_mobiLSTM.h5")
import torchvision.models as models
# accident_model_loader.py

transform_accident = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



# # models.py
# from keras.models import load_model
# from backbone import mobilenet
# import torch
# from torchvision import transforms
# import ultralytics

# # --- Load all models ---
# print("✅ Loading models...")

# # YOLOv8 for weapon detection
# weapon_model = ultralytics.YOLO("best.pt")  # Replace with actual path

# # Police classifier
# mobilenet_feature_extractor = mobilenet
# police_model = load_model("policeOrDanger.h5")  # Replace with actual path

# # Violence classifier (CNN-LSTM)
# violence_model = load_model("violence_cnn_lstm.h5")  # Replace with actual path

# # Accident detector
# from accidentModel import accidentModel  # Your custom torch model
# accident_model = accidentModel()
# accident_model.load_state_dict(torch.load("accidentModel.pth", map_location="cpu"))
# accident_model.eval()

# # Transform for accident model
# transform_accident = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
