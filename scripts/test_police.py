import torch, numpy as np
import tensorflow as tf
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import torch.nn.functional as F

POLICE_CLASS_NAMES = ["NonViolence", "Violence", "guns", "knife", "police"]

# Load backbone
backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features.eval()

# Load head
head = tf.keras.models.load_model("models/police/police_or_danger.h5", compile=False)
print("Head output shape:", head.output_shape)   # must be (None, 5)

# Dummy image test
dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), np.uint8))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tensor = transform(dummy_img).unsqueeze(0)
with torch.no_grad():
    features = backbone(tensor)
    pooled = F.adaptive_avg_pool2d(features, (1,1)).view(1, -1).cpu().numpy()
preds = np.asarray(head.predict(pooled, verbose=0)[0])
scores = dict(zip(POLICE_CLASS_NAMES, preds.tolist()))
print("Dummy scores:", scores)
print("✅ Police pipeline OK")
