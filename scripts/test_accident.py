import torch, numpy as np, sys
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path("src/deployment").resolve()))
from accidentModel import load_accident_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_accident_model(Path("models/accident/accident_model.pth"), device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), np.uint8))
tensor = transform(dummy).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(tensor)
    probs  = torch.softmax(logits[0], dim=0).cpu().numpy()

print("Class probabilities:", probs)
print("Predicted class:", int(np.argmax(probs)))
print("✅ Accident model OK")
