import os

train_path = "C:/Users/Acer/Desktop/weaponnn/train/images"
num_images = len([f for f in os.listdir(train_path) if f.endswith('.jpg') or f.endswith('.png')])
print("Number of training images:", num_images)

epochs = 16000 / num_images
print(f"Approximate epochs: {epochs:.2f}")
