import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

# ✅ Enable mixed precision for faster training (if GPU is available)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ✅ Path to processed images
IMAGE_PATH = r"C:\Users\Acer\Desktop\violence\ProcessedFrames"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# ✅ Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,  
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    validation_split=0.2  
)

train_data = datagen.flow_from_directory(
    IMAGE_PATH, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary', 
    subset='training'
)

val_data = datagen.flow_from_directory(
    IMAGE_PATH, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary', 
    subset='validation'
)

# ✅ CNN Model (without LSTM)
model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# ✅ Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks for better training
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
]

# ✅ Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ✅ Save the Model for Real-Time Detection
model.save("final_model.h5")
print("✅ Model trained and saved successfully with GPU optimization! 🚀")
