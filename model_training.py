# model_training.py
# -----------------------------
# This script trains a simple CNN model on fake emotion data
# and saves it as face_emotionModel.h5
# -----------------------------

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1. Create fake training data
# -----------------------------
# We simulate grayscale 48x48 images and assign random emotion labels (0–6)
num_samples = 700  # small dataset for quick training
X = np.random.rand(num_samples, 48, 48, 1)
y = np.random.randint(0, 7, num_samples)
y = to_categorical(y, num_classes=7)

# -----------------------------
# 2. Build CNN model
# -----------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 3. Train model briefly
# -----------------------------
print("Training model on dummy data (for assignment demonstration)...")
model.fit(X, y, epochs=3, batch_size=32, verbose=1)
print("✅ Model training complete!")

# -----------------------------
# 4. Save model
# -----------------------------
model.save("face_emotionModel.h5")
print("✅ Model saved as face_emotionModel.h5")
