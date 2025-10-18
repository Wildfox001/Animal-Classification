import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import json
import os

# -------------------------
# Dataset paths
# -------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"  # Optional if you have a test folder

# -------------------------
# Image parameters
# -------------------------
img_height, img_width = 128, 128
batch_size = 32

# -------------------------
# Data generators
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary'
)

print("Class indices:", train_gen.class_indices)

# Save class indices to JSON so Flask app can load it
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("Class indices saved to class_indices.json")

test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary', shuffle=False
)

# -------------------------
# Build CNN model
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------
# Train the model
# -------------------------
epochs = 15
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[early_stop]
)

# -------------------------
# Evaluate on test set
# -------------------------
if os.path.exists(test_dir):
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

# -------------------------
# Plot training history
# -------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -------------------------
# Save the trained model
# -------------------------
model.save("cattle_buffalo_model.h5")
print("Model saved as cattle_buffalo_model.h5")
