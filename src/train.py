import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_dataset(data_dir):
    images = []
    labels = []
    class_dirs = ['healthy', 'unhealthy']
    for label, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize image to 224x224
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Assuming two classes: healthy and unhealthy
    ])
    model.compile(optimizer=Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    data_dir = 'data'  # Ensure you have a data folder with subfolders 'healthy' and 'unhealthy'
    X, y = load_dataset(data_dir)
    # Normalize pixel values to [0,1]
    X = X.astype('float32') / 255.0
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model()
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    
    # Save the trained model
    model.save('plant_health_model.h5')
    print("Model saved as plant_health_model.h5")
