"""
Crop Disease Detection Model
Uses computer vision to identify plant diseases from images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DiseaseDetectionModel:
    """Crop disease detection using CNN"""
    
    def __init__(self, model_path: Optional[str] = None, class_names: Optional[List[str]] = None):
        self.model = None
        self.class_names = class_names if class_names else []
        self.confidence_threshold = 0.7
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif self.class_names:  
            self.build_model()
        else:
            raise ValueError("Either model_path or class_names must be provided.")

    def build_model(self):
        """Build the CNN model architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(224, 224, 3)),

            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),

            # Convolutional layers
            layers.Conv2D(32, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            # Global average pooling
            layers.GlobalAveragePooling2D(),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',  # matches dataset labels
            metrics=['accuracy']
        )

        self.model = model
        logger.info("Disease detection model built successfully")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input"""
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)

    def predict(self, image_path: str) -> Dict:
        """Predict disease from image"""
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image)
        prediction = predictions[0]

        top_prediction_idx = np.argmax(prediction)
        confidence = float(prediction[top_prediction_idx])
        disease = self.class_names[top_prediction_idx]

        return {
            'disease_detected': disease,
            'confidence_score': confidence,
            'top_predictions': [
                {'disease': self.class_names[idx], 'confidence': float(prediction[idx])}
                for idx in np.argsort(prediction)[-3:][::-1]
            ],
            'is_healthy': "healthy" in disease.lower(),
            'needs_attention': confidence > self.confidence_threshold and "healthy" not in disease.lower()
        }

    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model:
            return self.model.summary()
        return "Model not built yet"
