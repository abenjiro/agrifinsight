"""
Disease Detection Model Definition (Optimized for Apple M1)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class DiseaseDetectionModel:
    def __init__(self, class_names):
        self.class_names = class_names
        self.img_size = (224, 224)
        self.model = self._build_model()

    def _build_model(self):
        # Enable mixed precision for M1 (saves memory + speeds up)
        keras.mixed_precision.set_global_policy("mixed_float16")

        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False  # Freeze feature extractor for now

        model = models.Sequential([
            keras.Input(shape=(*self.img_size, 3)),
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation="softmax", dtype="float32")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def predict_image(self, image_path):
        """Predict disease class for a single image."""
        img = keras.utils.load_img(image_path, target_size=self.img_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        predictions = self.model.predict(img_array)
        top_idx = tf.argmax(predictions[0]).numpy()
        return {
            "disease": self.class_names[top_idx],
            "confidence": float(predictions[0][top_idx]),
            "all_confidences": {
                self.class_names[i]: float(pred)
                for i, pred in enumerate(predictions[0])
            }
        }
