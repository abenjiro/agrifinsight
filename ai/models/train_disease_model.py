"""
Train Crop Disease Detection Model (Resumable + Optimized for Apple M1 8GB)
"""

import tensorflow as tf
import os
from datetime import datetime
from disease_detection import DiseaseDetectionModel

# ---------------------------
# Paths & Parameters
# ---------------------------
data_dir = "dataset"
checkpoint_dir = "checkpoints"
final_model_path = os.path.join("models", "crop_disease_model_final.keras")

img_size = (224, 224)
batch_size = 8              # smaller batch fits 8GB RAM
epochs = 20
seed = 123

# ---------------------------
# Load dataset efficiently
# ---------------------------
print("📦 Loading dataset...")

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

class_names = raw_train_ds.class_names
print(f"✅ Detected {len(class_names)} classes: {class_names[:5]} ...")

AUTOTUNE = tf.data.AUTOTUNE

# ⚠️ Do NOT use .cache() fully in memory — it kills RAM on large datasets
train_ds = raw_train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = raw_val_ds.prefetch(buffer_size=AUTOTUNE)

# ---------------------------
# Prepare checkpoints
# ---------------------------
os.makedirs(checkpoint_dir, exist_ok=True)
latest_weights = tf.train.latest_checkpoint(checkpoint_dir)

# ---------------------------
# Load existing model or start new
# ---------------------------
if latest_weights:
    print(f"🔄 Resuming from weights: {latest_weights}")
    model_wrapper = DiseaseDetectionModel(class_names=class_names)
    model = model_wrapper.model
    model.load_weights(latest_weights)
else:
    print("🚀 Starting new training session...")
    model_wrapper = DiseaseDetectionModel(class_names=class_names)
    model = model_wrapper.model

# ---------------------------
# Callbacks (safe + resumable)
# ---------------------------
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.weights.h5")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # lightweight checkpointing
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [checkpoint_cb, tensorboard_cb, earlystop_cb]

# ---------------------------
# Train / Resume Training
# ---------------------------
print(f"🧠 Training for {epochs} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

# ---------------------------
# Save final model
# ---------------------------
os.makedirs("models", exist_ok=True)
model.save(final_model_path)
print(f"✅ Final model saved to {final_model_path}")
print(f"✅ Checkpoints in {checkpoint_dir}")
print(f"✅ TensorBoard logs at {log_dir}")
