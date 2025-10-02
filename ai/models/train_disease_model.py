"""
Train Crop Disease Detection Model with Auto Resume + Flexible Epochs
"""

import tensorflow as tf
import os
import re
from datetime import datetime
from disease_detection import DiseaseDetectionModel

# ---------------------------
# Paths & Parameters
# ---------------------------
data_dir = "dataset"
checkpoint_dir = "checkpoints"
final_model_path = os.path.join("models", "crop_disease_model_final.keras")

img_size = (224, 224)
batch_size = 32
epochs = 15  # 🔄 you can increase this anytime (e.g., 15, 20, 30...)

# ---------------------------
# Load dataset
# ---------------------------
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = raw_train_ds.class_names
print(f"✅ Detected {len(class_names)} classes")
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------
# Load checkpoint if exists
# ---------------------------
os.makedirs(checkpoint_dir, exist_ok=True)

keras_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")]
if keras_checkpoints:
    latest_checkpoint = max(
        [os.path.join(checkpoint_dir, f) for f in keras_checkpoints],
        key=os.path.getctime
    )
    print(f"🔄 Resuming training from checkpoint: {latest_checkpoint}")
    model = tf.keras.models.load_model(latest_checkpoint)

    # Extract epoch number from filename
    match = re.search(r"epoch_(\d+)", latest_checkpoint)
    initial_epoch = int(match.group(1)) if match else 0
else:
    print("🚀 Starting training from scratch...")
    model_wrapper = DiseaseDetectionModel(class_names=class_names)
    model = model_wrapper.model
    initial_epoch = 0

# ---------------------------
# Callbacks
# ---------------------------
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras")

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=False,
    verbose=1
)

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

callbacks = [checkpoint_cb, tensorboard_cb, earlystop_cb]

# ---------------------------
# Train
# ---------------------------
print(f"👉 Resuming from epoch {initial_epoch}, training until {epochs} total epochs.")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)

# ---------------------------
# Save final model
# ---------------------------
if initial_epoch < epochs:
    os.makedirs("models", exist_ok=True)
    model.save(final_model_path)
    print(f"✅ Training complete. Final model saved at {final_model_path}")
else:
    print("⚠️ Training already reached this epoch count — final model not overwritten.")
