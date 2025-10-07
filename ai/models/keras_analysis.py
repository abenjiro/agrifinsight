import keras
model = keras.models.load_model('crop_disease_model_final.keras')
model.summary()
