import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy",
    "Corn Leaf Spot",
    "Corn Common Rust",
    "Corn Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Healthy"
]

class DiseasePredictor:
    def __init__(self, model_path='model/plant_disease_model.h5'):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print("Model loaded successfully")
            else:
                print("Model not found, using mock")
        except Exception as e:
            print(f"Error loading model: {e}")

    def preprocess_image(self, image_input):
        image = Image.open(image_input).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_input):
        try:
            img = self.preprocess_image(image_input)

            if self.model is None:
                return {
                    'class_idx': 0,
                    'disease': CLASS_NAMES[0],
                    'confidence': 95.0
                }

            preds = self.model.predict(img)

            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

            # 🔥 SAFE MAPPING
            if class_idx < len(CLASS_NAMES):
                disease_name = CLASS_NAMES[class_idx]
            else:
                disease_name = "Unknown"

            return {
                'class_idx': class_idx,
                'disease': disease_name,
                'confidence': round(confidence, 2)
            }

        except Exception as e:
            return {"error": str(e)}