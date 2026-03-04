import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from CNN_Classifier.utils.common import read_yaml
import json
from pathlib import Path




class PredictionPipeline:
    def __init__(self):
        self.config = read_yaml("config.yaml")["model_training"]
        self.model_path = self.config["trained_model_dir_kpca_svm"]
        label_path = Path(self.config["root_dir"]) / "class_names.json"
        with open(label_path, "r") as f:
            self.class_names = json.load(f)

    def predict(self, img_path):
        # Load the trained model
        model = load_model(self.model_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
        img_array = image.img_to_array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess input for EfficientNet

        # Make prediction
        prob = model.predict(img_array)[0]
        index = np.argmax(prob)
        confidence = float(prob[index])
        label = self.class_names[index]

        return {
            "predicted_class": label,
            "predicted_class_index": index,
            "confidence": confidence,
        }