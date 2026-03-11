import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from CNN_Classifier.utils.common import read_yaml, load_object
import json
from scipy.special import expit as sigmoid
from pathlib import Path




class PredictionPipeline:
    def __init__(self):
        self.config = read_yaml("config.yaml")["model_deployment"]
        self.model_path = Path(self.config.model_dir)
        self.feature_extractor_path = Path(self.config.feature_extractor_dir)
        label_path = Path(self.config["root_dir"]) / "class_names.json"
        with open(label_path, "r") as f:
            self.class_names = json.load(f)
        self.feature_extractor = tf.keras.models.load_model(self.feature_extractor_path)
        self.kpca_svm_model = load_object(self.model_path)
    @staticmethod
    def _image_to_feature_vector(feature_extractor, img_path: str) -> np.ndarray:
        """
        Returns shape (1, D) feature vector.
        """
        img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
        x = image.img_to_array(img).astype(np.float32)
        x = np.expand_dims(x, axis=0)

        # must match training preprocessing
        x = preprocess_input(x)

        # feature extractor outputs (1, D) or (1, H, W, C) depending on model
        feats = feature_extractor(x, training=False).numpy()

        # ensure 2D (1, D)
        feats = feats.reshape(feats.shape[0], -1)
        return feats

    def predict(self, img_path):
        # Load the trained model
        x_feat = self._image_to_feature_vector(self.feature_extractor, img_path)
 
        # Make prediction
        index =int(self.kpca_svm_model.predict(x_feat)[0]) 
        scores = self.kpca_svm_model.decision_function(x_feat)
        confidence = float(sigmoid(np.max(scores)))  # Get the confidence score for the predicted class
        label = self.class_names[index]

        return {
            "predicted_class": label,
            "confidence": confidence,
        }
    
if __name__ == "__main__":
        pipeline = PredictionPipeline()
        result = pipeline.predict("Image_8154ab106a464e0eb4aace70e92c1392.jpg")
        print(result)