from CNN_Classifier import logger
from CNN_Classifier.entity.config_entity import BaseModelConfig
import tensorflow as tf
from pathlib import Path

class BaseModelPreparation:
    def __init__(self, config: BaseModelConfig):
        self.config = config
# A method to download the base model if it doesn't exist and save it to the specified path
    def get_base_model(self):
        logger.info("Preparing the base model ....")
        if not self.config.base_model_path.exists():
            logger.info(f"Base model not found at {self.config.base_model_path}. Downloading...")
            base_model = tf.keras.applications.ResNet50V2(
                include_top=self.config.params_include_top,
                weights=self.config.params_weights,
                input_shape=self.config.params_input_shape,
                pooling=self.config.params_pooling,
                classes=self.config.params_classes,
                classifier_activation=self.config.params_classifier_activation
            )
            base_model.save(self.config.base_model_path)
            logger.info(f"Base model downloaded and saved at {self.config.base_model_path}")
        else:
            logger.info(f"Base model already exists at {self.config.base_model_path}. Loading the model...")

    # Prepare the updated base model for transer learning by adding a new classification head and freezing the layers of the base model
    
    @staticmethod
    def _prepare_transfer_learning_model(base_model_path: str,
                                     updated_base_model_path: str,
                                     params_learning_rate: float, 
                                     num_classes: int, classifier_activation: str,
                                     freez_all=True,freez_to=None):
        
        if  Path(updated_base_model_path).exists():
            logger.info(f"Updated base model already exists at {updated_base_model_path}. Loading the model...")
            return tf.keras.models.load_model(updated_base_model_path)
       
        logger.info("Preparing the updated base model ....")
        base_model = tf.keras.models.load_model(base_model_path)
        if freez_all:
            for layer in base_model.layers:
                layer.trainable = False
        elif freez_to is not None and isinstance(freez_to, int):
            for layer in base_model.layers[:-freez_to]:
                layer.trainable = False
        else:
            logger.warning("Invalid freez_to value. No layers will be frozen.")
    # Example of adding a new classification head
        x = base_model.output
        x=tf.keras.layers.Dense(1024, activation='relu')(x)
        predictions = tf.keras.layers.Dense(num_classes, activation=classifier_activation)(x)
        updated_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    # Compile the updated model with the specified learning rate
        updated_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params_learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        print("Transfer learning model prepared with the following architecture:")
        updated_model.summary()
        return updated_model
    
    # Prepare the feature extractor for hybrid learning
    @staticmethod
    def _prepare_feature_extractor(base_model_path: str, feature_extractor_path: str,pooling: str="avg", freez_all=True):
        if Path(feature_extractor_path).exists():
            logger.info(f"Feature extractor already exists at {feature_extractor_path}. Loading the model...")
            return tf.keras.models.load_model(feature_extractor_path)
        logger.info("Preparing the feature extractor for hybrid learning ....")
        base_model = tf.keras.models.load_model(base_model_path)
        if freez_all:
            for layer in base_model.layers:
                layer.trainable = False
        x = base_model.output

    # Force a consistent 2D embedding output for any saved base model configuration
        if len(x.shape) == 4:  # (batch, H, W, C)
            if pooling == "avg":
                x = tf.keras.layers.GlobalAveragePooling2D(name="feat_gap")(x)
            elif pooling == "max":
                x = tf.keras.layers.GlobalMaxPooling2D(name="feat_gmp")(x)
            else:
                raise ValueError("pooling must be 'avg' or 'max'")

        elif len(x.shape) != 2:
            raise ValueError(f"Unexpected base model output rank {len(x.shape)} with shape {x.shape}")

        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=x, name="feature_extractor")
   
        
        return feature_extractor
    
    # A method to prepare the updated base model and save it to the specified path
    def update_base_model(self):
        updated_base_model = self._prepare_transfer_learning_model(
            base_model_path=str(self.config.base_model_path),
            updated_base_model_path=str(self.config.updated_base_model_path),
            params_learning_rate=self.config.params_learning_rate,
            num_classes=self.config.params_classes,
            classifier_activation=self.config.params_classifier_activation,
            freez_all=True
        )
           
        updated_base_model.save(self.config.updated_base_model_path)
        logger.info(f"Updated base model prepared and saved at {self.config.updated_base_model_path}")

    def create_feature_extractor(self):
        feature_extractor = self._prepare_feature_extractor(
            base_model_path=str(self.config.base_model_path),
            feature_extractor_path=str(self.config.feature_extract_dir),
            pooling=self.config.params_pooling,
            freez_all=True
        )
        
        feature_extractor.save(self.config.feature_extract_dir)
        logger.info(f"Feature extractor saved at {self.config.feature_extract_dir}")
        