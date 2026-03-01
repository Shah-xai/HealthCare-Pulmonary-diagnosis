from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_training import ModelTraining

class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_training_config()
    def initiate_model_training(self):
        logger.info("Starting model training pipeline...")
        try:
            model_trainer = ModelTraining(config=self.config)
            model_trainer.train_model()
            logger.info("Model training pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
