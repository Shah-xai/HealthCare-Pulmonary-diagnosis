from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_evaluation import ModelEvaluation

class ModelEvaluationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.model_evaluation = ModelEvaluation(config=self.config)

    def initiate_model_evaluation(self):
        logger.info("Initiating model evaluation...")
        self.model_evaluation.evaluate_model()