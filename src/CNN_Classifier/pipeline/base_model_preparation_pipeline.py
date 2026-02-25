from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.base_model_preparation import BaseModelPreparation

class BaseModelPreparationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_base_model_config()
        self.base_model_preparation = BaseModelPreparation(config=self.config)

    def initiate_model_preparation(self):
        # Step 1: Get the base model
        self.base_model_preparation.get_base_model()

        # Step 2: Prepare the updated base model for transfer learning
        self.base_model_preparation.update_base_model()

        # Step 3: Prepare the feature extractor for hybrid learning
        self.base_model_preparation.create_feature_extractor()