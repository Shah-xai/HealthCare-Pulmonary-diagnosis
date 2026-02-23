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
        self.base_model_preparation._prepare_transfer_learning_model(
            base_model_path=str(self.config.base_model_path),
            updated_base_model_path=str(self.config.updated_base_model_path),
            params_learning_rate=self.config.params_learning_rate,
            num_classes=self.config.params_classes,
            classifier_activation=self.config.params_classifier_activation
        )

        # Step 3: Prepare the feature extractor for hybrid learning
        self.base_model_preparation._prepare_feature_extractor(
             base_model_path=str(self.config.base_model_path),
             feature_extractor_path=str(self.config.feature_extract_dir),
             pooling=self.config.params_pooling,
             freez_all=True
         )