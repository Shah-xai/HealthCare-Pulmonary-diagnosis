from CNN_Classifier import logger
from CNN_Classifier.components.data_transformation import DataTransformation
from CNN_Classifier.config.configuration import ConfigurationManager

class DataTransformationPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_transformation_config = self.config.get_data_transformation_config()

    def initiate_data_transformation(self):
        logger.info("Starting data transformation....")
        data_transformation = DataTransformation(config=self.data_transformation_config)
        data_transformation.data_transformation()