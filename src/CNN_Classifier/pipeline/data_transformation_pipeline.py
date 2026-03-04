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

if __name__ == "__main__":
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()