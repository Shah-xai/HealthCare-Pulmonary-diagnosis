from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
    def initiate_data_ingestion(self):
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_data()
        except Exception as e:
            logger.exception(f"An error occurred during data ingestion: {e}")
            raise e
if __name__ == "__main__":
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()