from CNN_Classifier import logger
from CNN_Classifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from CNN_Classifier.pipeline.base_model_preparation_pipeline import BaseModelPreparationPipeline
from CNN_Classifier.pipeline.data_transformation_pipeline import DataTransformationPipeline

# logger.info("Starting the CNN Classifier application...")

# # Data ingestion
# logger.info("Ingesting data...")
# try:
#     data_ingestion_pipeline = DataIngestionPipeline()
#     data_ingestion_pipeline.initiate_data_ingestion()
# except Exception as e:
#     logger.error(f"Data ingestion failed: {e}")
# # Data transformation
logger.info("Transforming data...")
try:
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
except Exception as e:
    logger.error(f"Data transformation failed: {e}")

# # Base model preparation
# logger.info("Preparing the base model...")
# try:
#     base_model_preparation_pipeline = BaseModelPreparationPipeline()
#     base_model_preparation_pipeline.initiate_model_preparation()
# except Exception as e:
#     logger.error(f"Base model preparation failed: {e}")