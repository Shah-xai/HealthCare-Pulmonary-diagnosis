from pathlib import Path

from CNN_Classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from CNN_Classifier.utils.common import read_yaml,create_directories
from CNN_Classifier.entity.config_entity import (DataIngestionConfig,
                                                  BaseModelConfig, 
                                                  DataTransformationConfig,
                                                  ModelTrainingConfig)

class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
        )
        return data_ingestion_config
    def get_base_model_config(self)-> BaseModelConfig:
        config = self.config.base_model_preparation
        params = self.params
        create_directories([config.root_dir])
        base_model_config = BaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=Path(config.base_model_dir),
            updated_base_model_path=Path(config.updated_model_dir),
            feature_extract_dir=Path(config.feature_extract_dir),
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES,
            params_input_shape=tuple(params.INPUT_SHAPE),
            params_classifier_activation=params.CLASSIFIER_ACTIVATION,
            params_pooling=params.POOLING
        )
        return base_model_config
    def get_data_transformation_config(self)->DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            raw_data_dir=config.raw_data_dir,
            root_dir=config.root_dir,
            target_size=tuple(params.TARGET_IMAGE_SIZE),
            seed=params.SEED
        )
        return data_transformation_config
    
    def get_model_training_config(self)->ModelTrainingConfig:
        config = self.config.model_training
        params = self.params
        create_directories([config.root_dir])
        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            training_data_dir=config.training_data_dir,
            validation_data_dir=config.validation_data_dir,
            updated_model_dir=config.updated_model_dir,
            feature_extract_dir=config.feature_extract_dir,
            trained_model_dir=config.trained_model_dir,
            trained_model_dir_svm=config.trained_model_dir_svm,
            trained_model_dir_pca_svm=config.trained_model_dir_pca_svm,
            trained_model_dir_kpca_svm=config.trained_model_dir_kpca_svm,
            SEED=params.SEED,
            EPOCHS=params.EPOCHS,
            PATIENCE=params.PATIENCE,
            IMAGE_SIZE=tuple(params.TARGET_IMAGE_SIZE),
            BATCH_SIZE=params.BATCH_SIZE
        )
        return model_training_config