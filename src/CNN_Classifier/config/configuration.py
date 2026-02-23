from CNN_Classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from CNN_Classifier.utils.common import read_yaml,create_directories
from CNN_Classifier.entity.config_entity import DataIngestionConfig, BaseModelConfig

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
            base_model_path=config.base_model_dir,
            updated_base_model_path=config.updated_model_dir,
            feature_extract_dir=config.feature_extract_dir,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES,
            params_input_shape=tuple(params.INPUT_SHAPE),
            params_classifier_activation=params.CLASSIFIER_ACTIVATION,
            params_pooling=params.POOLING
        )
        return base_model_config