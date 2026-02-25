from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path

@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    feature_extract_dir: Path
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_input_shape: tuple
    params_learning_rate: float
    params_classifier_activation: str
    params_pooling: str

