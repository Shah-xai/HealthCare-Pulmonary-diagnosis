from CNN_Classifier import logger   
from CNN_Classifier.config.configuration import DataTransformationConfig
from PIL import Image
from pathlib import Path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @staticmethod
    def _data_transformer(config, split: str) -> None:
        logger.info(f"Transforming data for split: {split}")
        image_src_dir = Path(config.raw_data_dir)/split
        transformed_data_dir = Path(config.root_dir) / f"transformed_{split}"
        class_dirs = [d for d in image_src_dir.iterdir() if d.is_dir()]
        for class_dir in class_dirs:
            transformed_class_dir = transformed_data_dir / class_dir.name
            transformed_class_dir.mkdir(parents=True, exist_ok=True)
            for image_path in class_dir.rglob("*"):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png",".tiff",".tif"]:
                    try:
                        with Image.open(image_path) as img:
                            img = img.convert("RGB")
                            img = img.resize(config.target_size)
                            img.save(transformed_class_dir / f"{image_path.stem}.png", format="PNG", optimize=True)
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {e}")
    def data_transformation(self) -> None:
        logger.info("Starting data transformation process")
        splits = ["train","valid","test"]
        for split in splits:
            self._data_transformer(self.config, split)
        logger.info("Data transformation process completed successfully")
