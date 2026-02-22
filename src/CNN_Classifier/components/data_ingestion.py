from CNN_Classifier import logger
from CNN_Classifier.entity.config_entity import DataIngestionConfig
from CNN_Classifier.utils.common import extract_zip
from pathlib import Path
import kagglehub
import shutil

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def download_data(self):
        logger.info("Starting data ingestion process...")

        target_path: Path = Path(self.config.local_data_file) 
        if target_path.exists():
            logger.info("Dataset already exists. Skipping download.")
        else:
            # 1) download to kaggle cache
            cache_path = Path(kagglehub.dataset_download(self.config.source_URL))
            logger.info(f"Data downloaded from Kaggle cache to {cache_path}.")

            # 2) copy into your desired project directory
            logger.info(f"Copying dataset from {cache_path} to {target_path}.")
            shutil.copytree(cache_path, target_path)

            # 3) clean folders in YOUR project copy
            logger.info(f"Cleaning folder names under {target_path}.")
        
        self.clean_split_folders()

        logger.info("Data ingestion process completed successfully.")
        return target_path
    
    from pathlib import Path

    def clean_split_folders(self):
        root = Path(self.config.local_data_file)
        subdirs =[d for d in root.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            subdir=subdirs[0]
            root = subdir
        # The splits we want to clean
        for split in ["train", "valid", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                continue

            patterns = {
                "*normal*": "normal",
                "*adeno*": "adenocarcinoma",
                "*large*": "large.cell.carcinoma",
                "*squamous*": "squamous.cell.carcinoma",
            }

            for pattern, clean_name in patterns.items():
                for folder in split_dir.glob(pattern):
                    if folder.is_dir() and folder.name != clean_name:
                        target = split_dir / clean_name
                        if not target.exists():
                            folder.rename(target)
                            logger.info(f"[{split}] {folder.name} → {clean_name}")
                        else:
                            logger.info(f"[{split}] Skipping {folder.name} (target exists)")

