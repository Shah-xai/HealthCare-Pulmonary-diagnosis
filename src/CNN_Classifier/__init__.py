import logging
from pathlib import Path
# Logging path
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "cnn_classifier.log"
# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s -%(module)s- %(message)s', 
                    handlers=[logging.FileHandler(LOG_FILE),
                               logging.StreamHandler()])
logger=logging.getLogger(__name__)