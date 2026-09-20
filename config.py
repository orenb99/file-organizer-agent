import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Model + log file are env-configurable; src/dst/batch-size come from the
# CLI instead (see main.py) since those change on every run.
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-31b-it")
LOG_FILE = Path(os.getenv("LOG_FILE", "organizer.log"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
