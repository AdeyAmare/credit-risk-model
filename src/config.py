import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Simple project configuration.

    Attributes
    ----------
    PROJECT_ROOT : Path
        Root directory of the project.
    DATA_DIR : Path
        Directory where all project data is stored. Can be overridden
        using the DATA_DIR environment variable or a .env file.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "raw"))


