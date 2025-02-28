from typing import List
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """
    Application settings.
    """
    PROJECT_NAME: str = "DLW 2025 API"
    PROJECT_DESCRIPTION: str = "A FastAPI server for the DLW 2025 project."
    VERSION: str = "1.0.0"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = "gpt-4o-mini"
    
    # File upload settings
    UPLOAD_FOLDER: str = "uploads"


# Create global settings object
settings = Settings()