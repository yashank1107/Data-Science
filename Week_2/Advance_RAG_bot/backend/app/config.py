from pydantic_settings import BaseSettings
from typing import Dict, Any, List, Optional
import os

class Settings(BaseSettings):
    # API Keys - Let Pydantic handle loading from .env
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    
    # Observability
    OPIK_API_KEY: str = ""
    OPIK_ENDPOINT: str = "http://localhost:4318"
    
    # LLM Configurations
    GEMINI_MODELS: List[str] = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash"]
    GROQ_MODELS: List[str] = ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"]
    COHERE_MODELS: List[str] = ["command-a-03-2025", "command-r-plus-08-2024", "command-a-vision-07-2025"]
    
    @property
    def AVAILABLE_LLMS(self) -> List[str]:
        all_models = []
        if self.GEMINI_API_KEY:
            all_models.extend([f"gemini:{model}" for model in self.GEMINI_MODELS])
        if self.GROQ_API_KEY:
            all_models.extend([f"groq:{model}" for model in self.GROQ_MODELS])
        if self.COHERE_API_KEY:
            all_models.extend([f"cohere:{model}" for model in self.COHERE_MODELS])
        return all_models
    
    # RAG Configurations
    RAG_VARIANTS: List[str] = ["basic", "knowledge_graph", "hybrid"]
    
    # Vector Store
    VECTOR_STORE_PATH: str = "./data/vector_store"
    
    # Document Processing
    SUPPORTED_FILE_TYPES: List[str] = [".pdf", ".txt", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".pptx", ".xlsx"]
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Security
    GUARDRAIL_MODEL: str = "microsoft/DialoGPT-medium"
    
    class Config:
        # Build an absolute path to the .env file which is two directories up from this config file.
        # This makes loading the .env file independent of the current working directory.
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
        env_file_encoding = 'utf-8'

settings = Settings()