import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    MODEL_PATH = "./models/trained_classifier"
    CLASSIFICATION_THRESHOLD = 0.7
    
    # API settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    
    # PII Detection settings
    PII_CONFIDENCE_THRESHOLD = 0.8