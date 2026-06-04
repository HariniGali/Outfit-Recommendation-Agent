import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
CATALOG_CSV_PATH = os.environ.get('CATALOG_CSV_PATH', './data/florence2_descriptions.csv')
IMAGE_DIR = os.environ.get('IMAGE_DIR', './images')
CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', './chroma_fashion_db')
USE_S3 = os.environ.get('USE_S3', 'false').lower() == 'true'
IMAGE_BASE_URL = os.environ.get('IMAGE_BASE_URL', '')
FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.environ.get('FLASK_PORT', '5000'))
