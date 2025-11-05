import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "ml_models", "best_model.pkl")
ENCODERS_PATH = os.path.join(PROJECT_ROOT, "ml_models", "label_encoders.pkl")

