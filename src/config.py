from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PARTIDAS_DIR = DATA_DIR / "partidas"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TRAINING_FILE = DATA_RAW_DIR / "DATOS ENTRENAMIENTO.csv"
PARTIDAS_FILE = DATA_PARTIDAS_DIR / "partidas.csv"

NB_MODEL_FILE = MODELS_DIR / "nb_model.joblib"
TFIDF_VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.joblib"

ENCODING = "latin-1"
CSV_SEPARATOR = ";"
PARTIDAS_SEPARATOR = ","
PARTIDAS_ENCODING = "utf-8-sig"