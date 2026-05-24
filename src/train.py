import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    TRAINING_FILE, ENCODING, CSV_SEPARATOR,
    NB_MODEL_FILE, TFIDF_VECTORIZER_FILE, LABEL_ENCODER_FILE,
    MODELS_DIR, REPORTS_DIR
)
from src.features import limpiar_texto


MIN_SAMPLES = 10


def cargar_datos():
    df = pd.read_csv(TRAINING_FILE, sep=CSV_SEPARATOR, encoding=ENCODING)
    df.columns = [c.strip() for c in df.columns]
    return df


def filtrar_partidas(df):
    counts = df['PARTIDA'].value_counts()
    valid_partidas = counts[counts >= MIN_SAMPLES].index
    return df[df['PARTIDA'].isin(valid_partidas)].copy()


def entrenar():
    print("Cargando datos...")
    df = cargar_datos()
    print(f"Total registros: {len(df)}")

    print("Filtrando partidas con < {} muestras...".format(MIN_SAMPLES))
    df = filtrar_partidas(df)
    print(f"Registros después del filtrado: {len(df)}")
    print(f"Partidas restantes: {df['PARTIDA'].nunique()}")

    print("Limpiando texto...")
    df['texto_limpio'] = df['ITEM'].apply(limpiar_texto)

    X = df['texto_limpio'].values
    y = df['PARTIDA'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Entrenando TF-IDF + ComplementNB...")
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_tfidf = tfidf.fit_transform(X)

    model = ComplementNB(alpha=0.01)
    model.fit(X_tfidf, y_encoded)

    print("Calculando accuracy con cross-validation...")
    cv_scores = cross_val_score(model, X_tfidf, y_encoded, cv=5, scoring='accuracy')
    accuracy = float(cv_scores.mean())
    print(f"Accuracy CV: {accuracy:.4f} (+/- {cv_scores.std():.4f})")

    print("Persistiendo modelos...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, NB_MODEL_FILE)
    joblib.dump(tfidf, TFIDF_VECTORIZER_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)

    metadata = {
        "accuracy_cv": round(accuracy, 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "n_partidas": int(df['PARTIDA'].nunique()),
        "n_muestras": int(len(df)),
        "partidas": sorted(df['PARTIDA'].unique().tolist()),
        "model_type": "ComplementNB",
        "alpha": 0.01,
        "tfidf_max_features": 1000,
        "tfidf_ngram_range": [1, 2],
        "min_samples_threshold": MIN_SAMPLES
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Entrenamiento completado.")
    return accuracy


if __name__ == "__main__":
    entrenar()