import joblib
from typing import Tuple, List
import numpy as np

from src.config import NB_MODEL_FILE, TFIDF_VECTORIZER_FILE, LABEL_ENCODER_FILE
from src.features import limpiar_texto

model = None
tfidf = None
label_encoder = None


def load_models():
    global model, tfidf, label_encoder
    if model is None:
        model = joblib.load(NB_MODEL_FILE)
        tfidf = joblib.load(TFIDF_VECTORIZER_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)


def predecir(item: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    load_models()

    texto_limpio = limpiar_texto(item)
    X = tfidf.transform([texto_limpio])

    probabilidades = model.predict_proba(X)[0]
    clase_idx = np.argmax(probabilidades)
    confianza = probabilidades[clase_idx]

    partida_codigo = label_encoder.inverse_transform([clase_idx])[0]
    partida_str = str(partida_codigo)

    probas_ordenadas = sorted(
        zip(label_encoder.classes_, probabilidades),
        key=lambda x: x[1],
        reverse=True
    )
    top_probabilidades = [(str(p[0]), float(p[1])) for p in probas_ordenadas[:3]]

    return partida_str, float(confianza), top_probabilidades