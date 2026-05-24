import pandas as pd
from src.config import PARTIDAS_FILE, PARTIDAS_ENCODING, PARTIDAS_SEPARATOR

_partidas_cache = None


def cargar_partidas():
    global _partidas_cache
    if _partidas_cache is None:
        _partidas_cache = pd.read_csv(PARTIDAS_FILE, encoding=PARTIDAS_ENCODING, sep=PARTIDAS_SEPARATOR)
    return _partidas_cache


def obtener_descripcion_partida(partida) -> str:
    df = cargar_partidas()
    try:
        partida_int = int(partida)
    except (ValueError, TypeError):
        return "Descripcion no disponible"
    match = df[df['PARTIDA'] == partida_int]
    if not match.empty:
        return match.iloc[0]['descripcion']
    return "Descripcion no disponible"