from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel


class ClasificacionMeta(BaseModel):
    item_procesado: str
    timestamp: str


class ClasificacionDetalle(BaseModel):
    partida_predicha: str
    descripcion_partida: str
    confianza: float


class Alternativa(BaseModel):
    partida: str
    descripcion: str
    confianza: float


class ClasificacionResponseExito(BaseModel):
    status: str = "exito"
    meta: ClasificacionMeta
    clasificacion: ClasificacionDetalle
    distribucion_probabilidades: Dict[str, float]


class ClasificacionResponseAmbiguo(BaseModel):
    status: str = "ambiguo"
    meta: ClasificacionMeta
    clasificacion: ClasificacionDetalle
    distribucion_probabilidades: Dict[str, float]
    alternativas: List[Alternativa]


class HealthResponse(BaseModel):
    status: str
    timestamp: str