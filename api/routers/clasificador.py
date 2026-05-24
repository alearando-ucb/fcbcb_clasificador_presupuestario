from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from api.schemas import (
    ClasificacionResponseExito,
    ClasificacionResponseAmbiguo,
    ClasificacionMeta,
    ClasificacionDetalle,
    Alternativa,
)
from src.predict import predecir
from src.data import obtener_descripcion_partida

router = APIRouter(prefix="/clasificador", tags=["clasificador"])

CONFIANZA_UMBRAL = 0.9


@router.get("/clasificar", response_model_by_alias=False)
def clasificar(item: str = Query(..., min_length=1, description="Descripcion del item a clasificar")):
    if not item or not item.strip():
        raise HTTPException(status_code=400, detail="El parametro 'item' no puede estar vacio")

    partida_predicha, confianza, top_probabilidades = predecir(item)

    descripcion = obtener_descripcion_partida(partida_predicha)

    distribucion = {}
    for partida, prob in top_probabilidades:
        distribucion[partida] = prob
    otros = 1.0 - sum(prob for _, prob in top_probabilidades)
    if otros > 0:
        distribucion["otros"] = round(otros, 4)

    meta = ClasificacionMeta(
        item_procesado=item.strip(),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    clasificacion = ClasificacionDetalle(
        partida_predicha=partida_predicha,
        descripcion_partida=descripcion,
        confianza=round(confianza, 4)
    )

    if confianza >= CONFIANZA_UMBRAL:
        return ClasificacionResponseExito(
            status="exito",
            meta=meta,
            clasificacion=clasificacion,
            distribucion_probabilidades=distribucion
        )
    else:
        alternativas = []
        for partida, prob in top_probabilidades[1:]:
            desc = obtener_descripcion_partida(partida)
            alternativas.append(Alternativa(
                partida=partida,
                descripcion=desc,
                confianza=round(prob, 4)
            ))

        return ClasificacionResponseAmbiguo(
            status="ambiguo",
            meta=meta,
            clasificacion=clasificacion,
            distribucion_probabilidades=distribucion,
            alternativas=alternativas
        )