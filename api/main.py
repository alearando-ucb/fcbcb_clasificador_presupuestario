from datetime import datetime
from fastapi import FastAPI
from api.schemas import HealthResponse
from api.routers import clasificador

app = FastAPI(
    title="Clasificador de Partidas Presupuestarias FCBCB",
    description="API REST para clasificar partidas presupuestarias a partir de la descripcion de un item",
    version="1.0.0",
)

app.include_router(clasificador.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )