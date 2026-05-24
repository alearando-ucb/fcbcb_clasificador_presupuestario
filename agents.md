# Plan de Desarrollo: Clasificador de Partidas Presupuestarias FCBCB

## Objetivo
Desplegar un modelo de clasificación de partidas presupuestarias como API REST con Swagger, usando FastAPI + Docker.

**Input**: descripción de item (ej: "bolígrafo micropunta color azul")
**Output (confianza >= 0.9)**:
```json
{
  "status": "success",
  "meta": {
    "item_procesado": "bolígrafo micropunta color azul",
    "timestamp": "2026-05-23T21:48:10Z"
  },
  "clasificacion": {
    "partida_predicha": "39500",
    "descripcion_partida": "Útiles de Escritorio y Oficina",
    "confianza": 0.9452
  },
  "distribucion_probabilidades": {
    "39500": 0.9452,
    "32200": 0.0210,
    "39100": 0.0115,
    "otros": 0.0223
  }
}
```

**Output (confianza < 0.9 - respuesta ambigua)**:
```json
{
  "status": "ambiguo",
  "meta": {
    "item_procesado": "bolígrafo micropunta color azul",
    "timestamp": "2026-05-23T21:48:10Z"
  },
  "clasificacion": {
    "partida_predicha": "39500",
    "descripcion_partida": "Útiles de Escritorio y Oficina",
    "confianza": 0.4523
  },
  "distribucion_probabilidades": {
    "39500": 0.4523,
    "32200": 0.3210,
    "39100": 0.1567,
    "otros": 0.07
  },
  "alternativas": [
    {"partida": "32200", "descripcion": "Materiales de Oficina", "confianza": 0.3210},
    {"partida": "39100", "descripcion": "Muebles y Enseres", "confianza": 0.1567}
  ]
}
```

---

## Arquitectura de Directorios

```
mlops_churn_project/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app (swagger en /docs)
│   ├── routers/
│   │   ├── __init__.py
│   │   └── clasificador.py  # Endpoint /clasificar
│   └── schemas.py           # Pydantic models
├── data/
│   ├── raw/
│   │   └── DATOS ENTRENAMIENTO.csv
│   └── partidas/            # Lookup table partidas
│       └── partidas.csv    # PARTIDA,descripcion
├── models/                  # Modelos persistidos
├── reports/
│   └── training_metadata.json
├── src/
│   ├── __init__.py
│   ├── config.py            # Rutas y constantes
│   ├── data.py             # Carga de datos
│   ├── features.py         # TF-IDF + text cleaning
│   ├── train.py            # Pipeline entrenamiento
│   ├── predict.py          # Predicción con probabilidades
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_clasificador.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

## Hitos de Desarrollo

### Hito 1: Estructura Base y Datos
**Aprobación requerida**: Verificar estructura de directorios y presencia de archivos

- [ ] Crear directorios `data/partidas/`, `api/routers/`
- [ ] Crear `src/features.py` con función `limpiar_texto()` del notebook:
  ```python
  def limpiar_texto(texto):
      if pd.isna(texto):
          return ""
      texto = str(texto).lower()
      texto = re.sub(r'[^\w\sáéíóúñ]', ' ', texto)  # elimina caracteres especiales pero mantiene letras con acentos
      texto = re.sub(r'\d+', '', texto)  # elimina todos los números
      texto = re.sub(r'\s+', ' ', texto)  # normaliza espacios
      texto = texto.strip()
      return texto
  ```
- [ ] Crear `data/partidas/partidas.csv` con formato:
  ```
  PARTIDA,descripcion
  39500,Útiles de Escritorio y Oficina
  21600,Energía Eléctrica
  ...
  ```
- [ ] Crear `src/data.py` para carga del CSV de partidas
- [ ] Ajustar `src/config.py` con nuevas rutas

**Commit al aprobar**: `feat: establish base directory structure and partida lookup`

---

### Hito 2: Pipeline de Entrenamiento
**Aprobación requerida**: Validar métricas del modelo (accuracy > 75%)

- [ ] Migrar lógica del notebook a `src/train.py`:
  - Carga `DATOS ENTRENAMIENTO.csv` (sep=';', encoding='latin-1')
  - Limpieza de texto (`limpiar_texto`)
  - **Filtrado de partidas con < 10 muestras** (41 partidas eliminadas, 29 clases restantes)
  - TF-IDF Vectorizer (max_features=1000, ngram_range=(1,2), min_df=2, max_df=0.95)
  - Entrenamiento **ComplementNB** (alpha=0.01) - mejor modelo en pruebas externas
  - LabelEncoder para mapeo partidas
- [ ] Persistir: modelo (`modelos/nb_model.joblib`), tfidf (`modelos/tfidf_vectorizer.joblib`), label_encoder (`modelos/label_encoder.joblib`)
- [ ] Guardar metadata de entrenamiento en `reports/training_metadata.json`
- [ ] Crear `scripts/entrenar.py` ejecutable

**Commit al aprobar**: `feat: implement training pipeline with ComplementNB model`

---

### Hito 3: Endpoint de Predicción
**Aprobación requerida**: Probar endpoint con item ejemplo "bolígrafo micropunta color azul"

- [ ] Crear `api/schemas.py` con Pydantic models:
  - `ClasificacionRequest(item: str)`
  - `ClasificacionResponse` (estructura JSON objetivo con status, meta, clasificacion, distribucion_probabilidades)
- [ ] Crear `api/routers/clasificador.py`:
  - `GET /clasificar?item=...`
  - Limpiar texto input
  - Vectorizar con TF-IDF cargado
  - Obtener probabilidades del modelo (predict_proba)
  - **Umbral de confianza: 0.9**
    - Si max_probabilidad >= 0.9: status="success"
    - Si max_probabilidad < 0.9: status="ambiguo" (incluir campo `alternativas` con top-3)
  - Lookup de descripción de partida desde `data/partidas/partidas.csv`
- [ ] Registrar router en `api/main.py`
- [ ] Configurar Swagger en `api/main.py` (FastAPI automática en `/docs`)
- [ ] Endpoint `/health` para verificación

**Commit al aprobar**: `feat: add /clasificar endpoint with confidence threshold`

---

### Hito 4: Dockerización
**Aprobación requerida**: Verificar que `docker-compose up` levanta el API en http://localhost:8000/docs

- [ ] Crear/actualizar `Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  # Copiar datos y modelos
  COPY data/ ./data/
  COPY models/ ./models/
  EXPOSE 8000
  CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] Actualizar `docker-compose.yml` si es necesario
- [ ] Actualizar `requirements.txt` (ya tiene fastapi, uvicorn, pydantic)
- [ ] Probar build y ejecución local

**Commit al aprobar**: `feat: dockerize API with FastAPI and Swagger`

---

### Hito 5: Tests y Validación
**Aprobación requerida**: Ejecutar `pytest` y verificar cobertura

- [ ] Crear `tests/test_clasificador.py`:
  - Test text cleaning (`limpiar_texto`)
  - Test predicción con item conocido
  - Test estructura de respuesta JSON
  - Test umbral de confianza (success vs ambiguo)
- [ ] Crear `tests/test_api.py`:
  - Test `/health`
  - Test `/clasificar` con varios items (alta y baja confianza)
  - Test código de estado HTTP
- [ ] Ejecutar suite de tests

**Commit al aprobar**: `feat: add unit and integration tests`

---

### Hito 6: Documentación Final
**Aprobación requerida**: Revisar README.md y Swagger

- [ ] Documentar en `README.md`:
  - Uso local (python -m uvicorn)
  - Uso Docker
  - Ejemplo de request/response
  - Formato del CSV partidas
- [ ] Verificar que Swagger (`/docs`) muestra todos los endpoints
- [ ] Actualizar `model_card.md` si existe

**Commit al aprobar**: `docs: complete API documentation and usage guide`

---

## Notas Técnicas

### Del Notebook al Proyecto
| Elemento | Notebook | Proyecto |
|----------|----------|----------|
| Text cleaning | `limpiar_texto()` inline | `src/features.py` |
| Vectorizer | TfidfVectorizer inline | `src/features.py` |
| **Model** | **SVM RBF (entrenamiento) / Naive Bayes (pruebas externas)** | `src/train.py` |
| Labels | LabelEncoder inline | `src/data.py` |
| Partidas lookup | No existe | `data/partidas/partidas.csv` |

### Decisión del Modelo
- **SVM RBF**: Mejor accuracy en cross-validation (0.8009) pero peor en pruebas externas
- **Naive Bayes (ComplementNB)**: Mejor en pruebas externas con datos no vistos
- Se usa **ComplementNB** para producción por su mejor generalización

### Dependencias (ya en requirements.txt)
- scikit-learn==1.5.1
- fastapi==0.111.0
- uvicorn==0.30.1
- pydantic==2.7.4
- pandas==2.2.2
- joblib==1.4.2

### Fórmulas de Probabilidad
- El modelo ComplementNB con `predict_proba()` retorna probabilidades para cada clase
- La respuesta `distribucion_probabilidades` muestra el top-3 más el acumulado "otros"
- `otros` = 1.0 - suma(top-3 probabilidades)