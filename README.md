# Clasificador de Partidas Presupuestarias FCBCB

Este proyecto es una API REST que permite clasificar partidas presupuestarias a partir de la descripcion de un item. Por ejemplo, si se ingresa "boligrafo micropunta color azul", el modelo devuelve la partida presupuestaria mas probable junto con el nivel de confianza de la prediccion.

El modelo utilizado es ComplementNB (Naive Bayes Complementario), seleccionado por su buen rendimiento en pruebas externas con datos no vistos.

---

## Requisitos previos

- **Python 3.11** (la version 3.13 NO es compatible)
- pip (administrador de paquetes de Python)
- Docker (opcional, para ejecucion en contenedor)

---

## Estructura del proyecto

```
fcbcb_clasificador_presupuestario/
├── api/                    # Aplicacion FastAPI
│   ├── routers/
│   │   └── clasificador.py
│   ├── schemas.py
│   └── main.py
├── data/
│   ├── raw/                # Datos de entrenamiento
│   │   └── DATOS ENTRENAMIENTO.csv
│   └── partidas/          # Tabla de referencia de partidas
│       └── partidas.csv
├── models/                # Modelos persistidos (TF-IDF, NB, LabelEncoder)
├── reports/                # Metadata del entrenamiento
├── scripts/
│   └── entrenar.py        # Script para reentrenar
├── src/
│   ├── config.py          # Rutas y constantes
│   ├── data.py            # Carga de datos y lookup de partidas
│   ├── features.py       # Limpieza de texto
│   ├── train.py           # Pipeline de entrenamiento
│   └── predict.py         # logica de prediccion
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Instalacion

**Importante**: Este proyecto requiere **Python 3.11** (no usar 3.12 ni 3.13).

1. Verificar la version de Python instalada:

```
python --version
```

2. Crear un entorno virtual de Python:

```
python -m venv venv
```

3. Activar el entorno virtual:

En Windows (PowerShell):
```
.\venv\Scripts\Activate.ps1
```

En Windows (CMD):
```
venv\Scripts\activate.bat
```

En Linux o macOS:
```
source venv/bin/activate
```

4. Instalar las dependencias:

```
pip install -r requirements.txt
```

---

## Como reentrenar el modelo

Si tienes un nuevo dataset de entrenamiento, sigue estos pasos para reentrenar el modelo.

### Paso 1: Preparar el archivo de entrenamiento

El archivo de entrenamiento debe ser un CSV con dos columnas separadas por punto y coma (`;`):

- `ITEM`: la descripcion del item o producto
- `PARTIDA`: el codigo numerico de la partida presupuestaria

Ejemplo del formato esperado:

```
ITEM;PARTIDA
IMPRESION DE TALONARIOS DE FACTURAS;25600
CONTRATACION DE PASANTIAS;26930
ADQUISICION DE LIBROS;32300
```

Guarda este archivo como `data/raw/DATOS ENTRENAMIENTO.csv`.

### Paso 2: Verificar la tabla de partidas

El archivo `data/partidas/partidas.csv` contiene la descripcion de cada partida. Asegurate de que este archivo tenga el siguiente formato:

```
PARTIDA;descripcion
21600;Energia Electrica
22110;Pasajes Aereos
25600;Imprenta
...
```

Cada partida que aparezca en tu dataset de entrenamiento debe tener una entrada en este archivo. Si falta alguna, el sistema no podra devolver la descripcion de esa partida en las respuestas.

### Paso 3: Ejecutar el script de entrenamiento

Con el entorno virtual activado, ejecuta:

```
python scripts/entrenar.py
```

El script realizara las siguientes operaciones:

- Cargara los datos desde `data/raw/DATOS ENTRENAMIENTO.csv`
- Eliminara las partidas que tengan menos de 10 muestras (para garantizar un minimo de representatividad)
- Limpiara y normalizara el texto de las descripciones
- Entrenara el vectorizador TF-IDF con unigrams y bigrams
- Entrenara el clasificador ComplementNB
- Calculata el accuracy mediante validacion cruzada con 5 folds
- Guardara los modelos en la carpeta `models/`
- Guardara la metadata del entrenamiento en `reports/training_metadata.json`

Al finalizar, el script imprimira en pantalla el accuracy obtenu, el numero de partidas restantes y el total de muestras utilizadas.

### Paso 4: Verificar los resultados

Despues del entrenamiento, podras ver los siguientes archivos actualizados:

- `models/nb_model.joblib`: el modelo clasificador
- `models/tfidf_vectorizer.joblib`: el vectorizador TF-IDF
- `models/label_encoder.joblib`: el codificador de etiquetas
- `reports/training_metadata.json`: metricas y configuracion del entrenamiento

Si el accuracy obtenido es inferior a lo esperado, puedes ajustar los parametros en `src/train.py`, tales como `alpha` del clasificador, `max_features` del TF-IDF, o el umbral minimo de muestras por partida.

---

## Como ejecutar la API

### Ejecucion local (sin Docker)

1. Activar el entorno virtual:

En Windows:
```
venv\Scripts\activate
```

En Linux o macOS:
```
source venv/bin/activate
```

2. Ejecutar el servidor:

```
python -m uvicorn api.main:app --reload
```

La API estara disponible en `http://localhost:8000`.

### Probando la API con Swagger

La forma recomendada de probar el clasificador es mediante la interfaz Swagger en:

**`http://localhost:8000/docs`**

Swagger permite:
- Ver todos los endpoints disponibles
- Probar el clasificador directamente desde el navegador
- Visualizar las respuestas JSON de forma estructurada

Para clasificar un item:
1. Ir a `http://localhost:8000/docs`
2. Hacer clic en el endpoint `GET /clasificador/clasificar`
3. Hacer clic en "Try it out"
4. Ingresar la descripcion del item (ej: "boligrafo micropunta color azul")
5. Ejecutar y ver la respuesta

### Endpoints disponibles

#### Clasificacion de items

```
GET /clasificador/clasificar?item=descripcion del item
```

Este endpoint recibe la descripcion de un item y devuelve la partida presupuestaria mas probable junto con el nivel de confianza. El parametro `item` es obligatorio y debe ser una cadena de texto con la descripcion del producto o servicio a clasificar.

Respuesta exitosa (cuando la confianza es mayor o igual a 0.9):

```json
{
  "status": "exito",
  "meta": {
    "item_procesado": "boligrafo micropunta color azul",
    "timestamp": "2026-05-23T21:48:10Z"
  },
  "clasificacion": {
    "partida_predicha": "39500",
    "descripcion_partida": "Utiles de Escritorio y Oficina",
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

Respuesta ambigua (cuando la confianza es menor a 0.9):

```json
{
  "status": "ambiguo",
  "meta": {
    "item_procesado": "boligrafo micropunta color azul",
    "timestamp": "2026-05-23T21:48:10Z"
  },
  "clasificacion": {
    "partida_predicha": "39500",
    "descripcion_partida": "Utiles de Escritorio y Oficina",
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

El campo `status` indica si la prediccion es confiable. Si es "success", la confianza es alta y la prediccion es fiable. Si es "ambiguo", la confianza es baja y se incluyen alternativas para que el usuario pueda elegir entre varias opciones. El campo `distribucion_probabilidades` muestra las tres partidas mas probables y un acumulado llamado "otros" que agrupa el resto de la probabilidad.

#### Verificacion del servicio

```
GET /health
```

Este endpoint permite verificar que la API esta corriendo correctamente. Devuelve un mensaje de confirmacion con la hora del servidor.

```json
{
  "status": "ok",
  "timestamp": "2026-05-23T21:48:10Z"
}
```

---

## Ejecucion con Docker

### Requisitos

- Docker instalado
- Docker Compose

### Construccion y ejecucion

Desde la raiz del proyecto:

```
docker compose up --build
```

La API estara disponible en `http://localhost:8000`.

### Probando con Swagger

Ir a **`http://localhost:8000/docs`** para probar el clasificador desde la interfaz interactiva de Swagger.

### Detener el contenedor

```
docker compose down
```

---

## Notas tecnicas

El modelo clasificador filtra automaticamente las partidas que tengan menos de 10 muestras en el conjunto de entrenamiento. Esto se hace para evitar que clases con muy pocos ejemplos generen predicciones poco confiables. Si necesitas cambiar este umbral, modifica la constante `MIN_SAMPLES` en `src/train.py`.

El umbral de confianza para distinguir entre respuestas exitosas y ambiguas es de 0.9. Si la probabilidad maxima predicha por el modelo es menor a este valor, la respuesta incluyera alternativas con las siguientes probabilidades mas altas. Para modificar este umbral, busca la constante correspondiente en `api/routers/clasificador.py`.

---

## Dependencias del proyecto

- scikit-learn==1.5.1
- fastapi==0.111.0
- uvicorn==0.30.1
- pydantic==2.7.4
- pandas==2.2.2
- joblib==1.4.2