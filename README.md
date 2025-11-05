# Alpha Beta Predictor API

API de predicción basada en FastAPI para estimar valores Alpha y Beta utilizando un modelo pre-entrenado de CatBoost.

## Descripción

Este proyecto implementa una API REST que permite predecir valores Alpha y Beta basándose en características del cliente. Utiliza un modelo de machine learning pre-entrenado (CatBoost) y está construido con FastAPI para proporcionar una interfaz robusta y documentada.

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Acceso a una terminal o línea de comandos

## Instalación

1. Clona el repositorio o descarga los archivos del proyecto:
```bash
git clone https://github.com/arleyserna/Proyecto-Summa.git
cd alpha_betha_predictor_api
```

2. Crea un entorno virtual y actívalo:
```bash
# Windows
python -m venv .env
.env\Scripts\activate

# Linux/macOS
python3 -m venv .env
source .env/bin/activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
alpha_betha_predictor_api/
├── app/
│   ├── adapters/           # Adaptadores para transformación de datos
│   ├── data_models/        # Modelos de datos Pydantic
│   ├── ml_models/          # Modelos pre-entrenados
│   ├── routers/            # Rutas de la API
│   ├── services/           # Lógica de negocio
│   ├── utils/              # Utilidades y configuración
│   ├── main.py            # Punto de entrada de la aplicación
│   └── test.py            # Pruebas
└── requirements.txt        # Dependencias del proyecto
```

## Uso

1. Inicia el servidor de desarrollo:
```bash
cd alpha_betha_predictor_api
uvicorn app.main:app --reload
```

2. La API estará disponible en: `http://localhost:2500`, Puede probarla, con los datos 'data\to_predict.json'

3. Accede a la documentación interactiva en: `http://localhost:2500/docs`

4. En la carpeta Notebooks, se encuentran los cuadernos de Jupiter, en los cuales se desarrollaron el modelo de Forecast de la demanda y el clasificador Alpha, Betha.

5. Los modelos de predicción se encuentran en la carpeta *ml_models*

## Ejemplo de Uso

Puedes realizar predicciones enviando una solicitud POST a `/predict` con los datos del cliente:

```python
import requests

data = {
    "autoID": "9695-TERGH",
    "SeniorCity": "0",
    "Partner": "No",
    "Dependents": "No",
    "Service1": "Yes",
    "Service2": "No",
    "Security": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "Charges": "96.05",
    "Demand": "",
    "Class": ""
}

response = requests.post("http://localhost:2500/predict", json=data)
predictions = response.json()
print(predictions)
```

## Pruebas

Para ejecutar las pruebas del proyecto:

```bash
python -m pytest app/test.py
```

## Solución de Problemas Comunes

1. Error de importación de módulos:
   - Asegúrate de estar ejecutando los comandos desde el directorio raíz del proyecto
   - Verifica que el entorno virtual esté activado

2. Error al cargar el modelo:
   - Verifica que los archivos `.pkl` estén presentes en `app/ml_models/`
   - Asegúrate de haber instalado todas las dependencias

## Tecnologías Utilizadas

- FastAPI: Framework web para APIs
- CatBoost: Biblioteca de machine learning
- Pandas: Procesamiento de datos
- Scikit-learn: Herramientas de machine learning
- Uvicorn: Servidor ASGI para Python

## Autor

Hoover Serna Electronics Engineer | Contact: arleyserna@msn.com