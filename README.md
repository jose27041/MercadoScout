# MercadoScout — Clasificación binaria de scouting

Proyecto académico de analítica deportiva que estima si un futbolista pertenece a la categoría **Élite** o **Estándar** a partir de posición, edad y estatura.

## Regla de clasificación del objetivo
- **Élite (1):** valor de mercado igual o superior al percentil 75 (Top 25%).
- **Estándar (0):** valor de mercado inferior al percentil 75.

El umbral se calcula sobre la muestra de jugadores obtenida de MySQL durante el entrenamiento. La API devuelve la clase estimada y las probabilidades de ambas clases.

## Estructura principal
- `src/database/conector.py`: conexión y consulta de jugadores en MySQL.
- `src/pipeline/features.py`: preparación común de variables.
- `entrenar_modelo.py`: preprocesamiento, división estratificada, búsqueda GridSearchCV, Random Forest, evaluación y serialización.
- `models/modelo_scouting_rf.joblib`: modelo generado al entrenar.
- `api.py`: API REST FastAPI, endpoint `POST /predict`.
- `web/index.html`: interfaz web cliente.
- `clasificar_lote.py`: clasificación opcional desde CSV.

## Instalación
Se recomienda Python 3.10 o superior.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate

pip install -r requirements.txt
```

Configura las credenciales de MySQL mediante las variables de entorno que utiliza `src/database/conector.py`. No compartas el archivo `.env` ni incluyas credenciales reales en el ZIP.

## Ejecución
Desde la carpeta raíz del proyecto:

```bash
python entrenar_modelo.py
uvicorn api:app --reload --host 127.0.0.1 --port 8001
```

Abre `http://127.0.0.1:8001/` para usar la interfaz y `http://127.0.0.1:8001/docs` para probar la API.

## Ejemplo de solicitud
`POST /predict`

```json
{
  "nombre": "Nuevo prospecto",
  "posicion": "Midfield",
  "edad": 21,
  "estatura_cm": 178
}
```

La respuesta incluye `prediccion` (Élite/Estándar), `clase` (1/0), `probabilidad_elite`, `confianza` y el porcentaje estimado para cada clase.

**Importante:** las predicciones son estimaciones estadísticas basadas en las variables disponibles. No equivalen a una valoración profesional completa del futbolista.


## Experimentos y evidencias del entrenamiento

Al ejecutar `python entrenar_modelo.py` se generan:

- `reports/figures/overfitting_arboles.png`: Accuracy Train/Test del árbol sin poda y del árbol con pre-poda.
- `reports/figures/curva_roc_comparativa.png`: curvas ROC del árbol sin poda, árbol con pre-poda y Random Forest.
- `reports/figures/feature_importance.png`: importancia Gini del Random Forest.
- `reports/figures/matriz_confusion.png`: matriz de confusión del Random Forest.
- `reports/metricas.json`: métricas del Random Forest y resultados de los dos árboles.
- `models/modelo_scouting_rf.joblib`: pipeline final de Random Forest.

El preprocesamiento ahora incorpora `MedianaAgrupadaPorPosicion`, que aprende las medianas numéricas por posición exclusivamente a partir de los datos usados para ajustar cada pipeline. Si una posición no tiene mediana disponible para una variable, utiliza la mediana global aprendida en entrenamiento.

Los parámetros de pre-poda utilizados para el experimento comparativo son `max_depth=8`, `min_samples_leaf=5` y `min_samples_split=10`. El árbol sin poda usa `max_depth=None`, `min_samples_leaf=1` y `min_samples_split=2`. Estos valores son configuraciones experimentales explícitas y deben interpretarse junto con los resultados obtenidos al ejecutar el script.
