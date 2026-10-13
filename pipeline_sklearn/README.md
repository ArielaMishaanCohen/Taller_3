# Pipeline de scikit-learn — Titanic

Machine Learning Engineering (CC3105) · UVG · Grupo 6

El pipeline de Titanic (extracción → limpieza → entrenamiento → evaluación) se
empaquetó en `src/titanic_pipeline/` y se expone mediante entry points
declarados en `pyproject.toml`
([especificación de PyPA](https://packaging.python.org/en/latest/specifications/entry-points/)).

## Entry points declarados

| Grupo | Nombre | Referencia |
|---|---|---|
| `console_scripts` | `titanic-pipeline` | `titanic_pipeline.cli:main` |
| `console_scripts` | `titanic-extract` | `titanic_pipeline.cli:extraer` |
| `console_scripts` | `titanic-clean` | `titanic_pipeline.cli:limpiar` |
| `console_scripts` | `titanic-train` | `titanic_pipeline.cli:entrenar` |
| `console_scripts` | `titanic-predict` | `titanic_pipeline.cli:predecir` |
| `console_scripts` | `titanic-models` | `titanic_pipeline.cli:listar_modelos` |
| `titanic_pipeline.modelos` | `random_forest`, `logistic_regression`, `gradient_boosting` | `titanic_pipeline.modelos:*` |

El grupo `titanic_pipeline.modelos` funciona como sistema de plugins: el
pipeline busca el estimador con `importlib.metadata.entry_points()` y otro
paquete puede registrar un modelo nuevo en ese grupo sin modificar este código.

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
make install              # pip install -e . → crea los comandos en .venv/bin
make test
titanic-models            # lista los modelos descubiertos
titanic-pipeline --modelo gradient_boosting
make pipeline MODELO=logistic_regression

# Con uv
uv venv && uv pip install -e . && uv run titanic-train --modelo random_forest
```

## Taller 3 — Tracking con MLflow

El notebook [`notebooks/taller3_mlflow_pipeline.ipynb`](notebooks/taller3_mlflow_pipeline.ipynb)
conecta MLflow con este mismo pipeline, sin modificar el paquete:

| Run | Qué muestra |
|---|---|
| `autolog_random_forest` | `mlflow.sklearn.autolog()` sobre el `Pipeline`: 69 parámetros anidados, métricas de entrenamiento y modelo con firma |
| `comparacion_modelos` (+3 hijos) | Logging manual de los tres estimadores registrados como entry points: métricas de test, CV, matriz de confusión, `classification_report` y modelo |
| `gridsearch_random_forest` (+24 hijos) | `GridSearchCV` sobre preprocesamiento y modelo; autolog crea un run por combinación |
| `modelo_final` | Mejor modelo registrado en el Model Registry como `titanic-pipeline` y cargado de vuelta con `pyfunc` |

```bash
make install-tracking     # pip install -e ".[tracking]"
make mlflow-notebook      # ejecuta el notebook completo
make mlflow-ui            # http://127.0.0.1:5000
```

El backend es SQLite local (`notebooks/mlflow.db`) y los artefactos quedan en `notebooks/mlruns/`; ambos
están en `.gitignore`.
