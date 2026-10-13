PYTHON ?= python
OUTPUT_DIR ?= artifacts
MODELO ?= random_forest
DATA ?=
DATA_ARG = $(if $(DATA),--data $(DATA),)

.PHONY: install install-tracking check test models extract clean train predict pipeline dist mlflow-notebook mlflow-ui

# Instala el paquete en modo editable: registra los entry points en el ambiente.
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

check:
	$(PYTHON) -m compileall -q src tests

test:
	$(PYTHON) -m unittest discover -s tests -v

models:
	titanic-models

# Cada etapa llama al comando creado por su entry point (ya no a "python main.py").
extract:
	titanic-extract $(DATA_ARG) --output-dir $(OUTPUT_DIR)

clean: extract
	titanic-clean $(DATA_ARG) --output-dir $(OUTPUT_DIR)

train: clean
	titanic-train $(DATA_ARG) --output-dir $(OUTPUT_DIR) --modelo $(MODELO)

predict: train
	titanic-predict --input src/titanic_pipeline/data/titanic.csv --model $(OUTPUT_DIR)/modelo_titanic.joblib --output $(OUTPUT_DIR)/predicciones_batch.csv

pipeline: check test predict

dist:
	$(PYTHON) -m pip install build
	$(PYTHON) -m build

# --- Taller 3: MLflow ---
install-tracking:
	$(PYTHON) -m pip install -e ".[tracking]"

# Ejecuta el notebook completo y deja los runs en notebooks/mlflow.db
mlflow-notebook:
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace taller3_mlflow_pipeline.ipynb

mlflow-ui:
	cd notebooks && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
