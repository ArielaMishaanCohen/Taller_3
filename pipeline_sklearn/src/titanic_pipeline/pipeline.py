"""Etapas reproducibles del pipeline de Machine Learning para Titanic."""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
TARGET = "Survived"
NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
REQUIRED_COLUMNS = FEATURES + [TARGET]

# Grupo de entry points donde se registran los estimadores disponibles.
GRUPO_MODELOS = "titanic_pipeline.modelos"
MODELO_POR_DEFECTO = "random_forest"


def modelos_disponibles() -> dict:
    """Devuelve los entry points registrados en el grupo de modelos."""
    return {ep.name: ep for ep in entry_points(group=GRUPO_MODELOS)}


def cargar_estimador(nombre: str = MODELO_POR_DEFECTO):
    """Busca el estimador por nombre en los metadatos del paquete instalado."""
    disponibles = modelos_disponibles()
    if nombre not in disponibles:
        raise ValueError(
            f"Modelo '{nombre}' no registrado. Disponibles: {sorted(disponibles)}"
        )
    fabrica = disponibles[nombre].load()  # importa module:attr bajo demanda
    return fabrica()


def resolver_ruta(ruta_csv: str | Path | None = None) -> Path:
    """Resuelve la ubicación del dataset sin depender del directorio actual."""
    base = Path(__file__).resolve().parent
    if ruta_csv is None:
        candidates = [base / "data" / "titanic.csv", base.parent / "titanic.csv"]
    else:
        ruta = Path(ruta_csv)
        if ruta.is_absolute():
            candidates = [ruta]
        else:
            candidates = [ruta, base / ruta, base / "data" / ruta, base.parent / ruta]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No se encontró el dataset. Rutas evaluadas: {candidates}")


def extraer_datos(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    """Carga el CSV y valida el contrato mínimo de columnas y registros."""
    ruta = resolver_ruta(ruta_csv)
    datos = pd.read_csv(ruta)
    faltantes = sorted(set(REQUIRED_COLUMNS) - set(datos.columns))
    if faltantes:
        raise ValueError(f"El dataset no contiene las columnas requeridas: {faltantes}")
    if datos.empty:
        raise ValueError("El dataset está vacío.")
    return datos


def limpiar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """Selecciona variables, normaliza texto y elimina targets inválidos."""
    limpios = datos[REQUIRED_COLUMNS].copy()
    limpios = limpios.dropna(subset=[TARGET])
    limpios[TARGET] = limpios[TARGET].astype(int)

    for columna in ["Sex", "Embarked"]:
        limpios[columna] = limpios[columna].map(
            lambda valor: valor.strip().lower() if isinstance(valor, str) else None
        )

    if not set(limpios[TARGET].unique()).issubset({0, 1}):
        raise ValueError("La variable objetivo Survived solo puede contener 0 y 1.")
    return limpios


def construir_modelo(nombre_modelo: str = MODELO_POR_DEFECTO) -> Pipeline:
    """Construye el preprocesamiento y el estimador sin ajustar datos aún."""
    pipeline_numerico = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline_categorico = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocesador = ColumnTransformer(
        [
            ("num", pipeline_numerico, NUMERIC_FEATURES),
            ("cat", pipeline_categorico, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocesamiento", preprocesador),
            ("modelo", cargar_estimador(nombre_modelo)),
        ]
    )


def entrenar_modelo(
    datos_limpios: pd.DataFrame,
    nombre_modelo: str = MODELO_POR_DEFECTO,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Divide, entrena y evalúa el modelo de manera determinista."""
    X = datos_limpios[FEATURES]
    y = datos_limpios[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    modelo = construir_modelo(nombre_modelo)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    resultados = X_test.copy()
    resultados["Survived_Real"] = y_test.to_numpy()
    resultados["Survived_Pred"] = predicciones
    metricas = {
        "modelo": nombre_modelo,
        "accuracy": round(float(accuracy_score(y_test, predicciones)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classification_report": classification_report(
            y_test, predicciones, output_dict=True, zero_division=0
        ),
    }
    return modelo, resultados, metricas


def guardar_artefactos(
    modelo: Pipeline,
    resultados: pd.DataFrame,
    metricas: dict,
    directorio_salida: str | Path,
) -> dict[str, Path]:
    """Guarda modelo, predicciones y métricas para auditoría o despliegue."""
    salida = Path(directorio_salida)
    salida.mkdir(parents=True, exist_ok=True)
    rutas = {
        "modelo": salida / "modelo_titanic.joblib",
        "predicciones": salida / "predicciones_titanic.csv",
        "metricas": salida / "metricas.json",
    }
    joblib.dump(modelo, rutas["modelo"])
    resultados.to_csv(rutas["predicciones"], index=False)
    rutas["metricas"].write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rutas


def ejecutar_pipeline(
    ruta_csv: str | Path | None = None,
    directorio_salida: str | Path | None = None,
    nombre_modelo: str = MODELO_POR_DEFECTO,
) -> Pipeline:
    """Ejecuta extracción, limpieza, entrenamiento y persistencia."""
    salida = directorio_salida or Path(__file__).resolve().parent / "artifacts"
    datos = extraer_datos(ruta_csv)
    limpios = limpiar_datos(datos)
    modelo, resultados, metricas = entrenar_modelo(limpios, nombre_modelo)
    rutas = guardar_artefactos(modelo, resultados, metricas, salida)
    print(f"Pipeline ejecutado correctamente. Accuracy: {metricas['accuracy']:.4f}")
    for nombre, ruta in rutas.items():
        print(f"{nombre.capitalize()}: {ruta}")
    return modelo
