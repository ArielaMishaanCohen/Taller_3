"""Funciones expuestas como console_scripts (un comando por etapa).

Según la especificación, el instalador llama a cada función sin argumentos y
usa su valor de retorno como código de salida (None equivale a 0). Por eso
cada función lee sus propios argumentos con argparse y devuelve un int.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from . import __version__
from .pipeline import (
    FEATURES,
    MODELO_POR_DEFECTO,
    ejecutar_pipeline,
    entrenar_modelo,
    extraer_datos,
    guardar_artefactos,
    limpiar_datos,
    modelos_disponibles,
)

DEFAULT_OUTPUT = Path("artifacts")


def _parser(descripcion: str, modelo: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument("--data", default=None, help="CSV de entrada (por defecto, el incluido en el paquete)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directorio de artefactos")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    if modelo:
        parser.add_argument(
            "--modelo",
            default=MODELO_POR_DEFECTO,
            choices=sorted(modelos_disponibles()),
            help="Estimador registrado en el grupo titanic_pipeline.modelos",
        )
    return parser


def _salida(args: argparse.Namespace) -> Path:
    salida = Path(args.output_dir)
    salida.mkdir(parents=True, exist_ok=True)
    return salida


def extraer(argv: list[str] | None = None) -> int:
    """titanic-extract: valida el CSV y guarda datos_extraidos.csv."""
    args = _parser("Etapa 1: extracción y validación").parse_args(argv)
    destino = _salida(args) / "datos_extraidos.csv"
    datos = extraer_datos(args.data)
    datos.to_csv(destino, index=False)
    print(f"Extracción validada: {len(datos)} filas -> {destino}")
    return 0


def limpiar(argv: list[str] | None = None) -> int:
    """titanic-clean: limpia los datos extraídos y guarda datos_limpios.csv."""
    args = _parser("Etapa 2: limpieza").parse_args(argv)
    salida = _salida(args)
    origen = salida / "datos_extraidos.csv"
    datos = pd.read_csv(origen) if origen.exists() else extraer_datos(args.data)
    limpios = limpiar_datos(datos)
    limpios.to_csv(salida / "datos_limpios.csv", index=False)
    print(f"Limpieza completada: {len(limpios)} filas -> {salida / 'datos_limpios.csv'}")
    return 0


def entrenar(argv: list[str] | None = None) -> int:
    """titanic-train: entrena, evalúa y guarda modelo, predicciones y métricas."""
    args = _parser("Etapa 3: entrenamiento y evaluación", modelo=True).parse_args(argv)
    salida = _salida(args)
    origen = salida / "datos_limpios.csv"
    datos = pd.read_csv(origen) if origen.exists() else extraer_datos(args.data)
    modelo, resultados, metricas = entrenar_modelo(limpiar_datos(datos), args.modelo)
    guardar_artefactos(modelo, resultados, metricas, salida)
    print(f"Entrenamiento completado ({args.modelo}). Accuracy: {metricas['accuracy']:.4f}")
    return 0


def predecir(argv: list[str] | None = None) -> int:
    """titanic-predict: carga el modelo guardado y genera predicciones (inferencia batch)."""
    parser = argparse.ArgumentParser(description="Inferencia batch con el modelo entrenado")
    parser.add_argument("--input", required=True, help="CSV con las columnas de entrada")
    parser.add_argument("--model", default=str(DEFAULT_OUTPUT / "modelo_titanic.joblib"))
    parser.add_argument("--output", default="predicciones.csv")
    args = parser.parse_args(argv)

    if not Path(args.model).exists():
        print(f"Error: no existe el modelo {args.model}. Ejecuta titanic-train primero.", file=sys.stderr)
        return 1
    modelo = joblib.load(args.model)
    datos = pd.read_csv(args.input)
    faltantes = sorted(set(FEATURES) - set(datos.columns))
    if faltantes:
        print(f"Error: faltan columnas {faltantes}", file=sys.stderr)
        return 2
    # Se reutiliza la misma limpieza del entrenamiento; Survived es un marcador temporal.
    entrada = limpiar_datos(datos.assign(Survived=0))[FEATURES]
    datos["Survived_Pred"] = modelo.predict(entrada)
    datos.to_csv(args.output, index=False)
    print(f"Predicciones generadas: {len(entrada)} filas -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """titanic-pipeline: ejecuta todas las etapas de principio a fin."""
    args = _parser("Pipeline completo: extracción, limpieza y entrenamiento", modelo=True).parse_args(argv)
    ejecutar_pipeline(args.data, _salida(args), args.modelo)
    metricas = json.loads((Path(args.output_dir) / "metricas.json").read_text(encoding="utf-8"))
    print(f"Modelo usado: {metricas['modelo']}")
    return 0


def listar_modelos(argv: list[str] | None = None) -> int:
    """titanic-models: muestra los modelos descubiertos vía entry points."""
    for nombre, ep in sorted(modelos_disponibles().items()):
        print(f"{nombre:<22} -> {ep.value}")
    return 0
