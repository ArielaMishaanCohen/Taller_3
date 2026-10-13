"""Fábricas de estimadores registradas como entry points.

Cada función se declara en pyproject.toml dentro del grupo
"titanic_pipeline.modelos". Otro paquete instalado puede registrar sus
propios modelos en el mismo grupo sin modificar este código.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def random_forest():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)


def regresion_logistica():
    return LogisticRegression(max_iter=1000, random_state=42)


def gradient_boosting():
    return GradientBoostingClassifier(random_state=42)
