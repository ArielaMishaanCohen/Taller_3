"""Verifica que los entry points quedaron registrados al instalar el paquete."""

import shutil
import subprocess
import tempfile
import unittest
from importlib.metadata import entry_points
from pathlib import Path

from titanic_pipeline.pipeline import GRUPO_MODELOS, cargar_estimador

COMANDOS = [
    "titanic-pipeline",
    "titanic-extract",
    "titanic-clean",
    "titanic-train",
    "titanic-predict",
    "titanic-models",
]


class EntryPointsTest(unittest.TestCase):
    def test_console_scripts_registrados(self):
        nombres = {ep.name for ep in entry_points(group="console_scripts")}
        for comando in COMANDOS:
            self.assertIn(comando, nombres)
            self.assertIsNotNone(shutil.which(comando), f"{comando} no está en el PATH")

    def test_grupo_de_modelos_se_descubre(self):
        nombres = {ep.name for ep in entry_points(group=GRUPO_MODELOS)}
        self.assertTrue({"random_forest", "logistic_regression", "gradient_boosting"} <= nombres)
        self.assertTrue(hasattr(cargar_estimador("logistic_regression"), "fit"))

    def test_modelo_no_registrado_falla(self):
        with self.assertRaises(ValueError):
            cargar_estimador("modelo_inexistente")

    def test_comandos_de_punta_a_punta(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cmd in (["titanic-extract"], ["titanic-clean"],
                        ["titanic-train", "--modelo", "logistic_regression"]):
                r = subprocess.run(cmd + ["--output-dir", tmp], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((Path(tmp) / "modelo_titanic.joblib").exists())


if __name__ == "__main__":
    unittest.main()
