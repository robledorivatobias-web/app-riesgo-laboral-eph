"""
Labor Transition Model — carga el logit entrenado y predice
la probabilidad de caída del empleo formal + factores de riesgo.
"""
import joblib
import numpy as np
import pandas as pd


class LaborModel:
    def __init__(self, ruta_modelo="modelo_app.joblib"):
        self.modelo = joblib.load(ruta_modelo)
        self.params = self.modelo.params

    def predict(self, mujer, edad, nivel_ed, region, tam_empresa, antiguedad):
        fila = pd.DataFrame([{
            "mujer": mujer,
            "CH06_t1": edad,
            "NIVEL_ED_t1": nivel_ed,
            "REGION_t1": region,
            "tam_empresa": tam_empresa,
            "antiguedad": antiguedad,
        }])

        prob = float(self.modelo.predict(fila).iloc[0])

        if prob < 0.05:
            categoria, color = "Bajo", "verde"
        elif prob < 0.10:
            categoria, color = "Moderado", "amarillo"
        elif prob < 0.18:
            categoria, color = "Alto", "naranja"
        else:
            categoria, color = "Muy alto", "rojo"

        return {
            "probabilidad": round(prob * 100, 1),
            "categoria": categoria,
            "color": color,
            "factores": self._factores(mujer, nivel_ed, region, tam_empresa, antiguedad),
        }

    def _factores(self, mujer, nivel_ed, region, tam_empresa, antiguedad):
        contribuciones = []
        mapa = {
            f"C(NIVEL_ED_t1)[T.{nivel_ed}]": "Nivel educativo",
            f"C(REGION_t1)[T.{region}]": "Region",
            f"C(tam_empresa)[T.{tam_empresa}]": "Tamano de empresa",
            f"C(antiguedad)[T.{antiguedad}]": "Antiguedad",
        }
        for clave, etiqueta in mapa.items():
            if clave in self.params:
                contribuciones.append((etiqueta, self.params[clave]))
        if mujer == 1:
            contribuciones.append(("Genero (mujer)", self.params.get("mujer", 0)))

        contribuciones.sort(key=lambda x: x[1], reverse=True)
        return contribuciones
