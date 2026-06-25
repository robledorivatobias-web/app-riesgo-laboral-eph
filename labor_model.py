"""
Labor Transition Model - carga el logit entrenado, predice la probabilidad
de caida del empleo formal y reporta los efectos marginales (AME) de cada
caracteristica del perfil elegido.
"""
import joblib
import pandas as pd


AME = {
    "nivel_ed": {
        2: (-1.35, False, "Primaria completa"),
        3: (-2.41, True,  "Secundaria incompleta"),
        4: (-4.26, True,  "Secundaria completa"),
        5: (-5.57, True,  "Universitaria incompleta"),
        6: (-8.02, True,  "Universitaria completa"),
        7: (1.43,  False, "Sin instruccion"),
    },
    "antiguedad": {
        "b_1_a_5": (-4.02, True,  "Antiguedad 1 a 5 anos"),
        "c_mas_5": (-9.22, True,  "Antiguedad mas de 5 anos"),
        "ns_nr":   (0.93,  False, "Antiguedad sin dato"),
    },
    "tam_empresa": {
        "grande":  (-5.62, True, "Empresa grande"),
        "mediana": (-3.53, True, "Empresa mediana"),
        "ns_nr":   (-2.19, True, "Tamano sin dato"),
    },
    "region": {
        40: (-0.79, True,  "Region NOA"),
        41: (-0.18, False, "Region NEA"),
        42: (-0.37, False, "Region Cuyo"),
        43: (-0.83, True,  "Region Pampeana"),
        44: (-1.90, True,  "Region Patagonia"),
    },
}

AME_MUJER = (1.80, True, "Genero femenino")
AME_EDAD_POR_ANO = -0.23


class LaborModel:
    def __init__(self, ruta_modelo="modelo_app.joblib"):
        self.modelo = joblib.load(ruta_modelo)

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
            "factores": self._factores(mujer, edad, nivel_ed, region, tam_empresa, antiguedad),
        }

    def _factores(self, mujer, edad, nivel_ed, region, tam_empresa, antiguedad):
        items = []

        if mujer == 1:
            ef, sig, etq = AME_MUJER
            items.append((etq, ef, sig))

        edad_ref = 40
        ef_edad = round((edad - edad_ref) * AME_EDAD_POR_ANO, 2)
        if ef_edad != 0:
            items.append(("Edad (" + str(edad) + " anos, vs 40)", ef_edad, True))

        for valor, tabla in [
            (nivel_ed, AME["nivel_ed"]),
            (antiguedad, AME["antiguedad"]),
            (tam_empresa, AME["tam_empresa"]),
            (region, AME["region"]),
        ]:
            if valor in tabla:
                ef, sig, etq = tabla[valor]
                items.append((etq, ef, sig))

        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return items
