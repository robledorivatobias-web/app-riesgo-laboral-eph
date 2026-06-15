"""
Simulador de Riesgo de Perdida del Empleo Formal — Streamlit
Modelo logit sobre microdatos de panel de la EPH (INDEC).
"""
import streamlit as st
import plotly.graph_objects as go
from labor_model import LaborModel

st.set_page_config(page_title="Riesgo Laboral EPH", page_icon=":bar_chart:", layout="wide")

if "model" not in st.session_state:
    st.session_state.model = LaborModel("modelo_app.joblib")
model = st.session_state.model

NIVEL_ED = {"Primaria incompleta": 1, "Primaria completa": 2, "Secundaria incompleta": 3,
            "Secundaria completa": 4, "Universitaria incompleta": 5, "Universitaria completa": 6}
REGION   = {"GBA": 1, "Noroeste (NOA)": 40, "Noreste (NEA)": 41,
            "Cuyo": 42, "Pampeana": 43, "Patagonia": 44}
TAM      = {"Chica (<=25)": "chica", "Mediana (26-100)": "mediana", "Grande (>100)": "grande"}
ANTIG    = {"Menos de 1 ano": "a_menos_1", "Entre 1 y 5 anos": "b_1_a_5", "Mas de 5 anos": "c_mas_5"}
COLORES  = {"verde": "#00c853", "amarillo": "#ffd600", "naranja": "#ff9100", "rojo": "#d50000"}

with st.sidebar:
    st.markdown("## Perfil del trabajador")
    sexo_lbl   = st.radio("Sexo", ["Varon", "Mujer"])
    edad       = st.slider("Edad", 18, 59, 35)
    nivel_lbl  = st.selectbox("Nivel educativo", list(NIVEL_ED))
    region_lbl = st.selectbox("Region", list(REGION))
    tam_lbl    = st.selectbox("Tamano de empresa", list(TAM))
    antig_lbl  = st.selectbox("Antiguedad en el empleo", list(ANTIG))
    evaluar    = st.button("Evaluar riesgo", use_container_width=True)

st.title("Riesgo de perdida del empleo formal")
st.caption("Probabilidad de transicion a desempleo o informalidad en el proximo trimestre - "
           "modelo logit sobre panel EPH (INDEC)")

if evaluar:
    r = model.predict(
        mujer=1 if sexo_lbl == "Mujer" else 0,
        edad=edad,
        nivel_ed=NIVEL_ED[nivel_lbl],
        region=REGION[region_lbl],
        tam_empresa=TAM[tam_lbl],
        antiguedad=ANTIG[antig_lbl],
    )
    color = COLORES[r["color"]]
    col1, col2 = st.columns([1, 1])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r["probabilidad"],
            number={"suffix": "%", "font": {"size": 48, "color": color}},
            gauge={
                "axis": {"range": [0, 30]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 5],   "color": "#1b2a1b"},
                    {"range": [5, 10],  "color": "#2a2a1b"},
                    {"range": [10, 18], "color": "#2a221b"},
                    {"range": [18, 30], "color": "#2a1b1b"},
                ],
            },
            title={"text": "Riesgo " + r["categoria"]},
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Factores que mas influyen")
        st.caption("Ordenados de mayor a menor empuje al riesgo")
        for etiqueta, coef in r["factores"]:
            signo = "(+)" if coef > 0 else "(-)"
            efecto = "sube el riesgo" if coef > 0 else "baja el riesgo"
            st.write(signo + " **" + etiqueta + "** - " + efecto)
else:
    st.info("Completa el perfil en el panel izquierdo y hace clic en Evaluar riesgo.")
