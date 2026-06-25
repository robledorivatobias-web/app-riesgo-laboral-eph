"""
Predictor de Riesgo de Perdida del Empleo Formal - Streamlit
Modelo logit sobre microdatos de panel de la EPH (INDEC).
"""
import streamlit as st
import plotly.graph_objects as go
from labor_model import LaborModel

st.set_page_config(page_title="Riesgo Laboral EPH", page_icon=":bar_chart:", layout="wide")

AZUL  = "#1b3a5b"
AZUL2 = "#2e5c8a"
VERDE = "#2e7d5b"
ROJO  = "#b5384d"
GRIS  = "#8a94a0"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f7f8fa; color: #1a2330; }
    .stApp { background-color: #f7f8fa; }
    h1 { color: #1b3a5b; font-weight: 700; letter-spacing: -0.5px; }
    h2, h3, h4 { color: #2a3645; font-weight: 600; }
    .stButton>button { background-color: #1b3a5b; color: #ffffff; font-weight: 600;
        border: none; border-radius: 6px; padding: 0.6rem 2rem; width: 100%; }
    .stButton>button:hover { background-color: #2e5c8a; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e6ea; }
    .metric-card { background: #ffffff; border: 1px solid #e2e6ea; border-radius: 8px;
        padding: 1.2rem 1.5rem; margin: 0.3rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .metric-label { font-size: 0.7rem; letter-spacing: 1px; color: #8a94a0; text-transform: uppercase; font-weight: 600; }
    .metric-value { font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin-top: 0.3rem; }
    .factor-row { padding: 8px 0; border-bottom: 1px solid #e8ebee; }
    .factor-pp { font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

NIVEL_ED = {"Primaria incompleta": 1, "Primaria completa": 2, "Secundaria incompleta": 3,
            "Secundaria completa": 4, "Universitaria incompleta": 5, "Universitaria completa": 6}
REGION   = {"GBA": 1, "Noroeste (NOA)": 40, "Noreste (NEA)": 41,
            "Cuyo": 42, "Pampeana": 43, "Patagonia": 44}
TAM      = {"Chica (<=25)": "chica", "Mediana (26-100)": "mediana", "Grande (>100)": "grande"}
ANTIG    = {"Menos de 1 ano": "a_menos_1", "Entre 1 y 5 anos": "b_1_a_5", "Mas de 5 anos": "c_mas_5"}

PERIODOS = ["2022T1","2022T2","2022T3","2022T4","2023T1","2023T2","2023T3","2023T4",
            "2024T1","2024T2","2024T3","2024T4","2025T1","2025T2","2025T3"]
TASA = [7.53,7.29,6.41,6.99,7.02,6.87,6.84,9.69,7.57,7.87,7.54,7.15,8.05,7.99,7.57]

# Curvas ROC (test set, calculadas en Colab)
FPR = [0.0,0.012,0.027,0.046,0.067,0.094,0.126,0.154,0.185,0.22,0.269,0.314,0.371,0.434,0.504,0.57,0.622,0.672,0.721,0.774,0.821,0.865,0.909,0.955,1.0]
TPR = [0.0,0.093,0.18,0.252,0.325,0.399,0.465,0.525,0.584,0.646,0.694,0.747,0.794,0.837,0.86,0.889,0.914,0.928,0.943,0.958,0.965,0.977,0.986,0.992,1.0]
FPR_RF = [0.0,0.012,0.029,0.047,0.072,0.09,0.12,0.152,0.189,0.224,0.267,0.314,0.377,0.453,0.511,0.573,0.63,0.676,0.72,0.773,0.822,0.868,0.918,0.961,1.0]
TPR_RF = [0.0,0.101,0.172,0.24,0.32,0.387,0.45,0.519,0.579,0.64,0.69,0.744,0.787,0.829,0.861,0.891,0.917,0.933,0.947,0.963,0.973,0.981,0.985,0.994,1.0]

# AME por region (pp) y significancia, para las barras
REGION_AME = [("GBA (ref.)", 0.0, True), ("NOA", -0.79, True), ("NEA", -0.18, False),
              ("Cuyo", -0.37, False), ("Pampeana", -0.83, True), ("Patagonia", -1.90, True)]

# Tabla de AME completa: (variable, categoria, efecto_pp, significativo)
TABLA_AME = [
    ("Genero", "Mujer (vs varon)", 1.80, True),
    ("Edad", "Por cada ano", -0.23, True),
    ("Educacion", "Primaria completa", -1.35, False),
    ("Educacion", "Secundaria incompleta", -2.41, True),
    ("Educacion", "Secundaria completa", -4.26, True),
    ("Educacion", "Universitaria incompleta", -5.57, True),
    ("Educacion", "Universitaria completa", -8.02, True),
    ("Educacion", "Sin instruccion", 1.43, False),
    ("Antiguedad", "1 a 5 anos", -4.02, True),
    ("Antiguedad", "Mas de 5 anos", -9.22, True),
    ("Tamano empresa", "Mediana", -3.53, True),
    ("Tamano empresa", "Grande", -5.62, True),
    ("Region", "NOA", -0.79, True),
    ("Region", "NEA", -0.18, False),
    ("Region", "Cuyo", -0.37, False),
    ("Region", "Pampeana", -0.83, True),
    ("Region", "Patagonia", -1.90, True),
]

def color_riesgo(cat):
    return {"Bajo": VERDE, "Moderado": "#c9a227", "Alto": "#cc6a1f", "Muy alto": ROJO}.get(cat, AZUL)

if "model" not in st.session_state:
    st.session_state.model = LaborModel("modelo_app.joblib")
model = st.session_state.model

def evaluar_perfil(sexo, edad, nivel, region, tam, antig):
    return model.predict(mujer=1 if sexo == "Mujer" else 0, edad=edad,
        nivel_ed=NIVEL_ED[nivel], region=REGION[region], tam_empresa=TAM[tam], antiguedad=ANTIG[antig])

def controles(prefijo):
    sexo  = st.radio("Sexo", ["Varon", "Mujer"], key=prefijo+"_sexo")
    edad  = st.slider("Edad", 18, 59, 35, key=prefijo+"_edad")
    nivel = st.selectbox("Nivel educativo", list(NIVEL_ED), key=prefijo+"_nivel")
    reg   = st.selectbox("Region", list(REGION), key=prefijo+"_region")
    tam   = st.selectbox("Tamano de empresa", list(TAM), key=prefijo+"_tam")
    antig = st.selectbox("Antiguedad", list(ANTIG), key=prefijo+"_antig")
    return sexo, edad, nivel, reg, tam, antig

with st.sidebar:
    st.markdown("## Perfil del trabajador")
    st.markdown("---")
    sexo_lbl   = st.radio("Sexo", ["Varon", "Mujer"])
    edad       = st.slider("Edad", 18, 59, 35)
    nivel_lbl  = st.selectbox("Nivel educativo", list(NIVEL_ED))
    region_lbl = st.selectbox("Region", list(REGION))
    tam_lbl    = st.selectbox("Tamano de empresa", list(TAM))
    antig_lbl  = st.selectbox("Antiguedad en el empleo", list(ANTIG))
    st.markdown("---")
    evaluar    = st.button("EVALUAR RIESGO")

st.title("Riesgo de perdida del empleo formal")
st.caption("Probabilidad de transicion a desempleo o informalidad en el proximo trimestre  |  "
           "modelo logit sobre panel rotativo EPH (INDEC)")

tab1, tab2, tab3, tab4 = st.tabs(
    ["  Evaluacion  ", "  Comparador  ", "  Contexto macro  ", "  Metodologia  "])

# ---------- TAB 1: EVALUACION ----------
with tab1:
    if evaluar:
        r = evaluar_perfil(sexo_lbl, edad, nivel_lbl, region_lbl, tam_lbl, antig_lbl)
        color = color_riesgo(r["categoria"]); de_cada_100 = round(r["probabilidad"])
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Prob. de caida</div>
                <div class="metric-value" style="color:{color};">{r['probabilidad']}%</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Categoria</div>
                <div class="metric-value" style="color:{color}; font-size:1.8rem;">{r['categoria']}</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">De cada 100 perfiles</div>
                <div class="metric-value" style="color:{color};">{de_cada_100}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=r["probabilidad"],
                number={"suffix": "%", "font": {"size": 44, "color": color}},
                gauge={"axis": {"range": [0, 30], "tickfont": {"color": "#8a94a0", "size": 11}},
                       "bar": {"color": color, "thickness": 0.3}, "bgcolor": "#ffffff", "bordercolor": "#e2e6ea",
                       "steps": [{"range": [0, 5], "color": "#e8f3ee"}, {"range": [5, 10], "color": "#f5f1e0"},
                                 {"range": [10, 18], "color": "#f7ebe0"}, {"range": [18, 30], "color": "#f7e5e8"}]}))
            fig.update_layout(paper_bgcolor="#f7f8fa", font_color="#2a3645", height=300, margin=dict(t=30, b=10, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Factores del perfil")
            st.caption("Efecto en puntos porcentuales (pp), respecto de la categoria de referencia. "
                       "En gris: no significativos (p > 0.05).")
            for etiqueta, pp, sig in r["factores"]:
                flecha, col_efecto = ("^", ROJO) if pp > 0 else ("v", VERDE)
                if not sig: col_efecto = GRIS
                signo = "+" if pp > 0 else ""; marca = "" if sig else "  (n.s.)"
                st.markdown(f"<div class='factor-row'><span class='factor-pp' style='color:{col_efecto};'>"
                            f"{flecha} {signo}{pp:.1f} pp</span>&nbsp;&nbsp;{etiqueta}{marca}</div>", unsafe_allow_html=True)
    else:
        st.info("Completa el perfil en el panel izquierdo y hace clic en EVALUAR RIESGO.")

# ---------- TAB 2: COMPARADOR ----------
with tab2:
    st.markdown("#### Comparar dos perfiles")
    st.caption("Defini dos trabajadores y compara su riesgo lado a lado.")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Perfil A**"); a = controles("a")
    with cB:
        st.markdown("**Perfil B**"); b = controles("b")
    if st.button("COMPARAR PERFILES"):
        ra = evaluar_perfil(*a); rb = evaluar_perfil(*b)
        diff = round(ra["probabilidad"] - rb["probabilidad"], 1)
        colA, colB = color_riesgo(ra["categoria"]), color_riesgo(rb["categoria"])
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Perfil A</div>
                <div class="metric-value" style="color:{colA};">{ra['probabilidad']}%</div>
                <div style="color:{colA}; font-weight:600;">{ra['categoria']}</div></div>""", unsafe_allow_html=True)
        with r2:
            signo = "+" if diff > 0 else ""; col_d = ROJO if diff > 0 else VERDE
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Diferencia A - B</div>
                <div class="metric-value" style="color:{col_d};">{signo}{diff}</div>
                <div style="color:#8a94a0; font-size:0.8rem;">puntos porcentuales</div></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Perfil B</div>
                <div class="metric-value" style="color:{colB};">{rb['probabilidad']}%</div>
                <div style="color:{colB}; font-weight:600;">{rb['categoria']}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=["Perfil A", "Perfil B"], y=[ra["probabilidad"], rb["probabilidad"]],
                               marker_color=[colA, colB], text=[f"{ra['probabilidad']}%", f"{rb['probabilidad']}%"],
                               textposition="outside", width=0.5))
        fig_c.update_layout(paper_bgcolor="#f7f8fa", plot_bgcolor="#ffffff", font_color="#2a3645", height=320,
            margin=dict(t=30, b=20, l=50, r=30),
            yaxis={"title": "Prob. de caida (%)", "gridcolor": "#eaedf0", "range": [0, max(ra["probabilidad"], rb["probabilidad"])*1.25]})
        st.plotly_chart(fig_c, use_container_width=True)

# ---------- TAB 3: CONTEXTO MACRO ----------
with tab3:
    st.markdown("#### El quiebre de 2023T4")
    st.caption("Tasa de caida del empleo formal por trimestre. El salto de 2023T4 coincide con la "
               "devaluacion de diciembre de 2023.")
    idx_q = PERIODOS.index("2023T4")
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=PERIODOS, y=TASA, mode="lines+markers", line={"color": AZUL, "width": 2.5},
        marker={"color": [AZUL]*idx_q + [ROJO] + [AZUL]*(len(PERIODOS)-idx_q-1),
                "size": [7]*idx_q + [14] + [7]*(len(PERIODOS)-idx_q-1)}))
    fig_s.add_annotation(x="2023T4", y=9.69, text="Devaluacion<br>dic-2023", showarrow=True, arrowhead=2,
                         arrowcolor=ROJO, font={"color": ROJO, "size": 12}, ax=0, ay=-45)
    fig_s.update_layout(paper_bgcolor="#f7f8fa", plot_bgcolor="#ffffff", font_color="#2a3645", height=380,
        margin=dict(t=20, b=50, l=55, r=30), xaxis={"tickangle": -45, "gridcolor": "#eaedf0"},
        yaxis={"title": "Tasa de caida (%)", "gridcolor": "#eaedf0", "range": [5, 11]})
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("#### Riesgo por region")
    st.caption("Efecto de cada region sobre la probabilidad de caida (pp, vs GBA). "
               "En gris claro: no significativos.")
    etiquetas_r = [x[0] for x in REGION_AME]
    valores_r   = [x[1] for x in REGION_AME]
    colores_r   = [AZUL if x[2] else "#c2cad3" for x in REGION_AME]
    fig_r = go.Figure(go.Bar(x=valores_r, y=etiquetas_r, orientation="h", marker_color=colores_r,
                             text=[f"{v:+.2f}" for v in valores_r], textposition="outside"))
    fig_r.update_layout(paper_bgcolor="#f7f8fa", plot_bgcolor="#ffffff", font_color="#2a3645", height=300,
        margin=dict(t=20, b=30, l=90, r=40), xaxis={"title": "Efecto (pp)", "gridcolor": "#eaedf0"},
        yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_r, use_container_width=True)

# ---------- TAB 4: METODOLOGIA ----------
with tab4:
    st.markdown("#### Como funciona")
    st.markdown("""
**Datos.** Panel rotativo de la EPH-INDEC, esquema 2-2-2: cada persona se sigue dos trimestres
consecutivos. Periodo 2022-2025, ~281.000 transiciones.

**Poblacion.** Asalariados formales de 18 a 59 anos (formal = con descuento jubilatorio).

**Variable predicha.** "Caida" = pasar a desempleo, inactividad (no jubilacion) o informalidad
en el trimestre siguiente. Tasa promedio: ~7,5%.

**Modelo.** Regresion logistica con efectos fijos de trimestre, comparada contra random forest.
""")

    st.markdown("#### Capacidad predictiva: curva ROC")
    st.caption("Ambos modelos empatan (AUC ~0,77). Se elige el logit por ser interpretable.")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=FPR, y=TPR, mode="lines", line={"color": AZUL, "width": 2.5},
                                 name="Logit (AUC=0.778)"))
    fig_roc.add_trace(go.Scatter(x=FPR_RF, y=TPR_RF, mode="lines", line={"color": ROJO, "width": 2.5},
                                 name="Random Forest (AUC=0.774)"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line={"color": GRIS, "width": 1.5, "dash": "dash"}, name="Azar (AUC=0.500)"))
    fig_roc.update_layout(paper_bgcolor="#f7f8fa", plot_bgcolor="#ffffff", font_color="#2a3645", height=420,
        margin=dict(t=20, b=50, l=55, r=30), legend={"x": 0.55, "y": 0.1},
        xaxis={"title": "Tasa de falsos positivos", "gridcolor": "#eaedf0", "range": [0, 1]},
        yaxis={"title": "Tasa de verdaderos positivos", "gridcolor": "#eaedf0", "range": [0, 1]})
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("#### Efectos marginales completos (AME)")
    st.caption("Cambio en la probabilidad de caida (pp) por cada caracteristica, respecto de su "
               "categoria de referencia. (n.s.) = no significativo (p > 0.05).")
    filas = ""
    for var, cat, pp, sig in TABLA_AME:
        col = ROJO if pp > 0 else VERDE
        if not sig: col = GRIS
        ns = "" if sig else " (n.s.)"
        signo = "+" if pp > 0 else ""
        filas += (f"<tr><td style='padding:6px 12px; border-bottom:1px solid #e8ebee;'>{var}</td>"
                  f"<td style='padding:6px 12px; border-bottom:1px solid #e8ebee;'>{cat}</td>"
                  f"<td style='padding:6px 12px; border-bottom:1px solid #e8ebee; text-align:right; "
                  f"font-weight:600; color:{col};'>{signo}{pp:.2f}{ns}</td></tr>")
    st.markdown(f"""<table style="width:100%; border-collapse:collapse; background:#ffffff; border:1px solid #e2e6ea; border-radius:8px;">
        <tr style="background:#f0f2f5;"><th style="padding:8px 12px; text-align:left;">Variable</th>
        <th style="padding:8px 12px; text-align:left;">Categoria</th>
        <th style="padding:8px 12px; text-align:right;">Efecto (pp)</th></tr>{filas}</table>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
**Limitaciones.** El panel pierde gente entre trimestres de forma no aleatoria. Se excluyen
cuentapropistas (formalidad no observable). La informalidad medida es un flujo trimestral,
distinto del stock de informalidad de la economia (~36%).
""")

st.markdown("---")
st.markdown("""<p style="text-align:center; color:#a0a8b0; font-size:0.75rem;">
    Predictor de transiciones laborales  |  EPH-INDEC 2022-2025  |  modelo logit con efectos fijos de periodo
</p>""", unsafe_allow_html=True)
