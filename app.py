# ==========================================================
# app.py  (text input + CSV uploader)
# ==========================================================
import streamlit as st, pandas as pd
from parser import parse_heat_map
from simulator import simulate

st.set_page_config(page_title="Progol Monte‑Carlo Analyzer", page_icon="⚽")
st.title("Progol 2278 · Calculadora de Probabilidades")

st.markdown(
"""
### Opciones de carga
* **CSV**: suba un archivo con columnas `match,Q1,…,Q20`.
* **Pegado de texto**: 21 filas, 20 códigos L/E/V (o dobles `L/E`).
""", unsafe_allow_html=True)

# ---------- carga CSV ----------
file = st.file_uploader("Cargar CSV", type="csv")
text = st.text_area("…o pegue la malla aquí", height=320)
iterations = st.slider("Iteraciones Monte‑Carlo", 1_000, 50_000, 10_000, 1_000)

if st.button("Calcular"):
    try:
        if file:
            df = pd.read_csv(file)
            expected_cols = ["match"] + [f"Q{i+1}" for i in range(20)]
            if list(df.columns) != expected_cols:
                raise ValueError("CSV debe tener columnas 'match,Q1,…,Q20' en ese orden.")
            df = df.set_index("match")
            if len(df) != 21:
                raise ValueError(f"CSV debe tener 21 filas; tiene {len(df)}.")
        else:
            df = parse_heat_map(text)
        p_reg, p_rev = simulate(df, iterations)
        st.metric("≥ 11 aciertos (Regular)",  f"{p_reg*100:.2f}%")
        st.metric("≥ 6 aciertos (Revancha)", f"{p_rev*100:.2f}%")
    except Exception as ex:
        st.error(f"Error: {ex}")
