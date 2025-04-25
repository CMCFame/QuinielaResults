# ===========================================================
# app.py
# ===========================================================
import streamlit as st
from parser import parse_heat_map
from simulator import simulate

st.set_page_config(page_title="Progol Monte‑Carlo Analyzer", page_icon="⚽")

st.title("Progol 2278 · Calculadora de Probabilidades")

st.markdown(
    "Pegue debajo **solo** las 21 líneas de partidos con 20 códigos (L/E/V o dobles) cada una.\n"
    "Si incluye la cabecera «Partido …» será ignorada automáticamente."
)

input_text = st.text_area("Heat‑map de 20 quinielas", height=380)
iterations = st.slider("Iteraciones Monte‑Carlo", 1000, 50000, 10000, 1000)

if st.button("Calcular") and input_text.strip():
    try:
        grid = parse_heat_map(input_text)
        with st.spinner("Simulando …"):
            p_reg, p_rev = simulate(grid, iterations)
        st.metric("≥ 11 aciertos (Regular)", f"{p_reg*100:.2f}%")
        st.metric("≥ 6 aciertos (Revancha)", f"{p_rev*100:.2f}%")
        st.caption(f"{iterations:,} simulaciones Monte‑Carlo · probabilidades implícitas 2025")
    except Exception as exc:
        st.error(f"Error al procesar la malla: {exc}")
