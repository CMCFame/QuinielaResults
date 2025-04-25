
# ===========================================================
# app.py
# ===========================================================
import streamlit as st
from parser import parse_heat_map
from simulator import simulate

st.set_page_config(page_title="Progol Monte‑Carlo Analyzer", page_icon="⚽")

st.title("Progol 2278 · Calculadora de Probabilidades")

st.markdown(
    "<small>Pegue abajo las 21 líneas (sin cabecera) con 20 códigos cada una — puede incluir dobles “L/E”.</small>",
    unsafe_allow_html=True,
)

input_text = st.text_area("Heat‑map de 20 quinielas", height=380)
iterations = st.slider("Iteraciones Monte‑Carlo", 1000, 50000, 10000, 1000)
show_debug = st.checkbox("Mostrar detalles de depuración")

if st.button("Calcular") and input_text.strip():
    try:
        grid = parse_heat_map(input_text, debug=show_debug)
        if show_debug:
            st.write("Shape de DataFrame:", grid.shape)
            st.dataframe(grid.head())
        with st.spinner("Simulando …"):
            p_reg, p_rev = simulate(grid, iterations)
        st.metric("≥ 11 aciertos (Regular)", f"{p_reg*100:.2f}%")
        st.metric("≥ 6 aciertos (Revancha)", f"{p_rev*100:.2f}%")
        st.caption(f"{iterations:,} simulaciones · probabilidades implícitas")
    except Exception as exc:
        st.error(f"Error al procesar la malla: {exc}")
