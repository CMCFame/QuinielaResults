import streamlit as st
from parser import parse_heat_map
from simulator import simulate

st.set_page_config(page_title="Progol Monte-Carlo Analyzer", page_icon="⚽")

st.title("Progol 2278 · Calculadora de Probabilidades")

st.markdown(
    "Pegue **21 filas** con **20 códigos (L/E/V)** por fila. "
    "Puede usar dobles «L/E». La cabecera “Partido C1…” se ignora.",
    unsafe_allow_html=True,
)

text = st.text_area("Heat-map de 20 quinielas", height=360)
iters = st.slider("Iteraciones Monte-Carlo", 1_000, 50_000, 10_000, 1_000)
dbg   = st.checkbox("Depuración")

if st.button("Calcular") and text.strip():
    try:
        grid = parse_heat_map(text, debug=dbg)
        if dbg:
            st.write("Shape:", grid.shape)
            st.dataframe(grid.head())
        with st.spinner("Simulando…"):
            p_reg, p_rev = simulate(grid, iters, debug=dbg)
        st.success("Resultados")
        st.metric("≥ 11 aciertos (Regular)",  f"{p_reg*100:.2f}%")
        st.metric("≥ 6 aciertos (Revancha)", f"{p_rev*100:.2f}%")
        st.caption(f"{iters:,} simulaciones Monte-Carlo")
    except Exception as e:
        st.error(f"Error: {e}")
