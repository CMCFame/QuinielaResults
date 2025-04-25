# ===========================================================
# app.py
# ===========================================================
import streamlit as st
from parser import parse_heat_map
from simulator import simulate

st.set_page_config(page_title="Progol Monte‑Carlo Analyzer", page_icon="⚽")
st.title("Progol 2278 · Probabilidad de Premio")

st.markdown("Pegue aquí su **malla de 20 quinielas** (21 filas, 20 columnas de L/E/V):")
user_text = st.text_area("", height=400)

iterations = st.slider("Iteraciones Monte‑Carlo", 1000, 50000, 10000, 1000)

if st.button("Calcular probabilidades") and user_text.strip():
    try:
        grid = parse_heat_map(user_text)
        if grid.shape[0] != 21 or grid.shape[1] != 20:
            st.error("La malla debe contener 21 filas y 20 columnas de predicciones.")
        else:
            with st.spinner("Simulando ..."):
                p_reg, p_rev = simulate(grid, n_iter=iterations)
            st.success("Resultado Monte‑Carlo")
            st.metric("≥ 11 aciertos (14 principales)", f"{p_reg*100:.2f}%")
            st.metric("≥ 6 aciertos (7 revancha)",   f"{p_rev*100:.2f}%")
            st.caption("Basado en probabilidades implícitas y {iterations:,} simulaciones.")
    except Exception as e:
        st.error(f"Error al procesar la malla: {e}")
