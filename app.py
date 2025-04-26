# ==========================================================
# app.py  (con visualización detallada por quiniela)
# ==========================================================
import streamlit as st, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from parser import parse_heat_map
from simulator import simulate

st.set_page_config(page_title="Progol Monte‑Carlo Analyzer", page_icon="⚽", layout="wide")
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
            
        # Simulación Monte-Carlo
        p_reg, p_rev, detail = simulate(df, iterations)
        
        # Mostrar métricas generales
        col1, col2 = st.columns(2)
        with col1:
            st.metric("≥ 11 aciertos (Regular)",  f"{p_reg*100:.2f}%")
        with col2:
            st.metric("≥ 6 aciertos (Revancha)", f"{p_rev*100:.2f}%")
            
        # Crear DataFrame con detalles por quiniela
        detail_df = pd.DataFrame({
            'Quiniela': list(detail.keys()),
            'Aciertos Promedio (Regular)': [detail[q]['avg_hits_reg'] for q in detail.keys()],
            'Probabilidad Ganar (Regular) %': [detail[q]['prob_win_reg'] * 100 for q in detail.keys()],
            'Aciertos Promedio (Revancha)': [detail[q]['avg_hits_rev'] for q in detail.keys()],
            'Probabilidad Ganar (Revancha) %': [detail[q]['prob_win_rev'] * 100 for q in detail.keys()]
        })
        
        # Ordenar por probabilidad de ganar (regular)
        detail_df = detail_df.sort_values('Probabilidad Ganar (Regular) %', ascending=False)
        
        # Mostrar visualizaciones
        st.subheader("Análisis detallado por quiniela")
        
        tabs = st.tabs(["Tabla de datos", "Gráfico Regular", "Gráfico Revancha", "Mapa de calor"])
        
        with tabs[0]:
            st.dataframe(detail_df, use_container_width=True)
        
        with tabs[1]:
            # Gráfico Regular
            fig_reg = px.bar(
                detail_df,
                x='Quiniela',
                y='Probabilidad Ganar (Regular) %',
                color='Aciertos Promedio (Regular)',
                color_continuous_scale='Viridis',
                title='Probabilidad de ganar por quiniela (Regular)',
                labels={'Probabilidad Ganar (Regular) %': 'Probabilidad %', 'Quiniela': 'Quiniela'}
            )
            fig_reg.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig_reg, use_container_width=True)
        
        with tabs[2]:
            # Gráfico Revancha
            fig_rev = px.bar(
                detail_df,
                x='Quiniela',
                y='Probabilidad Ganar (Revancha) %',
                color='Aciertos Promedio (Revancha)',
                color_continuous_scale='Viridis',
                title='Probabilidad de ganar por quiniela (Revancha)',
                labels={'Probabilidad Ganar (Revancha) %': 'Probabilidad %', 'Quiniela': 'Quiniela'}
            )
            fig_rev.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig_rev, use_container_width=True)
            
        with tabs[3]:
            # Mapa de calor de las probabilidades
            heat_data = detail_df.pivot(index='Quiniela', values=['Probabilidad Ganar (Regular) %', 'Probabilidad Ganar (Revancha) %'])
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_data.values,
                x=heat_data.columns.tolist(),
                y=heat_data.index.tolist(),
                colorscale='Viridis',
                colorbar=dict(title='Probabilidad %')
            ))
            fig_heat.update_layout(
                title='Mapa de calor de probabilidades por quiniela',
                xaxis_title='Tipo de juego',
                yaxis_title='Quiniela',
                height=600
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
    except Exception as ex:
        st.error(f"Error: {ex}")