# streamlit_app.py
"""
Interfaz gráfica Streamlit para Progol Optimizer - VERSIÓN CORREGIDA Y ROBUSTA
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import sys
import os
from pathlib import Path
import logging
import traceback

# Ajuste para la estructura de carpetas correcta
try:
    # Intenta importar asumiendo que el script se corre desde la raíz
    from main import ProgolOptimizer
    from config.constants import PROGOL_CONFIG
    from data.loader import DataLoader
except ImportError:
    # Fallback si se corre desde adentro de la carpeta `ui`
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from progol_optimizer.main import ProgolOptimizer
    from progol_optimizer.config.constants import PROGOL_CONFIG
    from progol_optimizer.data.loader import DataLoader


class ProgolStreamlitApp:
    """
    Aplicación Streamlit para el Progol Optimizer - VERSIÓN CORREGIDA Y ROBUSTA
    """

    def __init__(self):
        self.configurar_pagina()
        self.configurar_logging()
        # Inicializar el optimizador una sola vez y guardarlo en el estado de la sesión
        if 'optimizer' not in st.session_state:
            st.session_state.optimizer = ProgolOptimizer()

    def configurar_pagina(self):
        st.set_page_config(
            page_title="Progol Optimizer",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def configurar_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def run(self):
        st.title("⚽ Progol Optimizer")
        st.markdown("### Metodología Definitiva - Implementación Robusta")
        st.markdown("---")

        self.crear_sidebar()

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Datos & Configuración",
            "🎯 Optimización",
            "📈 Resultados",
            "📋 Validación"
        ])

        with tab1:
            self.tab_datos_configuracion()
        with tab2:
            self.tab_optimizacion()
        with tab3:
            self.tab_resultados()
        with tab4:
            self.tab_validacion()

    def crear_sidebar(self):
        with st.sidebar:
            st.header("🔧 Configuración")
            st.info(
                "**Basado en el documento técnico:**\n"
                "- Distribución: 38% L, 29% E, 33% V \n"
                "- Arquitectura: 4 Core + 26 Satélites \n"
                "- Optimización: GRASP-Annealing \n"
                "- Regla: 4-6 empates por quiniela "
            )
            st.subheader("Parámetros de Optimización")
            st.session_state.concurso_id = st.text_input(
                "ID del Concurso",
                value=st.session_state.get('concurso_id', '2283'),
                help="Identificador del concurso a procesar"
            )
            st.session_state.debug_mode = st.checkbox(
                "Modo Debug",
                value=st.session_state.get('debug_mode', False),
                help="Mostrar información detallada de debug en la consola"
            )
            if st.expander("Ver Configuración Completa"):
                st.json(PROGOL_CONFIG)

    def tab_datos_configuracion(self):
        st.header("📊 Datos y Configuración del Concurso")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Carga de Datos")
            if st.button("🎲 Usar Datos de Ejemplo Balanceados", key="load_example"):
                with st.spinner("Generando datos de ejemplo..."):
                    loader = DataLoader()
                    st.session_state.datos_partidos = loader._generar_datos_ejemplo()
                    st.session_state.archivo_origen = "Datos de Ejemplo Balanceados"
                    st.success(f"✅ Generados {len(st.session_state.datos_partidos)} partidos de ejemplo.")
                    st.rerun()

            st.markdown("**O subir archivo CSV:**")
            archivo_csv = st.file_uploader(
                "Seleccionar archivo CSV (14 partidos)",
                type=['csv'],
                help="El CSV debe tener 14 filas y columnas: home, away, prob_local, prob_empate, prob_visitante."
            )

            if archivo_csv is not None:
                try:
                    df = pd.read_csv(archivo_csv)
                    if len(df) != 14:
                        st.error(f"❌ El archivo debe tener exactamente 14 partidos, pero tiene {len(df)}.")
                        return
                    
                    st.success(f"✅ Archivo cargado: {archivo_csv.name} con {len(df)} partidos.")
                    loader = DataLoader()
                    datos_partidos = [loader._procesar_fila_csv(row, idx) for idx, row in df.iterrows()]
                    
                    st.session_state.datos_partidos = datos_partidos
                    st.session_state.archivo_origen = archivo_csv.name
                    st.success(f"✅ Datos procesados y listos para optimizar.")

                except Exception as e:
                    st.error(f"Error procesando CSV: {e}")
                    if st.session_state.debug_mode:
                        st.exception(e)

        with col2:
            st.subheader("Vista Previa de Datos")
            if 'datos_partidos' in st.session_state:
                datos = st.session_state.datos_partidos
                st.info(f"Fuente de datos: **{st.session_state.get('archivo_origen', 'N/A')}**")
                
                df_preview = pd.DataFrame([{
                    '#': i+1,
                    'Partido': f"{p['home']} vs {p['away']}",
                    'P(L)': f"{p['prob_local']:.3f}",
                    'P(E)': f"{p['prob_empate']:.3f}",
                    'P(V)': f"{p['prob_visitante']:.3f}",
                } for i, p in enumerate(datos)])
                st.dataframe(df_preview, use_container_width=True, hide_index=True)
            else:
                st.info("👆 Carga datos para ver la vista previa.")

    def tab_optimizacion(self):
        st.header("🎯 Optimización del Portafolio")
        if 'datos_partidos' not in st.session_state:
            st.warning("⚠️ Primero carga los datos en la pestaña 'Datos & Configuración'.")
            return

        st.info(f"Se optimizará el concurso **{st.session_state.concurso_id}** usando datos de **{st.session_state.archivo_origen}**.")

        if st.button("🚀 Ejecutar Optimización Completa", type="primary", use_container_width=True):
            with st.spinner("Ejecutando pipeline completo... Este proceso es robusto y no debería fallar."):
                optimizer = st.session_state.optimizer
                
                progress_bar = st.progress(0, text="Inicializando...")
                def update_progress(progress, text):
                    display_progress = 10 + int(progress * 80)
                    progress_bar.progress(display_progress, text=text)

                resultado = optimizer.procesar_concurso(
                    datos_partidos=st.session_state.datos_partidos,
                    concurso_id=st.session_state.concurso_id,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(100, "✅ ¡Proceso completado!")
                st.session_state.resultado_optimizacion = resultado
            st.rerun()

    def tab_resultados(self):
        st.header("📈 Resultados de la Optimización")
        if 'resultado_optimizacion' not in st.session_state:
            st.info("🔄 Ejecuta la optimización en la pestaña anterior.")
            return

        resultado = st.session_state.resultado_optimizacion
        
        # **CORRECCIÓN CLAVE**: Verificar si la optimización fue exitosa
        if not resultado.get("success", False):
            st.error(f"La optimización falló. Error reportado: {resultado.get('error', 'Desconocido')}")
            return

        portafolio = resultado["portafolio"]
        partidos = resultado["partidos"]
        metricas = resultado["metricas"]

        self.mostrar_resumen_ejecutivo(portafolio, metricas)
        self.crear_visualizaciones(portafolio, partidos)
        self.mostrar_tabla_quinielas(portafolio, partidos)
        self.mostrar_opciones_descarga(resultado)

    def tab_validacion(self):
        st.header("📋 Validación del Portafolio")
        if 'resultado_optimizacion' not in st.session_state:
            st.info("🔄 Ejecuta la optimización primero.")
            return

        resultado = st.session_state.resultado_optimizacion
        
        # **CORRECCIÓN CLAVE**: Verificar si la optimización fue exitosa
        if not resultado.get("success", False):
            st.error(f"No se puede mostrar la validación porque la optimización falló. Error: {resultado.get('error', 'Desconocido')}")
            return
            
        validacion = resultado["validacion"]

        if validacion.get("es_valido", False):
            st.success("✅ PORTAFOLIO VÁLIDO - Cumple todas las reglas obligatorias.")
        else:
            st.error("❌ PORTAFOLIO INVÁLIDO - Se encontró un problema inesperado.")

        st.subheader("Resumen del Validador")
        st.text(validacion.get("resumen", "No hay resumen disponible."))

        if "metricas" in validacion and validacion["metricas"]:
            st.subheader("Métricas Detalladas del Portafolio Final")
            st.json(validacion["metricas"])
        else:
            st.warning("No se generaron métricas detalladas.")

    # Las funciones de visualización y resumen no necesitan cambios
    def mostrar_resumen_ejecutivo(self, portafolio, metricas):
        st.subheader("📋 Resumen Ejecutivo del Portafolio")
        col1, col2, col3, col4, col5 = st.columns(5)
        dist = metricas["distribucion_global"]["porcentajes"]
        
        col1.metric("Total Quinielas", len(portafolio))
        col2.metric("Empates Prom.", f"{metricas['empates_estadisticas']['promedio']:.1f}")
        col3.metric("% Locales", f"{dist['L']:.1%}")
        col4.metric("% Empates", f"{dist['E']:.1%}")
        col5.metric("% Visitantes", f"{dist['V']:.1%}")

    def crear_visualizaciones(self, portafolio, partidos):
        st.subheader("📊 Visualizaciones")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(self.grafico_distribucion_tipos(portafolio), use_container_width=True)
        with col2:
            st.plotly_chart(self.grafico_empates_distribucion(portafolio), use_container_width=True)

    def grafico_distribucion_tipos(self, portafolio):
        df_tipos = pd.DataFrame([
            {"tipo": q['tipo'], "L": q['resultados'].count('L'), "E": q['resultados'].count('E'), "V": q['resultados'].count('V')}
            for q in portafolio
        ])
        df_agrupado = df_tipos.groupby('tipo').sum().reset_index()
        df_melted = df_agrupado.melt(id_vars='tipo', value_vars=['L', 'E', 'V'], var_name='Signo', value_name='Total')
        
        fig = px.bar(df_melted, x='tipo', y='Total', color='Signo', title="Distribución de Signos por Tipo de Quiniela",
                     labels={"tipo": "Tipo de Quiniela", "Total": "Cantidad de Signos"},
                     color_discrete_map={'L': 'lightblue', 'E': 'lightgray', 'V': 'lightcoral'})
        return fig

    def grafico_empates_distribucion(self, portafolio):
        empates = [q["resultados"].count("E") for q in portafolio]
        fig = px.histogram(x=empates, nbins=max(empates) - min(empates) + 1 if empates else 1, title="Distribución de Empates por Quiniela")
        fig.update_layout(xaxis_title="Número de Empates", yaxis_title="Cantidad de Quinielas")
        fig.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="Mín: 4")
        fig.add_vline(x=6, line_dash="dash", line_color="red", annotation_text="Máx: 6")
        return fig

    def mostrar_tabla_quinielas(self, portafolio, partidos):
        st.subheader("🎯 Quinielas Generadas")
        
        data_tabla = [{"ID": q["id"], "Tipo": q["tipo"], **{f"P{i+1}": res for i, res in enumerate(q["resultados"])}, "Empates": q["empates"]} for q in portafolio]
        df_quinielas = pd.DataFrame(data_tabla)
        
        with st.expander("Ver tabla detallada de quinielas"):
            st.dataframe(df_quinielas, use_container_width=True, hide_index=True)

    def mostrar_opciones_descarga(self, resultado):
        st.subheader("💾 Descargar Resultados")
        archivos = resultado.get("archivos_exportados", {})
        if archivos:
            st.success(f"✅ Se generaron {len(archivos)} archivos de reporte:")
            for tipo, ruta in archivos.items():
                if os.path.exists(ruta):
                    with open(ruta, 'rb') as file_content:
                        st.download_button(
                            label=f"📄 Descargar {os.path.basename(ruta)}",
                            data=file_content,
                            file_name=os.path.basename(ruta),
                            mime='application/octet-stream'
                        )
        else:
            st.warning("No se generaron archivos para descargar.")

def main():
    ProgolStreamlitApp().run()

if __name__ == "__main__":
    main()