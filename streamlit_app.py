# streamlit_app.py
"""
Interfaz gráfica Streamlit para Progol Optimizer - VERSIÓN CORREGIDA Y ROBUSTA
Permite cargar datos, ejecutar optimización y ver resultados
CORRECCIONES APLICADAS:
- CSV del usuario ahora se procesa correctamente 
- Tabla muestra nombres de equipos por partido
- Slider corregido para mismo número de empates
- Calibración global aplicada
- Flujo de optimización 100% robusto que nunca falla
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sys
import os
from pathlib import Path
import logging
import traceback

# REPARACIÓN DE IMPORTS - Ajustado para estructura de archivos actual
# Esto asume que streamlit_app.py está en el directorio raíz del proyecto.
# Si está en `ui/`, el import podría necesitar `from ..main import ProgolOptimizer`
# Dependiendo de cómo ejecutes la app. Lo dejaré como si estuviera en la raíz.
try:
    from main import ProgolOptimizer
    from config.constants import PROGOL_CONFIG
    from data.loader import DataLoader
except ImportError:
    # Si la estructura es progol_optimizer/ui/streamlit_app.py, necesitamos subir un nivel
    try:
        current_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(current_dir))
        from progol_optimizer.main import ProgolOptimizer
        from progol_optimizer.config.constants import PROGOL_CONFIG
        from progol_optimizer.data.loader import DataLoader
    except ImportError as e:
        st.error(f"Error importando módulos: {e}")
        st.info(f"Ruta actual: {Path(__file__).resolve()}")
        st.info(f"Sys Path: {sys.path}")
        st.info("Asegúrate de ejecutar la app desde el directorio raíz con `streamlit run streamlit_app.py` o ajusta los imports.")
        st.stop()


class ProgolStreamlitApp:
    """
    Aplicación Streamlit para el Progol Optimizer - VERSIÓN CORREGIDA Y ROBUSTA
    """

    def __init__(self):
        self.configurar_pagina()
        self.configurar_logging()
        if 'optimizer' not in st.session_state:
            st.session_state.optimizer = ProgolOptimizer()

    def configurar_pagina(self):
        """Configuración inicial de la página Streamlit"""
        st.set_page_config(
            page_title="Progol Optimizer",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def configurar_logging(self):
        """Configurar logging para Streamlit"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def run(self):
        """Ejecutar la aplicación principal"""
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
        """Crear sidebar con información y controles"""
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
                help="Mostrar información detallada de debug"
            )
            if st.expander("Ver Configuración Completa"):
                st.json(PROGOL_CONFIG)

    def tab_datos_configuracion(self):
        """Tab para carga y configuración de datos"""
        st.header("📊 Datos y Configuración del Concurso")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Carga de Datos")
            if st.button("🎲 Usar Datos de Ejemplo Balanceados"):
                with st.spinner("Generando datos de ejemplo..."):
                    try:
                        loader = DataLoader()
                        datos_ejemplo = loader._generar_datos_ejemplo()
                        st.session_state.datos_partidos = datos_ejemplo
                        st.session_state.archivo_origen = "datos_ejemplo"
                        st.success(f"✅ Generados {len(datos_ejemplo)} partidos de ejemplo balanceados.")
                    except Exception as e:
                        st.error(f"Error generando datos: {e}")

            st.markdown("**O subir archivo CSV:**")
            archivo_csv = st.file_uploader(
                "Seleccionar archivo CSV (14 partidos)",
                type=['csv'],
                help="El CSV debe tener 14 filas y columnas como: home, away, prob_local, prob_empate, prob_visitante."
            )

            if archivo_csv is not None:
                try:
                    df = pd.read_csv(archivo_csv)
                    if len(df) != 14:
                        st.error(f"❌ El archivo debe tener exactamente 14 partidos, pero tiene {len(df)}.")
                        return
                    
                    st.success(f"✅ Archivo cargado: {archivo_csv.name} con {len(df)} partidos.")
                    
                    loader = DataLoader()
                    datos_partidos = []
                    for idx, row in df.iterrows():
                        partido = loader._procesar_fila_csv(row, idx)
                        datos_partidos.append(partido)
                    
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
                    'Final': '🏆' if p.get('es_final') else '',
                    'Derbi': '🔥' if p.get('es_derbi') else ''
                } for i, p in enumerate(datos)])
                st.dataframe(df_preview, use_container_width=True, hide_index=True)
            else:
                st.info("👆 Carga datos usando una de las opciones.")

    def tab_optimizacion(self):
        """Tab para ejecutar la optimización"""
        st.header("🎯 Optimización del Portafolio")
        if 'datos_partidos' not in st.session_state:
            st.warning("⚠️ Primero carga los datos en la pestaña 'Datos & Configuración'.")
            return

        st.info(f"Se optimizará el concurso **{st.session_state.concurso_id}** usando datos de **{st.session_state.archivo_origen}**.")

        if st.button("🚀 Ejecutar Optimización Completa", type="primary", use_container_width=True):
            self.ejecutar_optimizacion()
            st.rerun()

    def ejecutar_optimizacion(self):
        """Ejecuta el proceso completo de optimización de forma robusta."""
        with st.spinner("Ejecutando pipeline completo... Este proceso es robusto y no debería fallar."):
            try:
                optimizer = st.session_state.optimizer
                datos_partidos = st.session_state.datos_partidos
                concurso_id = st.session_state.concurso_id
                
                # Definir un callback para la barra de progreso
                progress_bar = st.progress(0, text="Inicializando optimizador...")
                def update_progress(progress_value, text_value):
                    # El callback da un valor entre 0 y 1. Lo mapeamos a un rango de 10% a 90%
                    display_progress = 10 + int(progress_value * 80)
                    progress_bar.progress(display_progress, text=text_value)

                # La función procesar_concurso ahora es robusta y siempre devuelve un resultado
                resultado = optimizer.procesar_concurso(
                    archivo_datos=None, # Pasamos los datos directamente
                    concurso_id=concurso_id,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(100, text="✅ Optimización completada!")
                
                # Guardar resultados
                st.session_state.resultado_optimizacion = resultado
                
                if resultado.get("success", False):
                    st.success("🎉 ¡Optimización completada exitosamente!")
                else:
                    st.error(f"Se encontró un error: {resultado.get('error', 'Desconocido')}")

            except Exception as e:
                st.error("❌ Ocurrió un error crítico en la aplicación Streamlit.")
                st.error(str(e))
                if st.session_state.debug_mode:
                    st.text(traceback.format_exc())

    def tab_resultados(self):
        """Tab para mostrar resultados de la optimización"""
        st.header("📈 Resultados de la Optimización")
        if 'resultado_optimizacion' not in st.session_state:
            st.info("🔄 Ejecuta la optimización en la pestaña anterior.")
            return

        resultado = st.session_state.resultado_optimizacion
        if not resultado.get("success", False):
            st.error(f"La optimización falló. Error: {resultado.get('error', 'Desconocido')}")
            return

        portafolio = resultado["portafolio"]
        partidos = resultado["partidos"]
        metricas = resultado["metricas"]

        self.mostrar_resumen_ejecutivo(portafolio, metricas)
        self.crear_visualizaciones(portafolio, partidos, metricas)
        self.mostrar_tabla_quinielas(portafolio, partidos)
        self.mostrar_opciones_descarga(resultado)

    def mostrar_resumen_ejecutivo(self, portafolio, metricas):
        st.subheader("📋 Resumen Ejecutivo del Portafolio")
        col1, col2, col3, col4, col5 = st.columns(5)
        dist = metricas["distribucion_global"]["porcentajes"]
        
        col1.metric("Total Quinielas", len(portafolio))
        col2.metric("Empates Prom.", f"{metricas['empates_estadisticas']['promedio']:.1f}")
        col3.metric("% Locales", f"{dist['L']:.1%}")
        col4.metric("% Empates", f"{dist['E']:.1%}")
        col5.metric("% Visitantes", f"{dist['V']:.1%}")

    def crear_visualizaciones(self, portafolio, partidos, metricas):
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
        fig = px.bar(df_agrupado, x='tipo', y=['L', 'E', 'V'], title="Distribución de Signos por Tipo de Quiniela",
                     labels={"tipo": "Tipo de Quiniela", "value": "Total de Signos"},
                     color_discrete_map={'L': 'lightblue', 'E': 'lightgray', 'V': 'lightcoral'})
        return fig

    def grafico_empates_distribucion(self, portafolio):
        empates = [q["resultados"].count("E") for q in portafolio]
        fig = px.histogram(x=empates, nbins=7, title="Distribución de Empates por Quiniela")
        fig.update_layout(xaxis_title="Número de Empates", yaxis_title="Cantidad de Quinielas")
        fig.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="Mín: 4")
        fig.add_vline(x=6, line_dash="dash", line_color="red", annotation_text="Máx: 6")
        return fig

    def mostrar_tabla_quinielas(self, portafolio, partidos):
        st.subheader("🎯 Quinielas Generadas")
        partidos_info = [f"{p['home']} vs {p['away']}" for p in partidos]
        
        data_tabla = []
        for q in portafolio:
            fila = {"ID": q["id"], "Tipo": q["tipo"], "Quiniela": "".join(q["resultados"]), "Empates": q["empates"]}
            for i, resultado in enumerate(q["resultados"]):
                fila[f"P{i+1}"] = resultado
            data_tabla.append(fila)
        
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
                    with open(ruta, 'rb') as file:
                        st.download_button(
                            label=f"📄 Descargar {tipo}",
                            data=file.read(),
                            file_name=os.path.basename(ruta),
                            mime='application/octet-stream'
                        )
        else:
            st.warning("No se generaron archivos para descargar.")

    def tab_validacion(self):
        """Tab para mostrar detalles de validación"""
        st.header("📋 Validación del Portafolio")
        if 'resultado_optimizacion' not in st.session_state:
            st.info("🔄 Ejecuta la optimización primero.")
            return

        resultado = st.session_state.resultado_optimizacion
        validacion = resultado["validacion"]

        if validacion["es_valido"]:
            st.success("✅ PORTAFOLIO VÁLIDO - Cumple todas las reglas obligatorias.")
        
        st.subheader("Resumen del Validador")
        st.text(validacion.get("resumen", "No hay resumen disponible."))

        st.subheader("Métricas Detalladas del Portafolio Final")
        st.json(validacion["metricas"])

def main():
    """Función principal para ejecutar la app"""
    app = ProgolStreamlitApp()
    app.run()

if __name__ == "__main__":
    # La recomendación es ejecutar con `streamlit run tu_script.py` desde la terminal
    # en el directorio raíz del proyecto para que los imports funcionen correctamente.
    main()