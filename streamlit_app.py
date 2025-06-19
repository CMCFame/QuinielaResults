# streamlit_app.py
"""
Interfaz gráfica Streamlit para Progol Optimizer - VERSIÓN CORREGIDA CON DEBUG AI
Permite cargar datos, ejecutar optimización y ver resultados
NUEVA FUNCIONALIDAD: Ventana de debug para ver comunicación con la IA
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
import tempfile
import numpy as np

# REPARACIÓN DE IMPORTS - Ajustado para estructura de archivos actual
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Importar directamente desde la raíz
try:
    from main import ProgolOptimizer
    from config.constants import PROGOL_CONFIG
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.info("Verificar que existan los archivos main.py y config/constants.py")
    st.stop()

class ProgolStreamlitApp:
    """
    Aplicación Streamlit para el Progol Optimizer - VERSIÓN CORREGIDA CON DEBUG AI
    """

    def __init__(self):
        self.configurar_pagina()
        self.configurar_logging()

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
        # Título principal
        st.title("⚽ Progol Optimizer")
        st.markdown("### Metodología Definitiva - Implementación Exacta")
        st.markdown("---")

        # Sidebar con configuración
        self.crear_sidebar()

        # Verificar si debemos ejecutar optimización con AI automáticamente
        if hasattr(st.session_state, 'ejecutar_con_ai') and st.session_state.ejecutar_con_ai:
            st.session_state.ejecutar_con_ai = False
            # Ir directamente a optimización
            tab_index = 1
        else:
            tab_index = 0

        # Contenido principal
        tabs = st.tabs([
            "📊 Datos & Configuración",
            "🎯 Optimización",
            "📈 Resultados",
            "📋 Validación"
        ])

        with tabs[0]:
            self.tab_datos_configuracion()

        with tabs[1]:
            self.tab_optimizacion()
            # Si venimos de validación, ejecutar automáticamente
            if hasattr(st.session_state, 'ejecutar_con_ai') and st.session_state.ejecutar_con_ai:
                self.ejecutar_optimizacion(forzar_ai=True)

        with tabs[2]:
            self.tab_resultados()

        with tabs[3]:
            self.tab_validacion()

    def crear_sidebar(self):
        """Crear sidebar con información y controles"""
        with st.sidebar:
            st.header("🔧 Configuración")

            # Información del documento
            st.info("""
            **Basado en el documento técnico:**
            - 38% Locales, 29% Empates, 33% Visitantes
            - 4-6 empates por quiniela
            - Arquitectura 4 Core + 26 Satélites
            - Optimización GRASP-Annealing
            """)

            # Parámetros configurables
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

            # NUEVA SECCIÓN AI
            st.markdown("---")
            st.subheader("🤖 Configuración AI")

            # Verificar si hay API key en secrets
            if "OPENAI_API_KEY" in st.secrets:
                # API key está en secrets
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                st.success("✅ AI habilitada (API key desde secrets)")
                st.caption("La clave API está configurada de forma segura")
                
                # Opción para usar una clave diferente temporalmente
                with st.expander("Usar API key diferente (opcional)"):
                    temp_key = st.text_input(
                        "API Key temporal",
                        type="password",
                        help="Solo si quieres usar una clave diferente para esta sesión"
                    )
                    if temp_key:
                        os.environ["OPENAI_API_KEY"] = temp_key
                        st.session_state.openai_api_key = temp_key
                        st.info("Usando API key temporal para esta sesión")
            else:
                # No hay API key en secrets, permitir ingreso manual
                st.warning("⚠️ No se encontró API key en configuración")
                
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Ingresa tu API key de OpenAI para habilitar corrección inteligente",
                    key="openai_api_key"
                )
                
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("✅ API Key configurada para esta sesión")
                else:
                    st.info("ℹ️ AI deshabilitada - Configura la API key en secrets o ingrésala arriba")
                    
                # Instrucciones para configurar en secrets
                with st.expander("📖 Cómo configurar API key permanentemente"):
                    st.markdown("""
                    **Para Streamlit Cloud:**
                    1. Ve a tu app en [share.streamlit.io](https://share.streamlit.io)
                    2. Click en ⚙️ Settings → Secrets
                    3. Agrega:
                    ```toml
                    OPENAI_API_KEY = "sk-tu-api-key-aqui"
                    ```
                    4. Click en "Save" y la app se reiniciará
                    
                    **Para desarrollo local:**
                    1. Crea `.streamlit/secrets.toml` en la raíz del proyecto
                    2. Agrega la misma línea de arriba
                    3. Asegúrate de que `.streamlit/secrets.toml` esté en `.gitignore`
                    """)

            # Mostrar configuración actual
            if st.expander("Ver Configuración Actual"):
                st.json(PROGOL_CONFIG)
                
    def tab_datos_configuracion(self):
        """Tab para carga y configuración de datos - CORREGIDO"""
        st.header("📊 Datos y Configuración")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Carga de Datos")

            # Opción 1: Usar datos de ejemplo
            if st.button("🎲 Usar Datos de Ejemplo", type="secondary"):
                with st.spinner("Generando datos de ejemplo con Anclas garantizadas..."):
                    try:
                        from data.loader import DataLoader
                        loader = DataLoader()
                        datos_ejemplo = loader._generar_datos_ejemplo()
                        st.session_state.datos_partidos = datos_ejemplo
                        st.session_state.archivo_origen = "datos_ejemplo_corregidos"
                        st.success(f"✅ Generados {len(datos_ejemplo)} partidos de ejemplo con Anclas garantizadas")

                    except Exception as e:
                        st.error(f"Error generando datos: {e}")

            # Opción 2: Subir archivo CSV - CORREGIDO
            st.markdown("**O subir archivo CSV:**")
            archivo_csv = st.file_uploader(
                "Seleccionar archivo CSV",
                type=['csv'],
                help="CSV con columnas: home, away, liga, prob_local, prob_empate, prob_visitante"
            )

            if archivo_csv is not None:
                try:
                    # CORRECCIÓN: Leer CSV del usuario correctamente
                    df = pd.read_csv(archivo_csv)
                    
                    # Verificar que tenga exactamente 14 filas
                    if len(df) != 14:
                        st.error(f"❌ El archivo debe tener exactamente 14 partidos, tiene {len(df)}")
                        return
                    
                    st.success(f"✅ Archivo cargado: {len(df)} partidos")

                    # CORRECCIÓN: Convertir DataFrame a formato interno correctamente
                    datos_partidos = []
                    for idx, row in df.iterrows():
                        # Verificar columnas requeridas
                        if 'home' not in row or 'away' not in row:
                            st.error("❌ El CSV debe tener columnas 'home' y 'away'")
                            return
                            
                        # Probabilidades: usar las del CSV o generar si no existen
                        if all(col in row for col in ['prob_local', 'prob_empate', 'prob_visitante']):
                            prob_local = float(row['prob_local'])
                            prob_empate = float(row['prob_empate'])
                            prob_visitante = float(row['prob_visitante'])
                            
                            # Verificar que sumen ~1.0
                            total = prob_local + prob_empate + prob_visitante
                            if abs(total - 1.0) > 0.05:
                                st.warning(f"⚠️ Partido {idx+1}: probabilidades suman {total:.3f}, normalizando...")
                                prob_local /= total
                                prob_empate /= total 
                                prob_visitante /= total
                        else:
                            # Generar probabilidades realistas si no están en CSV
                            from data.loader import DataLoader
                            loader = DataLoader()
                            prob_local, prob_empate, prob_visitante = loader._generar_probabilidades_balanceadas_por_partido(idx)

                        partido = {
                            'id': idx,
                            'home': str(row['home']).strip(),
                            'away': str(row['away']).strip(),
                            'liga': str(row.get('liga', 'Liga')).strip(),
                            'prob_local': prob_local,
                            'prob_empate': prob_empate,
                            'prob_visitante': prob_visitante,
                            'forma_diferencia': float(row.get('forma_diferencia', 0)),
                            'lesiones_impact': float(row.get('lesiones_impact', 0)),
                            'es_final': bool(row.get('es_final', False)),
                            'es_derbi': bool(row.get('es_derbi', False)),
                            'es_playoff': bool(row.get('es_playoff', False)),
                            'fecha': str(row.get('fecha', '2025-06-07')),
                            'jornada': int(row.get('jornada', 1)),
                            'concurso_id': str(row.get('concurso_id', st.session_state.concurso_id))
                        }
                        datos_partidos.append(partido)

                    # CORRECCIÓN: Asegurar que se guarden los datos del usuario
                    st.session_state.datos_partidos = datos_partidos
                    st.session_state.archivo_origen = archivo_csv.name
                    
                    st.success(f"✅ **Datos de {archivo_csv.name} cargados correctamente**")

                except Exception as e:
                    st.error(f"Error procesando CSV: {e}")
                    if st.session_state.debug_mode:
                        st.exception(e)

        with col2:
            st.subheader("Vista Previa de Datos")

            if 'datos_partidos' in st.session_state:
                datos = st.session_state.datos_partidos
                archivo_origen = st.session_state.get('archivo_origen', 'datos_ejemplo')

                # MEJORA: Mostrar origen de datos claramente
                if archivo_origen == 'datos_ejemplo_corregidos':
                    st.info("📊 **Usando datos de ejemplo CORREGIDOS con Anclas garantizadas**")
                elif archivo_origen == 'datos_ejemplo':
                    st.info("📊 **Usando datos de ejemplo generados**")
                else:
                    st.success(f"📄 **Usando datos de: {archivo_origen}**")

                # Crear DataFrame para mostrar - CON NOMBRES DE EQUIPOS
                df_preview = pd.DataFrame([
                    {
                        '#': i+1,
                        'Partido': f"{p['home']} vs {p['away']}",
                        'Liga': p['liga'],
                        'P(L)': f"{p['prob_local']:.3f}",
                        'P(E)': f"{p['prob_empate']:.3f}",
                        'P(V)': f"{p['prob_visitante']:.3f}",
                        'Final': '🏆' if p.get('es_final') else '',
                        'Derbi': '🔥' if p.get('es_derbi') else ''
                    }
                    for i, p in enumerate(datos)
                ])

                st.dataframe(df_preview, use_container_width=True)

                # Estadísticas rápidas
                st.markdown("**Estadísticas:**")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Total Partidos", len(datos))

                with col_b:
                    ligas = len(set(p['liga'] for p in datos))
                    st.metric("Ligas", ligas)

                with col_c:
                    # Verificar potenciales Anclas
                    anclas_potenciales = sum(1 for p in datos if max(p['prob_local'], p['prob_empate'], p['prob_visitante']) > 0.60)
                    st.metric("Anclas Potenciales", anclas_potenciales)

            else:
                st.info("👆 Carga datos usando una de las opciones de arriba")

    def tab_optimizacion(self):
        """Tab para ejecutar la optimización con selector de método."""
        st.header("🎯 Optimización del Portafolio")

        if 'datos_partidos' not in st.session_state:
            st.warning("⚠️ Primero carga los datos en la pestaña 'Datos & Configuración'")
            return

        st.info(f"📄 Se optimizará usando datos de: {st.session_state.get('archivo_origen', 'ejemplo')}")

        # --- NUEVO: SELECTOR DE MÉTODO ---
        st.subheader("1. Selecciona el Método de Optimización")
        metodo_seleccionado = st.radio(
            "Elige el motor para generar el portafolio:",
            options=["Híbrido (Recomendado)", "Heurístico (Heredado)"],
            captions=[
                "Usa Programación Entera + Annealing para garantizar un portafolio válido y óptimo.",
                "Usa el algoritmo original, más rápido pero puede no encontrar soluciones válidas."
            ],
            horizontal=True,
            key="metodo_optimizacion"
        )
        st.session_state.metodo_optimizacion_key = "hybrid" if "Híbrido" in metodo_seleccionado else "legacy"
        st.markdown("---")
        
        st.subheader("2. Ejecuta la Optimización")
        ai_disponible = "OPENAI_API_KEY" in os.environ or "OPENAI_API_KEY" in st.secrets

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Ejecutar Optimización", type="primary", use_container_width=True):
                self.ejecutar_optimizacion(forzar_ai=False)
        with col2:
            if ai_disponible:
                if st.button("🤖 Optimizar con AI (Forzado)", type="secondary", use_container_width=True,
                            help="Fuerza el uso de AI incluso si el resultado inicial es válido"):
                    self.ejecutar_optimizacion(forzar_ai=True)
            else:
                st.button("🤖 Optimizar con AI", disabled=True, use_container_width=True)

        if 'optimizacion_ejecutando' in st.session_state and st.session_state.optimizacion_ejecutando:
            self.mostrar_progreso_optimizacion()

    def ejecutar_optimizacion(self, forzar_ai=False):
        """Ejecutar el proceso completo de optimización"""
        st.session_state.optimizacion_ejecutando = True
        st.session_state.forzar_ai = forzar_ai

        progress_bar = st.progress(0, text="Inicializando...")

        try:
            progress_bar.progress(10, text="🔧 Inicializando optimizador...")
            optimizer = ProgolOptimizer()

            def update_progress(progress_value, text_value):
                display_progress = 30 + int(progress_value * 60)
                progress_bar.progress(display_progress, text=text_value)

            progress_bar.progress(30, text=f"⚙️ Ejecutando optimización con método {st.session_state.metodo_optimizacion_key.upper()}...")
            
            if forzar_ai and optimizer.ai_assistant.enabled:
                progress_bar.progress(35, text="🤖 Optimización con asistencia AI activada...")

            # Llamamos a la función que contiene la lógica principal
            self.ejecutar_optimizacion_directo(optimizer, progress_callback=update_progress, forzar_ai=forzar_ai)
            
            progress_bar.progress(100, text="✅ Optimización completada!")

        except Exception as e:
            st.error(f"❌ Error en optimización: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
        finally:
            st.session_state.optimizacion_ejecutando = False

    def ejecutar_optimizacion_directo(self, optimizer, progress_callback=None, forzar_ai=False):
        """Ejecutar optimización usando datos y método seleccionados."""
        
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        elif hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        
        datos_partidos = st.session_state.datos_partidos
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
            df = pd.DataFrame(datos_partidos)
            df.to_csv(tmp.name, index=False)
            temp_path = tmp.name
        
        try:
            resultado = optimizer.procesar_concurso(
                archivo_datos=temp_path,
                concurso_id=st.session_state.concurso_id,
                forzar_ai=forzar_ai,
                method=st.session_state.metodo_optimizacion_key
            )
            
            st.session_state.resultado_optimizacion = resultado
            
            if resultado and resultado.get("success"):
                self.mostrar_resumen_resultado(resultado)
            else:
                 st.error(f"La optimización falló: {resultado.get('error', 'Error desconocido')}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def mostrar_resumen_resultado(self, resultado):
        """Mostrar resumen inmediato del resultado"""
        
        if not resultado or not resultado.get("success"):
            st.error("❌ Error durante la optimización.")
            return

        st.success("🎉 ¡Optimización completada exitosamente!")

        if resultado.get("ai_utilizada"):
            st.info("🤖 **AI fue utilizada** para corregir problemas de validación")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Quinielas Generadas", len(resultado.get("portafolio", [])))

        with col2:
            es_valido = resultado.get("validacion", {}).get("es_valido", False)
            if es_valido:
                st.metric("Validación", "✅ VÁLIDO", delta="Cumple todas las reglas")
            else:
                st.metric("Validación", "❌ INVÁLIDO", delta="Revisa la pestaña Validación")

        with col3:
            cores = len([q for q in resultado.get("portafolio", []) if q["tipo"] == "Core"])
            st.metric("Quinielas Core", cores)

        with col4:
            satelites = len([q for q in resultado.get("portafolio", []) if q["tipo"] == "Satelite"])
            st.metric("Satélites", satelites)
        
        with col5:
            empates_prom = resultado.get("metricas", {}).get("empates_estadisticas", {}).get("promedio", 0)
            st.metric("Empates Promedio", f"{empates_prom:.1f}")

        if not es_valido:
            st.warning("⚠️ **El portafolio tiene problemas de validación**")
        else:
            st.success("✅ **Todas las reglas de validación cumplidas**")
        
        st.info("📊 Ve a las pestañas **Resultados** y **Validación** para más detalles")

    def mostrar_progreso_optimizacion(self):
        """Mostrar progreso durante la optimización"""
        st.info("⏳ Optimización en progreso...")

    def tab_resultados(self):
        """Tab para mostrar resultados de la optimización"""
        st.header("📈 Resultados de la Optimización")

        if 'resultado_optimizacion' not in st.session_state or not st.session_state.resultado_optimizacion.get("success"):
            st.info("🔄 Ejecuta una optimización exitosa primero en la pestaña 'Optimización'")
            return

        resultado = st.session_state.resultado_optimizacion
        portafolio = resultado["portafolio"]
        partidos = resultado["partidos"]
        metricas = resultado["metricas"]

        self.mostrar_resumen_ejecutivo(portafolio, metricas)
        self.crear_visualizaciones(portafolio, partidos, metricas)
        self.mostrar_tabla_quinielas(portafolio, partidos)
        self.mostrar_opciones_descarga(resultado)

    def mostrar_resumen_ejecutivo(self, portafolio, metricas):
        """Mostrar resumen ejecutivo de resultados"""
        st.subheader("📋 Resumen Ejecutivo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Quinielas", len(portafolio))
        
        with col2:
            cores = len([q for q in portafolio if q["tipo"] == "Core"])
            st.metric("Quinielas Core", cores)
        
        with col3:
            satelites = len([q for q in portafolio if q["tipo"] == "Satelite"])
            st.metric("Satélites", satelites)
        
        with col4:
            empates_prom = metricas.get("empates_estadisticas", {}).get("promedio", 0)
            st.metric("Empates Promedio", f"{empates_prom:.1f}")

        # Distribución global
        if "distribucion_global" in metricas:
            dist = metricas["distribucion_global"]["porcentajes"]
            st.subheader("Distribución Global del Portafolio")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Locales (L)", f"{dist['L']:.1%}", 
                         delta=f"Objetivo: 35-41%")
            with col_b:
                st.metric("Empates (E)", f"{dist['E']:.1%}", 
                         delta=f"Objetivo: 25-33%")
            with col_c:
                st.metric("Visitantes (V)", f"{dist['V']:.1%}", 
                         delta=f"Objetivo: 30-36%")

    def crear_visualizaciones(self, portafolio, partidos, metricas):
        """Crear visualizaciones de los resultados - VERSIÓN ROBUSTA"""
        st.subheader("📊 Visualizaciones")

        try:
            col1, col2 = st.columns(2)

            with col1:
                try:
                    fig_tipos = self.grafico_distribucion_tipos_seguro(portafolio)
                    if fig_tipos:
                        st.plotly_chart(fig_tipos, use_container_width=True)
                    else:
                        st.warning("No se pudo generar gráfico de distribución")
                except Exception as e:
                    st.error(f"Error en gráfico de tipos: {str(e)}")
                    # Mostrar gráfico alternativo simple
                    self.mostrar_tabla_distribucion_simple(portafolio)

            with col2:
                try:
                    fig_empates = self.grafico_empates_distribucion_seguro(portafolio)
                    if fig_empates:
                        st.plotly_chart(fig_empates, use_container_width=True)
                    else:
                        st.warning("No se pudo generar gráfico de empates")
                except Exception as e:
                    st.error(f"Error en gráfico de empates: {str(e)}")
                    self.mostrar_estadisticas_empates_simple(portafolio)

            # Gráfico de clasificación - con manejo robusto
            try:
                fig_clasificacion = self.grafico_clasificacion_partidos_seguro(partidos)
                if fig_clasificacion:
                    st.plotly_chart(fig_clasificacion, use_container_width=True)
                else:
                    st.info("Clasificación de partidos no disponible")
            except Exception as e:
                st.error(f"Error en gráfico de clasificación: {str(e)}")
                self.mostrar_clasificacion_simple(partidos)

        except Exception as e:
            st.error(f"Error general en visualizaciones: {str(e)}")
            st.info("Mostrando información básica en formato de tabla")
            self.mostrar_resumen_basico_seguro(portafolio, partidos, metricas)

    def grafico_distribucion_tipos_seguro(self, portafolio):
        """Gráfico de distribución L/E/V por tipo de quiniela - VERSIÓN SEGURA"""
        try:
            if not portafolio or len(portafolio) == 0:
                return None

            # Validar que el portafolio tenga la estructura correcta
            for q in portafolio:
                if not isinstance(q, dict) or 'tipo' not in q or 'distribución' not in q:
                    return None

            # Agrupar datos por tipo con validación
            datos_por_tipo = {}
            
            for quiniela in portafolio:
                tipo = quiniela.get("tipo", "Desconocido")
                dist = quiniela.get("distribución", {})
                
                if tipo not in datos_por_tipo:
                    datos_por_tipo[tipo] = {"L": 0, "E": 0, "V": 0, "count": 0}
                
                # Validar que la distribución tenga las claves correctas
                if isinstance(dist, dict):
                    datos_por_tipo[tipo]["L"] += dist.get("L", 0)
                    datos_por_tipo[tipo]["E"] += dist.get("E", 0)
                    datos_por_tipo[tipo]["V"] += dist.get("V", 0)
                    datos_por_tipo[tipo]["count"] += 1

            if not datos_por_tipo:
                return None

            # Crear datos para el gráfico
            tipos = list(datos_por_tipo.keys())
            l_values = []
            e_values = []
            v_values = []

            for tipo in tipos:
                count = datos_por_tipo[tipo]["count"]
                if count > 0:
                    l_values.append(datos_por_tipo[tipo]["L"] / count)
                    e_values.append(datos_por_tipo[tipo]["E"] / count)
                    v_values.append(datos_por_tipo[tipo]["V"] / count)
                else:
                    l_values.append(0)
                    e_values.append(0)
                    v_values.append(0)

            # Crear gráfico con validación
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Locales (L)', 
                x=tipos, 
                y=l_values, 
                marker_color='blue',
                text=[f"{v:.1f}" for v in l_values],
                textposition='auto'
            ))
            
            fig.add_trace(go.Bar(
                name='Empates (E)', 
                x=tipos, 
                y=e_values, 
                marker_color='gray',
                text=[f"{v:.1f}" for v in e_values],
                textposition='auto'
            ))
            
            fig.add_trace(go.Bar(
                name='Visitantes (V)', 
                x=tipos, 
                y=v_values, 
                marker_color='red',
                text=[f"{v:.1f}" for v in v_values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Distribución Promedio L/E/V por Tipo de Quiniela',
                xaxis_title='Tipo de Quiniela',
                yaxis_title='Promedio de Resultados',
                barmode='group',
                showlegend=True,
                height=400
            )
            
            return fig

        except Exception as e:
            st.error(f"Error creando gráfico de distribución: {e}")
            return None

    def grafico_empates_distribucion_seguro(self, portafolio):
        """Histograma de distribución de empates - VERSIÓN SEGURA"""
        try:
            if not portafolio or len(portafolio) == 0:
                return None

            # Extraer datos de empates con validación
            empates_data = []
            for q in portafolio:
                if isinstance(q, dict) and 'empates' in q:
                    empates = q.get('empates', 0)
                    if isinstance(empates, (int, float)) and 0 <= empates <= 14:
                        empates_data.append(int(empates))

            if not empates_data:
                return None

            # Crear histograma
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=empates_data,
                nbinsx=max(1, max(empates_data) - min(empates_data) + 1),
                marker_color='lightblue',
                opacity=0.7,
                name='Distribución de Empates'
            ))

            # Agregar líneas de referencia para rangos válidos
            fig.add_vline(x=4, line_dash="dash", line_color="green", 
                          annotation_text="Mín (4)")
            fig.add_vline(x=6, line_dash="dash", line_color="green", 
                          annotation_text="Máx (6)")

            fig.update_layout(
                title='Distribución de Empates por Quiniela',
                xaxis_title='Número de Empates',
                yaxis_title='Frecuencia',
                showlegend=False,
                height=400
            )

            return fig

        except Exception as e:
            st.error(f"Error creando gráfico de empates: {e}")
            return None

    def grafico_clasificacion_partidos_seguro(self, partidos):
        """Gráfico de clasificación de partidos - VERSIÓN SEGURA"""
        try:
            if not partidos or len(partidos) == 0:
                return None

            # Contar clasificaciones con validación
            clasificaciones = {}
            for partido in partidos:
                if isinstance(partido, dict):
                    clase = partido.get("clasificacion", "Sin clasificar")
                    if isinstance(clase, str):
                        clasificaciones[clase] = clasificaciones.get(clase, 0) + 1

            if not clasificaciones:
                return None

            # Crear gráfico de pastel
            labels = list(clasificaciones.keys())
            values = list(clasificaciones.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'][:len(labels)]
            ))

            fig.update_layout(
                title='Clasificación de Partidos',
                showlegend=True,
                height=400
            )

            return fig

        except Exception as e:
            st.error(f"Error creando gráfico de clasificación: {e}")
            return None

    def mostrar_tabla_distribucion_simple(self, portafolio):
        """Mostrar distribución en formato de tabla simple"""
        try:
            datos_tipo = {}
            for q in portafolio:
                tipo = q.get('tipo', 'Desconocido')
                if tipo not in datos_tipo:
                    datos_tipo[tipo] = {'L': 0, 'E': 0, 'V': 0, 'count': 0}
                
                dist = q.get('distribución', {})
                datos_tipo[tipo]['L'] += dist.get('L', 0)
                datos_tipo[tipo]['E'] += dist.get('E', 0)
                datos_tipo[tipo]['V'] += dist.get('V', 0)
                datos_tipo[tipo]['count'] += 1

            st.write("**Distribución por Tipo:**")
            for tipo, datos in datos_tipo.items():
                count = datos['count']
                if count > 0:
                    st.write(f"- **{tipo}**: L={datos['L']/count:.1f}, E={datos['E']/count:.1f}, V={datos['V']/count:.1f}")
        except Exception as e:
            st.write(f"Error mostrando distribución: {e}")

    def mostrar_estadisticas_empates_simple(self, portafolio):
        """Mostrar estadísticas de empates en formato simple"""
        try:
            empates = [q.get('empates', 0) for q in portafolio if 'empates' in q]
            if empates:
                st.write("**Estadísticas de Empates:**")
                st.write(f"- Promedio: {np.mean(empates):.1f}")
                st.write(f"- Mínimo: {min(empates)}")
                st.write(f"- Máximo: {max(empates)}")
        except Exception as e:
            st.write(f"Error mostrando estadísticas de empates: {e}")

    def mostrar_clasificacion_simple(self, partidos):
        """Mostrar clasificación en formato simple"""
        try:
            clasificaciones = {}
            for p in partidos:
                clase = p.get('clasificacion', 'Sin clasificar')
                clasificaciones[clase] = clasificaciones.get(clase, 0) + 1
            
            st.write("**Clasificación de Partidos:**")
            for clase, count in clasificaciones.items():
                st.write(f"- {clase}: {count} partidos")
        except Exception as e:
            st.write(f"Error mostrando clasificación: {e}")

    def mostrar_resumen_basico_seguro(self, portafolio, partidos, metricas):
        """Mostrar resumen básico cuando fallan los gráficos"""
        try:
            st.subheader("📊 Resumen Básico")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Quinielas", len(portafolio))
                cores = len([q for q in portafolio if q.get("tipo") == "Core"])
                st.metric("Quinielas Core", cores)
            
            with col2:
                satelites = len([q for q in portafolio if q.get("tipo") == "Satelite"])
                st.metric("Satélites", satelites)
                empates_prom = np.mean([q.get('empates', 0) for q in portafolio if 'empates' in q])
                st.metric("Empates Promedio", f"{empates_prom:.1f}")
            
            with col3:
                st.metric("Partidos", len(partidos))
                if metricas and 'distribucion_global' in metricas:
                    dist = metricas['distribucion_global']['porcentajes']
                    st.write(f"**Distribución Global:**")
                    st.write(f"L: {dist.get('L', 0):.1%}")
                    st.write(f"E: {dist.get('E', 0):.1%}")
                    st.write(f"V: {dist.get('V', 0):.1%}")

        except Exception as e:
            st.error(f"Error en resumen básico: {e}")
            st.write("Datos del portafolio no disponibles para mostrar")

    def mostrar_tabla_quinielas(self, portafolio, partidos):
        """Mostrar tabla interactiva de quinielas - CORREGIDA CON NOMBRES Y TIPO DE DATO PAR"""
        st.subheader("🎯 Quinielas Generadas")

        if not partidos:
            st.warning("No hay datos de partidos para mostrar en la tabla.")
            return

        partidos_info = [f"{p.get('home', 'N/A')} vs {p.get('away', 'N/A')}" for p in partidos]

        data_tabla = []
        for quiniela in portafolio:
            resultados_quiniela = quiniela.get("resultados", [])
            fila = {
                "ID": quiniela.get("id", "N/A"),
                "Tipo": quiniela.get("tipo", "N/A"),
                "Par": str(quiniela.get("par_id", "")),
                "Quiniela": "".join(resultados_quiniela),
                "Empates": resultados_quiniela.count("E"),
                "L": resultados_quiniela.count("L"),
                "E": resultados_quiniela.count("E"),
                "V": resultados_quiniela.count("V")
            }
            # Agregar partidos individuales
            for i, resultado in enumerate(resultados_quiniela):
                partido_desc = f"{partidos_info[i][:20]}..." if i < len(partidos_info) else f"P{i+1}"
                fila[f"P{i+1}"] = f"{resultado}"
            data_tabla.append(fila)

        if not data_tabla:
            st.info("No hay quinielas en el portafolio para mostrar.")
            return

        df_quinielas = pd.DataFrame(data_tabla)

        # Filtros
        col_filtro1, col_filtro2 = st.columns(2)
        with col_filtro1:
            tipos_disponibles = df_quinielas["Tipo"].unique()
            tipos_seleccionados = st.multiselect(
                "Filtrar por tipo:",
                options=tipos_disponibles,
                default=tipos_disponibles
            )
        with col_filtro2:
            empates_min = int(df_quinielas["Empates"].min())
            empates_max = int(df_quinielas["Empates"].max())
            if empates_min == empates_max:
                st.info(f"📊 Todas las quinielas tienen {empates_min} empates")
                rango_empates = (empates_min, empates_max)
            else:
                rango_empates = st.slider(
                    "Filtrar por número de empates:",
                    min_value=empates_min,
                    max_value=empates_max,
                    value=(empates_min, empates_max)
                )

        # Aplicar filtros
        df_filtrado = df_quinielas[
            (df_quinielas["Tipo"].isin(tipos_seleccionados)) &
            (df_quinielas["Empates"] >= rango_empates[0]) &
            (df_quinielas["Empates"] <= rango_empates[1])
        ]
        
        with st.expander("📋 Ver Partidos del Concurso"):
            partidos_df = pd.DataFrame([
                {"#": i+1, "Partido": f"{p.get('home', 'N/A')} vs {p.get('away', 'N/A')}", "Clasificación": p.get('clasificacion', 'N/A')}
                for i, p in enumerate(partidos)
            ])
            st.dataframe(partidos_df, use_container_width=True, hide_index=True)

        st.dataframe(df_filtrado, use_container_width=True, hide_index=True)
        st.caption(f"Mostrando {len(df_filtrado)} de {len(df_quinielas)} quinielas")

    def mostrar_opciones_descarga(self, resultado):
        """Mostrar opciones de descarga de archivos"""
        st.subheader("💾 Descargar Resultados")

        archivos = resultado.get("archivos_exportados", {})

        if archivos:
            st.success(f"✅ Se generaron {len(archivos)} archivos:")

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
            st.warning("No se encontraron archivos exportados")

    def mostrar_ventana_debug_ai(self):
        """NUEVA FUNCIÓN: Ventana de debug para mostrar interacciones con AI"""
        
        with st.expander("🔍 **Debug AI - Ver Comunicación con la IA**", expanded=False):
            
            # Toggle para activar/desactivar debug
            debug_ai_activo = st.toggle(
                "Mostrar respuestas detalladas de la IA", 
                value=st.session_state.get('debug_ai_activo', False),
                help="Activa esto para ver exactamente qué dice la IA en cada corrección"
            )
            st.session_state.debug_ai_activo = debug_ai_activo
            
            if debug_ai_activo:
                st.info("🤖 **Modo Debug AI Activado** - Las próximas respuestas de la IA se mostrarán aquí")
                
                # Mostrar historial de respuestas si existe
                if 'ai_debug_responses' in st.session_state and st.session_state.ai_debug_responses:
                    st.markdown("### 📝 Historial de Respuestas de la IA")
                    
                    for i, response in enumerate(st.session_state.ai_debug_responses):
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            
                            with col1:
                                timestamp = response.get('timestamp', 'N/A')
                                st.write(f"**{i+1}.** `{timestamp}`")
                                
                                quiniela_id = response.get('quiniela_id', 'N/A')
                                st.write(f"**ID:** {quiniela_id}")
                                
                                success = response.get('success', False)
                                if success:
                                    st.success("✅ Éxito")
                                else:
                                    st.error("❌ Falló")
                            
                            with col2:
                                problemas = response.get('problemas', [])
                                if problemas:
                                    st.write(f"**Problemas detectados:** {', '.join(problemas)}")
                                
                                # Mostrar prompt enviado
                                if st.button(f"Ver Prompt Enviado #{i+1}", key=f"show_prompt_{i}"):
                                    st.code(response.get('prompt', 'No disponible'), language='text')
                                
                                # Mostrar respuesta de la AI
                                if st.button(f"Ver Respuesta AI #{i+1}", key=f"show_response_{i}"):
                                    st.code(response.get('ai_response', 'No disponible'), language='json')
                                
                                # Mostrar resultado parseado
                                resultado_parseado = response.get('parsed_result', None)
                                if resultado_parseado:
                                    st.write(f"**Resultado:** {resultado_parseado}")
                                else:
                                    st.write("**Resultado:** No se pudo parsear")
                            
                            st.divider()
                    
                    # Botón para limpiar historial
                    if st.button("🗑️ Limpiar Historial de Debug"):
                        st.session_state.ai_debug_responses = []
                        st.rerun()
                        
                else:
                    st.write("No hay interacciones con la IA todavía. Ejecuta una optimización para ver los datos.")
            
            # Estadísticas de API usage
            if 'ai_usage_stats' in st.session_state:
                stats = st.session_state.ai_usage_stats
                
                st.markdown("### 📊 Estadísticas de Uso de la IA")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Llamadas", stats.get('total_calls', 0))
                with col2:
                    st.metric("Éxitos", stats.get('successful_calls', 0))
                with col3:
                    st.metric("Fallos", stats.get('failed_calls', 0))
                with col4:
                    costo_estimado = stats.get('total_calls', 0) * 0.002  # Estimación
                    st.metric("Costo Est. USD", f"${costo_estimado:.3f}")

    def tab_validacion(self):
        """Tab para mostrar detalles de validación - CON DEBUG AI"""
        st.header("📋 Validación del Portafolio")

        if 'resultado_optimizacion' not in st.session_state or not st.session_state.resultado_optimizacion.get("success"):
            st.info("🔄 Ejecuta la optimización primero")
            return

        resultado = st.session_state.resultado_optimizacion
        validacion = resultado["validacion"]

        es_valido = validacion.get("es_valido", False)
        if es_valido:
            st.success("✅ PORTAFOLIO VÁLIDO - Cumple todas las reglas obligatorias")
        else:
            st.error("❌ PORTAFOLIO INVÁLIDO - No cumple algunas reglas")

        # *** NUEVA VENTANA DE DEBUG AI ***
        self.mostrar_ventana_debug_ai()

        st.subheader("Detalle de Validaciones")

        validaciones = validacion.get("detalle_validaciones", {})
        descripciones = {
            "distribucion_global": "Distribución en rangos históricos (35-41% L, 25-33% E, 30-36% V)",
            "empates_individuales": "4-6 empates por quiniela",
            "concentracion_maxima": "≤70% concentración general, ≤60% en primeros 3 partidos",
            "arquitectura_core_satelites": "4 Core + 26 Satélites en 13 pares",
            "correlacion_jaccard": "Correlación Jaccard ≤ 0.57 entre pares de satélites",
            "distribucion_divisores": "Distribución equilibrada de resultados"
        }

        for regla, cumple in validaciones.items():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.success("✅ CUMPLE") if cumple else st.error("❌ FALLA")
                with col2:
                    st.write(f"**{regla.replace('_', ' ').title()}**")
                    st.caption(descripciones.get(regla, "Regla sin descripción."))

        if "resumen" in validacion:
            with st.expander("Ver Resumen Completo en Texto"):
                st.text(validacion["resumen"])

        if "metricas" in validacion:
            with st.expander("Ver Métricas Detalladas (JSON)"):
                st.json(validacion["metricas"])
        
        ai_disponible = "OPENAI_API_KEY" in os.environ or ("OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]) or st.session_state.get('openai_api_key')
        if ai_disponible:
            st.markdown("---")
            st.subheader("🤖 Asistente AI")
            
            if resultado.get("ai_utilizada"):
                st.success("✅ AI fue utilizada automáticamente durante la optimización")
            
            if not es_valido:
                st.error("❌ El portafolio actual tiene errores de validación")
                if st.button("🔧 Re-optimizar con AI", type="primary", use_container_width=True):
                    st.session_state.ejecutar_con_ai = True
                    st.rerun()

def main():
    """Función principal para ejecutar la app"""
    app = ProgolStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()