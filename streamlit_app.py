# streamlit_app.py
"""
Interfaz gráfica Streamlit para Progol Optimizer - VERSIÓN CORREGIDA
Permite cargar datos, ejecutar optimización y ver resultados
CORRECCIONES APLICADAS:
- CSV del usuario ahora se procesa correctamente 
- Tabla muestra nombres de equipos por partido
- Slider corregido para mismo número de empates
- Calibración global aplicada
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
    Aplicación Streamlit para el Progol Optimizer - VERSIÓN CORREGIDA
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

        # Contenido principal
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
                with st.spinner("Generando datos de ejemplo..."):
                    try:
                        from data.loader import DataLoader
                        loader = DataLoader()
                        datos_ejemplo = loader._generar_datos_ejemplo()
                        st.session_state.datos_partidos = datos_ejemplo
                        st.session_state.archivo_origen = "datos_ejemplo"
                        st.success(f"✅ Generados {len(datos_ejemplo)} partidos de ejemplo")

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
                            prob_local, prob_empate, prob_visitante = loader._generar_probabilidades_realistas()

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
                if archivo_origen == 'datos_ejemplo':
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
                    finales = sum(1 for p in datos if p.get('es_final'))
                    st.metric("Finales/Especiales", finales)

            else:
                st.info("👆 Carga datos usando una de las opciones de arriba")

    def tab_optimizacion(self):
        """Tab para ejecutar la optimización"""
        st.header("🎯 Optimización del Portafolio")

        if 'datos_partidos' not in st.session_state:
            st.warning("⚠️ Primero carga los datos en la pestaña 'Datos & Configuración'")
            return

        # Mostrar qué datos se van a usar
        archivo_origen = st.session_state.get('archivo_origen', 'datos_ejemplo')
        if archivo_origen == 'datos_ejemplo':
            st.info("📊 Se optimizará usando **datos de ejemplo generados**")
        else:
            st.success(f"📄 Se optimizará usando **datos de: {archivo_origen}**")

        # Botón principal de optimización
        if st.button("🚀 Ejecutar Optimización Completa", type="primary", use_container_width=True):
            self.ejecutar_optimizacion()

        # Mostrar progreso si está ejecutándose
        if 'optimizacion_ejecutando' in st.session_state and st.session_state.optimizacion_ejecutando:
            self.mostrar_progreso_optimizacion()

    def ejecutar_optimizacion(self):
        """Ejecutar el proceso completo de optimización"""
        st.session_state.optimizacion_ejecutando = True

        progress_bar = st.progress(0, text="Inicializando...")

        try:
            # Inicializar optimizador
            progress_bar.progress(10, text="🔧 Inicializando optimizador...")
            optimizer = ProgolOptimizer()

            def update_progress(progress_value, text_value):
                display_progress = 30 + int(progress_value * 60)
                progress_bar.progress(display_progress, text=text_value)

            # Ejecutar optimización
            progress_bar.progress(30, text="⚙️ Ejecutando optimización GRASP-Annealing...")

            resultado = self.ejecutar_optimizacion_directo(optimizer, progress_callback=update_progress)

            progress_bar.progress(100, text="✅ Optimización completada!")

            # Guardar resultados
            st.session_state.resultado_optimizacion = resultado
            st.session_state.optimizacion_ejecutando = False

            # Mostrar resumen inmediato
            self.mostrar_resumen_resultado(resultado)

        except Exception as e:
            st.error(f"❌ Error en optimización: {e}")
            st.session_state.optimizacion_ejecutando = False

            if st.session_state.debug_mode:
                st.exception(e)

    def ejecutar_optimizacion_directo(self, optimizer, progress_callback=None):
        """Ejecutar optimización usando datos ya cargados"""
        # Obtener datos de session_state
        datos_partidos = st.session_state.datos_partidos

        # PASO 1: Validación
        es_valido, errores = optimizer.data_validator.validar_estructura(datos_partidos)
        if not es_valido:
            raise ValueError(f"Datos inválidos: {errores}")

        # PASO 2: Clasificación y calibración - CORREGIDO CON CALIBRACIÓN GLOBAL
        # NUEVO: Usar calibración global en lugar de individual
        partidos_calibrados = optimizer.calibrator.calibrar_concurso_completo(datos_partidos)

        # Aplicar clasificación después de calibración
        partidos_clasificados = []
        for i, partido_calibrado in enumerate(partidos_calibrados):
            clasificacion = optimizer.classifier.clasificar_partido(partido_calibrado)
            
            partido_final = {
                **partido_calibrado,
                "id": i,
                "clasificacion": clasificacion
            }
            partidos_clasificados.append(partido_final)

        # PASO 3: Generar Core
        quinielas_core = optimizer.core_generator.generar_quinielas_core(partidos_clasificados)

        # PASO 4: Generar Satélites
        quinielas_satelites = optimizer.satellite_generator.generar_pares_satelites(
            partidos_clasificados, 26
        )

        # PASO 5: Optimizar
        portafolio_inicial = quinielas_core + quinielas_satelites
        portafolio_optimizado = optimizer.optimizer.optimizar_portafolio_grasp_annealing(
            portafolio_inicial, partidos_clasificados, progress_callback=progress_callback
        )

        # PASO 6: Validar
        resultado_validacion = optimizer.portfolio_validator.validar_portafolio_completo(
            portafolio_optimizado
        )

        # PASO 7: Exportar
        archivos_exportados = optimizer.exporter.exportar_portafolio_completo(
            portafolio_optimizado,
            partidos_clasificados,
            resultado_validacion["metricas"],
            st.session_state.concurso_id
        )

        return {
            "portafolio": portafolio_optimizado,
            "partidos": partidos_clasificados,
            "validacion": resultado_validacion,
            "metricas": resultado_validacion["metricas"],
            "archivos_exportados": archivos_exportados,
            "concurso_id": st.session_state.concurso_id
        }

    def mostrar_resumen_resultado(self, resultado):
        """Mostrar resumen inmediato del resultado"""
        st.success("🎉 ¡Optimización completada exitosamente!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Quinielas Generadas", len(resultado["portafolio"]))

        with col2:
            es_valido = resultado["validacion"]["es_valido"]
            st.metric("Validación", "✅ VÁLIDO" if es_valido else "❌ INVÁLIDO")

        with col3:
            cores = len([q for q in resultado["portafolio"] if q["tipo"] == "Core"])
            st.metric("Quinielas Core", cores)

        with col4:
            satelites = len([q for q in resultado["portafolio"] if q["tipo"] == "Satelite"])
            st.metric("Satélites", satelites)

    def mostrar_progreso_optimizacion(self):
        """Mostrar progreso durante la optimización"""
        st.info("⏳ Optimización en progreso...")

    def tab_resultados(self):
        """Tab para mostrar resultados de la optimización"""
        st.header("📈 Resultados de la Optimización")

        if 'resultado_optimizacion' not in st.session_state:
            st.info("🔄 Ejecuta la optimización primero en la pestaña 'Optimización'")
            return

        resultado = st.session_state.resultado_optimizacion
        portafolio = resultado["portafolio"]
        partidos = resultado["partidos"]
        metricas = resultado["metricas"]

        # Resumen ejecutivo
        self.mostrar_resumen_ejecutivo(portafolio, metricas)

        # Visualizaciones
        self.crear_visualizaciones(portafolio, partidos, metricas)

        # Tabla de quinielas - CORREGIDA
        self.mostrar_tabla_quinielas(portafolio, partidos)

        # Descarga de archivos
        self.mostrar_opciones_descarga(resultado)

    def mostrar_resumen_ejecutivo(self, portafolio, metricas):
        """Mostrar resumen ejecutivo de resultados"""
        st.subheader("📋 Resumen Ejecutivo")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Quinielas", len(portafolio))

        with col2:
            empates_prom = metricas["empates_estadisticas"]["promedio"]
            st.metric("Empates Promedio", f"{empates_prom:.1f}")

        with col3:
            dist = metricas["distribucion_global"]["porcentajes"]
            st.metric("% Locales", f"{dist['L']:.1%}")

        with col4:
            st.metric("% Empates", f"{dist['E']:.1%}")

        with col5:
            st.metric("% Visitantes", f"{dist['V']:.1%}")

        # Comparación con rangos históricos
        st.markdown("**Comparación con Rangos Históricos:**")

        rangos = PROGOL_CONFIG["RANGOS_HISTORICOS"]

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            actual_l = dist['L']
            min_l, max_l = rangos['L']
            cumple_l = min_l <= actual_l <= max_l
            color_l = "normal" if cumple_l else "inverse"
            st.metric(
                "Locales",
                f"{actual_l:.1%}",
                f"Rango: {min_l:.0%}-{max_l:.0%}",
                delta_color=color_l
            )

        with col_b:
            actual_e = dist['E']
            min_e, max_e = rangos['E']
            cumple_e = min_e <= actual_e <= max_e
            color_e = "normal" if cumple_e else "inverse"
            st.metric(
                "Empates",
                f"{actual_e:.1%}",
                f"Rango: {min_e:.0%}-{max_e:.0%}",
                delta_color=color_e
            )

        with col_c:
            actual_v = dist['V']
            min_v, max_v = rangos['V']
            cumple_v = min_v <= actual_v <= max_v
            color_v = "normal" if cumple_v else "inverse"
            st.metric(
                "Visitantes",
                f"{actual_v:.1%}",
                f"Rango: {min_v:.0%}-{max_v:.0%}",
                delta_color=color_v
            )

    def crear_visualizaciones(self, portafolio, partidos, metricas):
        """Crear visualizaciones de los resultados"""
        st.subheader("📊 Visualizaciones")

        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de distribución por tipo de quiniela
            fig_tipos = self.grafico_distribucion_tipos(portafolio)
            st.plotly_chart(fig_tipos, use_container_width=True)

        with col2:
            # Gráfico de empates por quiniela
            fig_empates = self.grafico_empates_distribucion(portafolio)
            st.plotly_chart(fig_empates, use_container_width=True)

        # Gráfico de clasificación de partidos
        fig_clasificacion = self.grafico_clasificacion_partidos(partidos)
        st.plotly_chart(fig_clasificacion, use_container_width=True)

        # Mapa de calor de correlaciones (para satélites)
        satelites = [q for q in portafolio if q["tipo"] == "Satelite"]
        if len(satelites) >= 4:
            fig_correlacion = self.grafico_correlaciones_satelites(satelites)
            st.plotly_chart(fig_correlacion, use_container_width=True)

    def grafico_distribucion_tipos(self, portafolio):
        """Gráfico de distribución L/E/V por tipo de quiniela"""
        # Agrupar por tipo
        datos_tipo = {}
        for quiniela in portafolio:
            tipo = quiniela["tipo"]
            if tipo not in datos_tipo:
                datos_tipo[tipo] = {"L": 0, "E": 0, "V": 0, "count": 0}

            for resultado in quiniela["resultados"]:
                datos_tipo[tipo][resultado] += 1
            datos_tipo[tipo]["count"] += 1

        # Convertir a porcentajes
        tipos = []
        locales = []
        empates = []
        visitantes = []

        for tipo, datos in datos_tipo.items():
            total = datos["count"] * 14  # 14 partidos por quiniela
            tipos.append(f"{tipo}\n({datos['count']} quinielas)")
            locales.append(datos["L"] / total * 100)
            empates.append(datos["E"] / total * 100)
            visitantes.append(datos["V"] / total * 100)

        fig = go.Figure(data=[
            go.Bar(name='Locales', x=tipos, y=locales, marker_color='lightblue'),
            go.Bar(name='Empates', x=tipos, y=empates, marker_color='lightgray'),
            go.Bar(name='Visitantes', x=tipos, y=visitantes, marker_color='lightcoral')
        ])

        fig.update_layout(
            title="Distribución L/E/V por Tipo de Quiniela",
            barmode='stack',
            yaxis_title="Porcentaje (%)",
            xaxis_title="Tipo de Quiniela"
        )

        return fig

    def grafico_empates_distribucion(self, portafolio):
        """Histograma de distribución de empates"""
        empates = [quiniela["resultados"].count("E") for quiniela in portafolio]

        fig = px.histogram(
            x=empates,
            nbins=7,
            title="Distribución de Empates por Quiniela",
            labels={"x": "Número de Empates", "y": "Cantidad de Quinielas"},
            color_discrete_sequence=['skyblue']
        )

        # Agregar líneas de límites
        fig.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="Mín: 4")
        fig.add_vline(x=6, line_dash="dash", line_color="red", annotation_text="Máx: 6")

        return fig

    def grafico_clasificacion_partidos(self, partidos):
        """Gráfico de clasificación de partidos"""
        clasificaciones = {}
        for partido in partidos:
            clase = partido.get("clasificacion", "Sin clasificar")
            clasificaciones[clase] = clasificaciones.get(clase, 0) + 1

        fig = px.pie(
            values=list(clasificaciones.values()),
            names=list(clasificaciones.keys()),
            title="Clasificación de Partidos según Taxonomía"
        )

        return fig

    def grafico_correlaciones_satelites(self, satelites):
        """Mapa de calor de correlaciones entre satélites"""
        # Tomar muestra para visualización
        muestra = satelites[:10] if len(satelites) > 10 else satelites

        # Calcular matriz de correlación
        n = len(muestra)
        matriz_corr = []
        nombres = []

        for i, sat_i in enumerate(muestra):
            fila_corr = []
            if i == 0:
                nombres = [sat["id"] for sat in muestra]

            for sat_j in muestra:
                # Calcular correlación Jaccard
                resultados_i = sat_i["resultados"]
                resultados_j = sat_j["resultados"]

                coincidencias = sum(1 for a, b in zip(resultados_i, resultados_j) if a == b)
                correlacion = coincidencias / len(resultados_i)
                fila_corr.append(correlacion)

            matriz_corr.append(fila_corr)

        fig = px.imshow(
            matriz_corr,
            x=nombres,
            y=nombres,
            title="Correlaciones entre Satélites (Jaccard)",
            color_continuous_scale="RdYlBu_r"
        )

        return fig

    def mostrar_tabla_quinielas(self, portafolio, partidos):
        """Mostrar tabla interactiva de quinielas - CORREGIDA CON NOMBRES Y TIPO DE DATO PAR"""
        st.subheader("🎯 Quinielas Generadas")

        partidos_info = [f"{p['home']} vs {p['away']}" for p in partidos]

        data_tabla = []
        for quiniela in portafolio:
            fila = {
                "ID": quiniela["id"],
                "Tipo": quiniela["tipo"],
                # CORRECCIÓN CRÍTICA: Convertir el par_id a string para evitar error de tipo
                "Par": str(quiniela.get("par_id", "")),
                "Quiniela": "".join(quiniela["resultados"]),
                "Empates": quiniela["resultados"].count("E"),
                "L": quiniela["resultados"].count("L"),
                "E": quiniela["resultados"].count("E"),
                "V": quiniela["resultados"].count("V")
            }
            # Agregar partidos individuales
            for i, resultado in enumerate(quiniela["resultados"]):
                partido_desc = f"{partidos_info[i][:15]}..." if i < len(partidos_info) else ""
                fila[f"P{i+1}"] = f"{resultado} ({partido_desc})"
            data_tabla.append(fila)

        df_quinielas = pd.DataFrame(data_tabla)

        # Filtros
        col_filtro1, col_filtro2 = st.columns(2)
        with col_filtro1:
            tipos_seleccionados = st.multiselect(
                "Filtrar por tipo:",
                options=df_quinielas["Tipo"].unique(),
                default=df_quinielas["Tipo"].unique()
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
                {"#": i+1, "Partido": f"{p['home']} vs {p['away']}", "Clasificación": p.get('clasificacion', 'N/A')}
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

    def tab_validacion(self):
        """Tab para mostrar detalles de validación"""
        st.header("📋 Validación del Portafolio")

        if 'resultado_optimizacion' not in st.session_state:
            st.info("🔄 Ejecuta la optimización primero")
            return

        resultado = st.session_state.resultado_optimizacion
        validacion = resultado["validacion"]

        # Estado general
        es_valido = validacion["es_valido"]
        if es_valido:
            st.success("✅ PORTAFOLIO VÁLIDO - Cumple todas las reglas obligatorias")
        else:
            st.error("❌ PORTAFOLIO INVÁLIDO - No cumple algunas reglas")

        # Detalle de cada validación
        st.subheader("Detalle de Validaciones")

        validaciones = validacion["detalle_validaciones"]
        descripciones = {
            "distribucion_global": "Distribución en rangos históricos (35-41% L, 25-33% E, 30-36% V)",
            "empates_individuales": "4-6 empates por quiniela",
            "concentracion_maxima": "≤70% concentración general, ≤60% en primeros 3 partidos",
            "arquitectura_core_satelites": "4 Core + 26 Satélites en 13 pares",
            "correlacion_jaccard": "Correlación Jaccard ≤ 0.57 entre pares de satélites",
            "distribucion_divisores": "Distribución equilibrada de resultados"
        }

        for regla, cumple in validaciones.items():
            col1, col2 = st.columns([1, 4])

            with col1:
                if cumple:
                    st.success("✅ CUMPLE")
                else:
                    st.error("❌ FALLA")

            with col2:
                descripcion = descripciones.get(regla, regla)
                st.write(f"**{regla.replace('_', ' ').title()}**: {descripcion}")

        # Resumen textual
        if "resumen" in validacion:
            st.subheader("Resumen Completo")
            st.text(validacion["resumen"])

        # Métricas detalladas
        st.subheader("Métricas Detalladas")
        st.json(validacion["metricas"])

def main():
    """Función principal para ejecutar la app"""
    app = ProgolStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()