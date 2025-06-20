# streamlit_app.py
"""
Progol Optimizer - Flujo Paso a Paso Simplificado
Interfaz que permite debuggear cada componente por separado
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import sys
import os
from pathlib import Path
import logging
import tempfile
import numpy as np

# Mantener todos los imports existentes
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from main import ProgolOptimizer
    from data.loader import DataLoader
    from data.validator import DataValidator
    from models.classifier import PartidoClassifier
    from models.calibrator import BayesianCalibrator
    from portfolio.core_generator import CoreGenerator
    from validation.portfolio_validator import PortfolioValidator
    from config.constants import PROGOL_CONFIG
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()

class StepByStepProgolApp:
    """
    Aplicación simplificada que expone cada paso del proceso por separado
    """
    
    def __init__(self):
        self.configurar_pagina()
        self.inicializar_componentes()
    
    def configurar_pagina(self):
        st.set_page_config(
            page_title="Progol Optimizer - Debug Paso a Paso",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def inicializar_componentes(self):
        """Inicializar solo los componentes necesarios"""
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.classifier = PartidoClassifier()
        self.calibrator = BayesianCalibrator()
        self.core_generator = CoreGenerator()
        self.validator = PortfolioValidator()
    
    def run(self):
        """Ejecutar la aplicación principal"""
        st.title("⚽ Progol Optimizer - Debug Paso a Paso")
        st.markdown("### Flujo simplificado para identificar problemas")
        
        # Sidebar con estado del flujo
        self.mostrar_sidebar_estado()
        
        # Tabs principales - 4 pasos
        tabs = st.tabs([
            "📊 PASO 1: Datos",
            "🏷️ PASO 2: Clasificación", 
            "🎯 PASO 3: Generación",
            "✅ PASO 4: Validación"
        ])
        
        with tabs[0]:
            self.paso_1_datos()
        
        with tabs[1]:
            self.paso_2_clasificacion()
            
        with tabs[2]:
            self.paso_3_generacion()
            
        with tabs[3]:
            self.paso_4_validacion()
    
    def mostrar_sidebar_estado(self):
        """Mostrar estado actual del flujo en sidebar"""
        with st.sidebar:
            st.header("🔍 Estado del Flujo")
            
            # Estado de cada paso
            paso1_ok = 'datos_validados' in st.session_state and st.session_state.datos_validados
            paso2_ok = 'partidos_clasificados' in st.session_state
            paso3_ok = 'quinielas_generadas' in st.session_state
            paso4_ok = 'validacion_completa' in st.session_state
            
            st.write("📊 Paso 1 - Datos:", "✅" if paso1_ok else "⏳")
            st.write("🏷️ Paso 2 - Clasificación:", "✅" if paso2_ok else "⏳")
            st.write("🎯 Paso 3 - Generación:", "✅" if paso3_ok else "⏳")
            st.write("✅ Paso 4 - Validación:", "✅" if paso4_ok else "⏳")
            
            st.markdown("---")
            
            # Botón de reset
            if st.button("🔄 Reiniciar Todo", type="secondary"):
                for key in list(st.session_state.keys()):
                    if key.startswith(('datos_', 'partidos_', 'quinielas_', 'validacion_')):
                        del st.session_state[key]
                st.rerun()
            
            # Mostrar configuración
            with st.expander("⚙️ Ver Configuración"):
                st.json({
                    "EMPATES_MIN": PROGOL_CONFIG["EMPATES_MIN"],
                    "EMPATES_MAX": PROGOL_CONFIG["EMPATES_MAX"],
                    "CONCENTRACION_MAX": PROGOL_CONFIG["CONCENTRACION_MAX_GENERAL"],
                    "RANGOS_L": PROGOL_CONFIG["RANGOS_HISTORICOS"]["L"],
                    "RANGOS_E": PROGOL_CONFIG["RANGOS_HISTORICOS"]["E"],
                    "RANGOS_V": PROGOL_CONFIG["RANGOS_HISTORICOS"]["V"]
                })
    
    def paso_1_datos(self):
        """PASO 1: Carga y validación de datos solamente"""
        st.header("📊 PASO 1: Carga y Validación de Datos")
        st.markdown("**Objetivo**: Asegurar que tenemos exactamente 14 partidos con probabilidades válidas")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Cargar Datos")
            
            # Opción 1: Datos de ejemplo
            if st.button("🎲 Generar Datos de Ejemplo", type="primary"):
                with st.spinner("Generando datos de ejemplo..."):
                    try:
                        datos = self.data_loader._generar_datos_ejemplo()
                        st.session_state.datos_raw = datos
                        st.session_state.origen_datos = "Datos de Ejemplo"
                        st.success(f"✅ Generados {len(datos)} partidos")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            # Opción 2: Subir CSV
            st.markdown("**O subir CSV:**")
            archivo_csv = st.file_uploader(
                "Archivo CSV",
                type=['csv'],
                help="Debe tener: home, away, prob_local, prob_empate, prob_visitante"
            )
            
            if archivo_csv:
                try:
                    df = pd.read_csv(archivo_csv)
                    if len(df) != 14:
                        st.error(f"❌ Se requieren 14 partidos, archivo tiene {len(df)}")
                    else:
                        # Convertir a formato interno
                        datos = self._convertir_csv_a_datos(df)
                        st.session_state.datos_raw = datos
                        st.session_state.origen_datos = archivo_csv.name
                        st.success(f"✅ Cargado: {archivo_csv.name}")
                except Exception as e:
                    st.error(f"❌ Error leyendo CSV: {e}")
        
        with col2:
            if 'datos_raw' in st.session_state:
                st.subheader("Vista Previa")
                datos = st.session_state.datos_raw
                
                # Mostrar origen
                st.info(f"📄 Origen: {st.session_state.get('origen_datos', 'Desconocido')}")
                
                # Tabla de vista previa
                preview_df = pd.DataFrame([
                    {
                        "#": i+1,
                        "Local": p['home'][:15],
                        "Visitante": p['away'][:15],
                        "P(L)": f"{p['prob_local']:.3f}",
                        "P(E)": f"{p['prob_empate']:.3f}",
                        "P(V)": f"{p['prob_visitante']:.3f}",
                        "Suma": f"{p['prob_local'] + p['prob_empate'] + p['prob_visitante']:.3f}"
                    }
                    for i, p in enumerate(datos)
                ])
                
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        # Validación de datos
        if 'datos_raw' in st.session_state:
            st.markdown("---")
            st.subheader("🔍 Validación de Datos")
            
            if st.button("▶️ Validar Datos", type="primary"):
                datos = st.session_state.datos_raw
                
                with st.spinner("Validando datos..."):
                    es_valido, errores = self.data_validator.validar_estructura(datos)
                    
                    st.session_state.datos_validados = es_valido
                    st.session_state.errores_validacion = errores
                
                # Mostrar resultados
                if es_valido:
                    st.success("✅ **DATOS VÁLIDOS** - Listos para el Paso 2")
                    
                    # Estadísticas
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Partidos", len(datos))
                    with col_b:
                        ligas = len(set(p['liga'] for p in datos))
                        st.metric("Ligas", ligas)
                    with col_c:
                        suma_L = sum(p['prob_local'] for p in datos)
                        st.metric("Suma P(L)", f"{suma_L:.2f}")
                    
                    # Guardar para siguientes pasos
                    st.session_state.datos_paso1 = datos
                    
                else:
                    st.error("❌ **DATOS INVÁLIDOS**")
                    
                    # Mostrar errores específicos
                    for error in errores:
                        st.error(f"• {error}")
                    
                    # Opción para corregir manualmente
                    with st.expander("🔧 Intentar Corrección Manual"):
                        st.warning("Los siguientes controles permiten ajustar datos manualmente")
                        
                        # Aquí podrías agregar controles para ajustar probabilidades manualmente
                        st.info("💡 Funcionalidad de corrección manual pendiente de implementar")
            
            # Mostrar estado de validación
            if 'datos_validados' in st.session_state:
                if st.session_state.datos_validados:
                    st.success("✅ Datos validados - Procede al **PASO 2: Clasificación**")
                else:
                    st.error("❌ Datos no válidos - Corrige los errores arriba")
    
    def paso_2_clasificacion(self):
        """PASO 2: Clasificación de partidos solamente"""
        st.header("🏷️ PASO 2: Clasificación de Partidos")
        st.markdown("**Objetivo**: Clasificar cada partido como Ancla/Divisor/TendenciaEmpate/Neutro")
        
        # Verificar prerequisitos
        if 'datos_paso1' not in st.session_state:
            st.warning("⚠️ Primero completa el **PASO 1: Datos**")
            return
        
        if not st.session_state.get('datos_validados', False):
            st.error("❌ Los datos del Paso 1 no están validados")
            return
        
        st.success("✅ Prerequisitos cumplidos")
        
        # NUEVA OPCIÓN: Modo de clasificación
        modo_clasificacion = st.radio(
            "🔧 Modo de Clasificación:",
            ["Sin Calibración (Datos Originales)", "Con Calibración Bayesiana"],
            help="Sin calibración usa probabilidades originales, con calibración las ajusta"
        )
        
        # NUEVA OPCIÓN: Ajustar umbrales temporalmente
        st.markdown("---")
        st.subheader("⚙️ Ajustes Temporales para Debug")
        
        usar_umbrales_debug = st.checkbox(
            "🔧 Usar umbrales más permisivos para debug",
            help="Reduce umbral de Ancla de 60% a 40% temporalmente"
        )
        
        if usar_umbrales_debug:
            st.warning("⚠️ Modo Debug: Umbral Ancla = 40% (en lugar de 60%)")
        
        # Botón para ejecutar clasificación
        if st.button("▶️ Ejecutar Clasificación", type="primary"):
            with st.spinner("Clasificando partidos..."):
                try:
                    # Tomar datos del paso anterior
                    datos = st.session_state.datos_paso1
                    
                    if modo_clasificacion == "Con Calibración Bayesiana":
                        # Aplicar calibración bayesiana primero
                        st.info("🔄 Aplicando calibración bayesiana...")
                        partidos_calibrados = self.calibrator.calibrar_concurso_completo(datos)
                    else:
                        # Usar datos originales sin calibración
                        st.info("🔄 Usando probabilidades originales...")
                        partidos_calibrados = datos
                    
                    # Clasificar cada partido
                    st.info("🔄 Clasificando partidos...")
                    
                    # Ajustar umbrales temporalmente si está en modo debug
                    if usar_umbrales_debug:
                        # Guardar umbrales originales
                        umbrales_originales = self.classifier.umbrales.copy()
                        # Usar umbrales más permisivos
                        self.classifier.umbrales["ancla_prob_min"] = 0.40  # 40% en lugar de 60%
                        st.info("🔧 Usando umbrales debug: Ancla = 40%")
                    
                    partidos_clasificados = []
                    
                    for i, partido in enumerate(partidos_calibrados):
                        clasificacion = self.classifier.clasificar_partido(partido)
                        partido_final = {
                            **partido,
                            "id": i,
                            "clasificacion": clasificacion
                        }
                        partidos_clasificados.append(partido_final)
                    
                    # Restaurar umbrales originales si fueron modificados
                    if usar_umbrales_debug:
                        self.classifier.umbrales = umbrales_originales
                        st.info("🔄 Umbrales restaurados")
                    
                    # Guardar resultados
                    st.session_state.partidos_clasificados = partidos_clasificados
                    st.session_state.estadisticas_clasificacion = self.classifier.obtener_estadisticas_clasificacion(partidos_clasificados)
                    st.session_state.modo_usado = modo_clasificacion
                    st.session_state.umbrales_debug_usados = usar_umbrales_debug
                    
                    st.success("✅ Clasificación completada")
                    
                except Exception as e:
                    st.error(f"❌ Error en clasificación: {e}")
                    st.exception(e)
        
        # Mostrar resultados de clasificación
        if 'partidos_clasificados' in st.session_state:
            st.markdown("---")
            st.subheader("📋 Resultados de Clasificación")
            
            # Mostrar modo usado
            modo_usado = st.session_state.get('modo_usado', 'Desconocido')
            umbrales_debug = st.session_state.get('umbrales_debug_usados', False)
            
            if modo_usado == "Sin Calibración (Datos Originales)":
                st.info("📊 **Modo usado**: Probabilidades originales (sin calibración)")
            else:
                st.info("📊 **Modo usado**: Con calibración bayesiana aplicada")
            
            if umbrales_debug:
                st.warning("🔧 **Umbrales Debug**: Ancla = 40% (en lugar de 60%)")
            else:
                st.info("⚙️ **Umbrales Normales**: Ancla = 60%")
            
            partidos = st.session_state.partidos_clasificados
            stats = st.session_state.estadisticas_clasificacion
            
            # Estadísticas por tipo
            col1, col2, col3, col4 = st.columns(4)
            
            distribución = stats["distribución"]
            with col1:
                st.metric("Anclas", distribución.get("Ancla", 0), help="Partidos >60% probabilidad")
            with col2:
                st.metric("Divisores", distribución.get("Divisor", 0), help="Partidos 40-60% probabilidad")
            with col3:
                st.metric("Tend. Empate", distribución.get("TendenciaEmpate", 0), help="Partidos con tendencia al empate")
            with col4:
                st.metric("Neutros", distribución.get("Neutro", 0), help="Resto de partidos")
            
            # Tabla detallada
            st.subheader("🔍 Detalle por Partido")
            
            detalle_df = pd.DataFrame([
                {
                    "#": i+1,
                    "Partido": f"{p['home']} vs {p['away']}",
                    "P(L)": f"{p['prob_local']:.3f}",
                    "P(E)": f"{p['prob_empate']:.3f}",
                    "P(V)": f"{p['prob_visitante']:.3f}",
                    "Clasificación": p['clasificacion'],
                    "Max Prob": f"{max(p['prob_local'], p['prob_empate'], p['prob_visitante']):.3f}"
                }
                for i, p in enumerate(partidos)
            ])
            
            # Colorear por clasificación
            def color_clasificacion(val):
                colors = {
                    "Ancla": "background-color: #90EE90",
                    "Divisor": "background-color: #FFE4B5", 
                    "TendenciaEmpate": "background-color: #87CEEB",
                    "Neutro": "background-color: #F0F0F0"
                }
                return colors.get(val, "")
            
            styled_df = detalle_df.style.applymap(color_clasificacion, subset=['Clasificación'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Verificar si hay suficientes anclas
            num_anclas = distribución.get("Ancla", 0)
            umbrales_debug = st.session_state.get('umbrales_debug_usados', False)
            umbral_texto = "40%" if umbrales_debug else "60%"
            
            if num_anclas == 0:
                st.error(f"❌ **PROBLEMA CRÍTICO**: No hay partidos Ancla (>{umbral_texto}). Las quinielas Core no se podrán generar correctamente.")
                if not umbrales_debug:
                    st.info("💡 **Solución**: Activa 'umbrales más permisivos para debug' arriba")
                else:
                    st.warning("💡 Los datos de ejemplo necesitan probabilidades más extremas")
            elif num_anclas < 2:
                st.warning(f"⚠️ Solo {num_anclas} Ancla detectada (>{umbral_texto}). Se recomienda al menos 2-3 para estabilidad")
            else:
                st.success(f"✅ {num_anclas} Anclas detectadas (>{umbral_texto}) - Suficiente para generar Core estables")
            
            # Botón para continuar al paso 3
            if num_anclas > 0:
                st.success("✅ Clasificación válida - Procede al **PASO 3: Generación**")
            else:
                st.error("❌ Clasificación problemática - Revisa los datos o ajusta umbrales")
    
    def paso_3_generacion(self):
        """PASO 3: Generación de quinielas Core solamente"""
        st.header("🎯 PASO 3: Generación de Quinielas Core")
        st.markdown("**Objetivo**: Generar solo 4 quinielas Core (sin satélites por ahora)")
        
        # Verificar prerequisitos
        if 'partidos_clasificados' not in st.session_state:
            st.warning("⚠️ Primero completa el **PASO 2: Clasificación**")
            return
        
        stats = st.session_state.estadisticas_clasificacion
        if stats["distribución"].get("Ancla", 0) == 0:
            st.error("❌ No hay partidos Ancla para generar quinielas Core")
            return
        
        st.success("✅ Prerequisitos cumplidos")
        
        # Mostrar información de partidos Ancla
        partidos = st.session_state.partidos_clasificados
        anclas = [p for p in partidos if p['clasificacion'] == 'Ancla']
        
        with st.expander(f"📌 Partidos Ancla Detectados ({len(anclas)})"):
            for ancla in anclas:
                max_prob = max(ancla['prob_local'], ancla['prob_empate'], ancla['prob_visitante'])
                resultado_ancla = 'L' if ancla['prob_local'] == max_prob else ('E' if ancla['prob_empate'] == max_prob else 'V')
                st.write(f"• **{ancla['home']} vs {ancla['away']}** → {resultado_ancla} ({max_prob:.3f})")
        
        # Botón para generar Core
        if st.button("▶️ Generar 4 Quinielas Core", type="primary"):
            with st.spinner("Generando quinielas Core..."):
                try:
                    # Usar el generador de Core existente
                    quinielas_core = self.core_generator.generar_quinielas_core(partidos)
                    
                    # Guardar resultados
                    st.session_state.quinielas_generadas = quinielas_core
                    st.session_state.tipo_generacion = "Solo Core"
                    
                    st.success(f"✅ Generadas {len(quinielas_core)} quinielas Core")
                    
                except Exception as e:
                    st.error(f"❌ Error generando Core: {e}")
                    st.exception(e)
        
        # Mostrar quinielas generadas
        if 'quinielas_generadas' in st.session_state:
            st.markdown("---")
            st.subheader("📋 Quinielas Core Generadas")
            
            quinielas = st.session_state.quinielas_generadas
            
            # Estadísticas rápidas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quinielas", len(quinielas))
            with col2:
                empates_promedio = sum(q['empates'] for q in quinielas) / len(quinielas)
                st.metric("Empates Promedio", f"{empates_promedio:.1f}")
            with col3:
                # Verificar si todas tienen los mismos resultados en anclas
                anclas_indices = [i for i, p in enumerate(partidos) if p['clasificacion'] == 'Ancla']
                anclas_consistentes = True
                if anclas_indices and len(quinielas) > 1:
                    primera_anclas = [quinielas[0]['resultados'][i] for i in anclas_indices]
                    for q in quinielas[1:]:
                        estas_anclas = [q['resultados'][i] for i in anclas_indices]
                        if primera_anclas != estas_anclas:
                            anclas_consistentes = False
                            break
                
                st.metric("Anclas Consistentes", "✅" if anclas_consistentes else "❌")
            
            # Tabla de quinielas
            quinielas_df = pd.DataFrame([
                {
                    "ID": q['id'],
                    "Quiniela": "".join(q['resultados']),
                    "Empates": q['empates'],
                    "L": q['distribución']['L'],
                    "E": q['distribución']['E'], 
                    "V": q['distribución']['V']
                }
                for q in quinielas
            ])
            
            st.dataframe(quinielas_df, use_container_width=True, hide_index=True)
            
            # Verificar problemas básicos
            problemas = []
            for q in quinielas:
                if q['empates'] < 4 or q['empates'] > 6:
                    problemas.append(f"{q['id']}: {q['empates']} empates (debe ser 4-6)")
                
                max_conc = max(q['distribución'].values()) / 14
                if max_conc > 0.70:
                    signo = max(q['distribución'], key=q['distribución'].get)
                    problemas.append(f"{q['id']}: concentración {signo} = {max_conc:.1%} (>70%)")
            
            if problemas:
                st.warning("⚠️ **Problemas detectados en quinielas Core:**")
                for problema in problemas:
                    st.warning(f"• {problema}")
            else:
                st.success("✅ Todas las quinielas Core cumplen reglas básicas")
            
            st.success("✅ Generación completada - Procede al **PASO 4: Validación**")
    
    def paso_4_validacion(self):
        """PASO 4: Validación regla por regla"""
        st.header("✅ PASO 4: Validación del Portafolio")
        st.markdown("**Objetivo**: Validar cada regla por separado para identificar problemas específicos")
        
        # Verificar prerequisitos
        if 'quinielas_generadas' not in st.session_state:
            st.warning("⚠️ Primero completa el **PASO 3: Generación**")
            return
        
        st.success("✅ Prerequisitos cumplidos")
        
        # Botón para validar
        if st.button("▶️ Ejecutar Validación Completa", type="primary"):
            with st.spinner("Validando portafolio..."):
                try:
                    quinielas = st.session_state.quinielas_generadas
                    
                    # Ejecutar validación usando el validador existente
                    resultado_validacion = self.validator.validar_portafolio_completo(quinielas)
                    
                    # Guardar resultados
                    st.session_state.validacion_completa = resultado_validacion
                    
                    st.success("✅ Validación completada")
                    
                except Exception as e:
                    st.error(f"❌ Error en validación: {e}")
                    st.exception(e)
        
        # Mostrar resultados de validación
        if 'validacion_completa' in st.session_state:
            st.markdown("---")
            validacion = st.session_state.validacion_completa
            
            # Estado general
            if validacion['es_valido']:
                st.success("🎉 **PORTAFOLIO COMPLETAMENTE VÁLIDO**")
            else:
                st.error("❌ **PORTAFOLIO INVÁLIDO** - Revisa reglas específicas")
            
            # Detalle regla por regla
            st.subheader("📋 Detalle por Regla")
            
            reglas = validacion['detalle_validaciones']
            descripciones = {
                "distribucion_global": "Distribución global en rangos históricos (35-41% L, 25-33% E, 30-36% V)",
                "empates_individuales": "4-6 empates por quiniela individual",
                "concentracion_maxima": "≤70% concentración general, ≤60% en primeros 3 partidos",
                "arquitectura_core_satelites": "Arquitectura correcta (actualmente solo Core)",
                "correlacion_jaccard": "Correlación entre pares ≤ 0.57 (no aplica para solo Core)",
                "distribucion_divisores": "Distribución equilibrada de resultados"
            }
            
            for regla, cumple in reglas.items():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if cumple:
                        st.success("✅ CUMPLE")
                    else:
                        st.error("❌ FALLA")
                
                with col2:
                    st.write(f"**{regla.replace('_', ' ').title()}**")
                    st.caption(descripciones.get(regla, "Sin descripción"))
                    
                    # Mostrar detalles específicos para reglas que fallan
                    if not cumple:
                        if regla == "distribucion_global":
                            self._mostrar_detalle_distribucion_global(validacion)
                        elif regla == "empates_individuales":
                            self._mostrar_detalle_empates_individuales()
                        elif regla == "concentracion_maxima":
                            self._mostrar_detalle_concentracion()
            
            # Resumen con próximos pasos
            st.markdown("---")
            st.subheader("🎯 Próximos Pasos")
            
            if validacion['es_valido']:
                st.success("🎉 **¡Felicitaciones!** Tu portafolio Core es completamente válido.")
                st.info("💡 **Siguientes opciones:**")
                st.info("• Agregar satélites para completar las 30 quinielas")
                st.info("• Optimizar con GRASP-Annealing")
                st.info("• Exportar las quinielas Core actuales")
            else:
                st.error("🔧 **Se requieren correcciones:**")
                reglas_fallidas = [regla for regla, cumple in reglas.items() if not cumple]
                for regla in reglas_fallidas:
                    st.error(f"• Corregir: {regla.replace('_', ' ')}")
                
                st.info("💡 **Opciones de corrección:**")
                st.info("• Volver al Paso 1 y ajustar datos")
                st.info("• Ajustar parámetros de clasificación")
                st.info("• Usar corrección manual/IA (próximamente)")
    
    def _mostrar_detalle_distribucion_global(self, validacion):
        """Mostrar detalles específicos de distribución global"""
        metricas = validacion.get('metricas', {})
        if 'distribucion_global' in metricas:
            dist = metricas['distribucion_global']['porcentajes']
            st.warning(f"Distribución actual: L={dist['L']:.1%}, E={dist['E']:.1%}, V={dist['V']:.1%}")
            st.info("Objetivo: L=35-41%, E=25-33%, V=30-36%")
    
    def _mostrar_detalle_empates_individuales(self):
        """Mostrar detalles de empates problemáticos"""
        if 'quinielas_generadas' in st.session_state:
            quinielas = st.session_state.quinielas_generadas
            problematicas = [q for q in quinielas if q['empates'] < 4 or q['empates'] > 6]
            if problematicas:
                st.warning("Quinielas problemáticas:")
                for q in problematicas:
                    st.warning(f"• {q['id']}: {q['empates']} empates")
    
    def _mostrar_detalle_concentracion(self):
        """Mostrar detalles de concentración problemática"""
        if 'quinielas_generadas' in st.session_state:
            quinielas = st.session_state.quinielas_generadas
            problematicas = []
            for q in quinielas:
                max_conc = max(q['distribución'].values()) / 14
                if max_conc > 0.70:
                    signo = max(q['distribución'], key=q['distribución'].get)
                    problematicas.append(f"{q['id']}: {signo} = {max_conc:.1%}")
            
            if problematicas:
                st.warning("Concentraciones problemáticas:")
                for problema in problematicas:
                    st.warning(f"• {problema}")
    
    def _convertir_csv_a_datos(self, df):
        """Convertir DataFrame de CSV al formato interno"""
        datos = []
        for idx, row in df.iterrows():
            # Verificar columnas mínimas
            if 'home' not in row or 'away' not in row:
                raise ValueError("CSV debe tener columnas 'home' y 'away'")
            
            # Probabilidades
            if all(col in row for col in ['prob_local', 'prob_empate', 'prob_visitante']):
                prob_local = float(row['prob_local'])
                prob_empate = float(row['prob_empate'])
                prob_visitante = float(row['prob_visitante'])
                
                # Normalizar si no suman 1
                total = prob_local + prob_empate + prob_visitante
                if abs(total - 1.0) > 0.01:
                    prob_local /= total
                    prob_empate /= total
                    prob_visitante /= total
            else:
                # Generar probabilidades balanceadas
                prob_local, prob_empate, prob_visitante = self.data_loader._generar_probabilidades_con_anclas_garantizadas(idx)
            
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
                'concurso_id': str(row.get('concurso_id', '2283'))
            }
            datos.append(partido)
        
        return datos

def main():
    """Función principal"""
    app = StepByStepProgolApp()
    app.run()

if __name__ == "__main__":
    main()