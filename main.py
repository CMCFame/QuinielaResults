# progol_optimizer/main.py
"""
Orquestador Principal MEJORADO con instrumentación completa y fallbacks robustos
Integra todos los componentes mejorados: logging, datos garantizados, optimizador híbrido y IA con safeguards
"""

import logging
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

# Configurar logging antes de importar otros módulos
from logging_setup import setup_progol_logging, get_instrumentor

# Imports de componentes MEJORADOS
from data.loader import EnhancedDataLoader
from data.validator import DataValidator
from models.classifier import PartidoClassifier
from models.calibrator import BayesianCalibrator
from models.ai_assistant import EnhancedProgolAIAssistant
from portfolio.core_generator import CoreGenerator
from portfolio.satellite_generator import SatelliteGenerator
from portfolio.optimizer import GRASPAnnealing

# Importación condicional del optimizador híbrido mejorado
try:
    from portfolio.hybrid_optimizer import EnhancedHybridOptimizer
    ENHANCED_HYBRID_AVAILABLE = True
except ImportError:
    ENHANCED_HYBRID_AVAILABLE = False

from validation.portfolio_validator import PortfolioValidator
from export.exporter import PortfolioExporter


class EnhancedProgolOptimizer:
    """
    Orquestador Principal MEJORADO que garantiza portafolios válidos
    
    Características mejoradas:
    1. Instrumentación completa con logging estructurado
    2. Múltiples estrategias de optimización con fallbacks automáticos  
    3. IA con safeguards contra cambios masivos
    4. Validación rigurosa en cada paso
    5. Trazabilidad completa de decisiones
    6. Manejo robusto de errores con recuperación automática
    """
    
    def __init__(self, log_level: str = "INFO", debug_ai: bool = False):
        # Configurar sistema de logging mejorado
        self.instrumentor = setup_progol_logging(
            log_level=log_level,
            enable_file_logging=True,
            enable_json_format=True,
            debug_ai=debug_ai
        )
        
        self.logger = logging.getLogger(__name__)
        self.session_timer = self.instrumentor.start_timer("session_complete")
        
        self.logger.info("🚀 Inicializando EnhancedProgolOptimizer...")
        
        # Configurar API keys si están disponibles
        self._setup_api_keys()
        
        # Inicializar componentes MEJORADOS con instrumentación
        init_timer = self.instrumentor.start_timer("component_initialization")
        
        try:
            self.data_loader = EnhancedDataLoader()
            self.data_validator = DataValidator()
            self.classifier = PartidoClassifier()
            self.calibrator = BayesianCalibrator()
            self.portfolio_validator = PortfolioValidator()
            self.exporter = PortfolioExporter()
            
            # IA con safeguards
            self.ai_assistant = EnhancedProgolAIAssistant()
            
            # Optimizadores (híbrido + legacy como fallback)
            self.legacy_optimizer = GRASPAnnealing()
            
            # Contadores de estrategias utilizadas
            self.strategy_usage = {
                "enhanced_hybrid": 0,
                "legacy_grasp": 0,
                "ai_corrections": 0,
                "emergency_fallbacks": 0
            }
            
            self.instrumentor.end_timer(init_timer, success=True, metrics={
                "components_initialized": 7,
                "ai_enabled": self.ai_assistant.enabled,
                "hybrid_available": ENHANCED_HYBRID_AVAILABLE
            })
            
            self.logger.info("✅ EnhancedProgolOptimizer inicializado correctamente")
            
        except Exception as e:
            self.instrumentor.end_timer(init_timer, success=False)
            self.logger.error(f"❌ Error inicializando componentes: {e}")
            raise
    
    def procesar_concurso_completo(self, archivo_datos: str = None, 
                                  concurso_id: str = "2283",
                                  metodo_preferido: str = "enhanced_hybrid",
                                  forzar_ai: bool = False,
                                  max_intentos: int = 3) -> Dict[str, Any]:
        """
        Procesa concurso completo con múltiples estrategias y recuperación automática
        
        Args:
            archivo_datos: Ruta al CSV de datos (None para generar ejemplos)
            concurso_id: ID del concurso
            metodo_preferido: "enhanced_hybrid", "legacy", "auto"
            forzar_ai: Si forzar uso de IA incluso si el resultado inicial es válido
            max_intentos: Máximo número de intentos antes de fallar
            
        Returns:
            Dict: Resultado completo con portafolio, validación y metadatos
        """
        session_start = time.time()
        self.logger.info(f"=== PROCESANDO CONCURSO {concurso_id} ===")
        self.logger.info(f"Método preferido: {metodo_preferido}, AI forzada: {forzar_ai}")
        
        for intento in range(max_intentos):
            attempt_timer = self.instrumentor.start_timer(f"attempt_{intento + 1}")
            
            try:
                self.logger.info(f"🔄 Intento {intento + 1}/{max_intentos}")
                
                # FASE 1: Preparación de datos con validación rigurosa
                partidos_procesados = self._fase_preparacion_datos(archivo_datos, intento)
                
                if not partidos_procesados:
                    self.instrumentor.end_timer(attempt_timer, success=False)
                    continue
                
                # FASE 2: Generación de portafolio con estrategia adaptativa
                portafolio_inicial = self._fase_generacion_portafolio(
                    partidos_procesados, metodo_preferido, intento
                )
                
                if not portafolio_inicial:
                    self.instrumentor.end_timer(attempt_timer, success=False)
                    continue
                
                # FASE 3: Validación y corrección inteligente
                portafolio_final, validacion_resultado = self._fase_validacion_y_correccion(
                    portafolio_inicial, partidos_procesados, forzar_ai
                )
                
                if not portafolio_final:
                    self.instrumentor.end_timer(attempt_timer, success=False)
                    continue
                
                # FASE 4: Exportación y finalización
                resultado_final = self._fase_exportacion_y_finalizacion(
                    portafolio_final, partidos_procesados, validacion_resultado, concurso_id
                )
                
                # Éxito - terminar timer y retornar
                session_duration = time.time() - session_start
                
                self.instrumentor.end_timer(attempt_timer, success=True, metrics={
                    "final_attempt": intento + 1,
                    "session_duration": session_duration,
                    "portfolio_valid": resultado_final["validacion"]["es_valido"]
                })
                
                self.logger.info(f"🎉 Concurso {concurso_id} procesado exitosamente en intento {intento + 1}")
                return resultado_final
                
            except Exception as e:
                self.instrumentor.end_timer(attempt_timer, success=False)
                self.logger.error(f"❌ Error en intento {intento + 1}: {e}")
                
                if intento == max_intentos - 1:
                    # Último intento fallido
                    self.logger.error("💥 Todos los intentos fallaron")
                    return self._generar_resultado_error(concurso_id, f"Falló después de {max_intentos} intentos: {e}")
        
        # No debería llegar aquí, pero por seguridad
        return self._generar_resultado_error(concurso_id, "Error desconocido en el procesamiento")
    
    def _fase_preparacion_datos(self, archivo_datos: str, intento: int) -> Optional[List[Dict[str, Any]]]:
        """
        Fase 1: Preparación rigurosa de datos con validación y clasificación
        """
        phase_timer = self.instrumentor.start_timer("phase_data_preparation")
        
        try:
            self.logger.info("📊 FASE 1: Preparación de datos")
            
            # 1.1: Cargar o generar datos
            if archivo_datos:
                partidos = self.data_loader.cargar_datos(archivo_datos)
            else:
                partidos = self.data_loader.generar_datos_ejemplo_mejorados()
            
            if not partidos:
                self.logger.error("❌ No se pudieron cargar los datos")
                self.instrumentor.end_timer(phase_timer, success=False)
                return None
            
            # 1.2: Validación estructural estricta
            es_valido, errores = self.data_validator.validar_estructura(partidos)
            if not es_valido:
                self.logger.error(f"❌ Validación estructural falló: {errores}")
                self.instrumentor.end_timer(phase_timer, success=False)
                return None
            
            # 1.3: Calibración bayesiana con instrumentación
            calibration_timer = self.instrumentor.start_timer("bayesian_calibration")
            partidos_calibrados = self.calibrator.calibrar_concurso_completo(partidos)
            self.instrumentor.end_timer(calibration_timer, success=True)
            
            # 1.4: Clasificación de partidos con validación de Anclas
            classification_timer = self.instrumentor.start_timer("match_classification")
            partidos_procesados = []
            
            for i, partido_calibrado in enumerate(partidos_calibrados):
                clasificacion = self.classifier.clasificar_partido(partido_calibrado)
                partido_final = {
                    **partido_calibrado,
                    "id": i,
                    "clasificacion": clasificacion
                }
                partidos_procesados.append(partido_final)
            
            # Validar que tenemos suficientes Anclas
            anclas_count = sum(1 for p in partidos_procesados if p["clasificacion"] == "Ancla")
            if anclas_count < 3:
                self.logger.warning(f"⚠️ Solo {anclas_count} partidos Ancla encontrados")
                # En caso de pocas anclas, relajar criterios
                if intento > 0:
                    self._forzar_anclas_minimas(partidos_procesados)
            
            stats_clasificacion = self.classifier.obtener_estadisticas_clasificacion(partidos_procesados)
            
            self.instrumentor.end_timer(classification_timer, success=True, metrics=stats_clasificacion)
            self.instrumentor.end_timer(phase_timer, success=True, metrics={
                "partidos_procesados": len(partidos_procesados),
                "anclas_encontradas": anclas_count
            })
            
            self.logger.info(f"✅ Fase 1 completada: {len(partidos_procesados)} partidos procesados")
            return partidos_procesados
            
        except Exception as e:
            self.instrumentor.end_timer(phase_timer, success=False)
            self.logger.error(f"❌ Error en preparación de datos: {e}")
            return None
    
    def _fase_generacion_portafolio(self, partidos: List[Dict[str, Any]], 
                                  metodo_preferido: str, intento: int) -> Optional[List[Dict[str, Any]]]:
        """
        Fase 2: Generación de portafolio con estrategia adaptativa
        """
        phase_timer = self.instrumentor.start_timer("phase_portfolio_generation")
        
        try:
            self.logger.info("🎯 FASE 2: Generación de portafolio")
            
            # Determinar estrategia basada en método preferido e intento
            if metodo_preferido == "auto":
                estrategia = "enhanced_hybrid" if intento == 0 else "legacy"
            else:
                estrategia = metodo_preferido
            
            self.logger.info(f"🔧 Usando estrategia: {estrategia}")
            
            portafolio = None
            
            # Ejecutar estrategia seleccionada
            if estrategia == "enhanced_hybrid" and ENHANCED_HYBRID_AVAILABLE:
                portafolio = self._ejecutar_estrategia_hibrida_mejorada(partidos)
                if portafolio:
                    self.strategy_usage["enhanced_hybrid"] += 1
            
            if not portafolio or estrategia == "legacy":
                # Fallback a estrategia legacy
                self.logger.info("🔄 Usando estrategia legacy como fallback")
                portafolio = self._ejecutar_estrategia_legacy(partidos)
                if portafolio:
                    self.strategy_usage["legacy_grasp"] += 1
            
            if not portafolio:
                # Fallback de emergencia
                self.logger.warning("🚨 Activando fallback de emergencia")
                portafolio = self._ejecutar_fallback_emergencia(partidos)
                if portafolio:
                    self.strategy_usage["emergency_fallbacks"] += 1
            
            if portafolio and len(portafolio) == 30:
                self.instrumentor.end_timer(phase_timer, success=True, metrics={
                    "strategy_used": estrategia,
                    "portfolio_size": len(portafolio)
                })
                
                self.logger.info(f"✅ Fase 2 completada: portafolio de {len(portafolio)} quinielas generado")
                return portafolio
            else:
                self.instrumentor.end_timer(phase_timer, success=False)
                self.logger.error("❌ No se pudo generar portafolio válido")
                return None
                
        except Exception as e:
            self.instrumentor.end_timer(phase_timer, success=False)
            self.logger.error(f"❌ Error en generación de portafolio: {e}")
            return None
    
    def _fase_validacion_y_correccion(self, portafolio: List[Dict[str, Any]], 
                                    partidos: List[Dict[str, Any]], 
                                    forzar_ai: bool) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Fase 3: Validación rigurosa y corrección inteligente con IA
        """
        phase_timer = self.instrumentor.start_timer("phase_validation_correction")
        
        try:
            self.logger.info("📋 FASE 3: Validación y corrección")
            
            # 3.1: Validación inicial completa
            validacion_inicial = self.portfolio_validator.validar_portafolio_completo(portafolio)
            
            self.instrumentor.log_validation_result(
                component="portfolio_initial",
                rules_passed=validacion_inicial["detalle_validaciones"],
                metrics=validacion_inicial["metricas"]
            )
            
            es_valido_inicial = validacion_inicial["es_valido"]
            portafolio_trabajando = portafolio
            ai_fue_utilizada = False
            
            # 3.2: Corrección con IA si es necesario o forzado
            if (not es_valido_inicial or forzar_ai) and self.ai_assistant.enabled:
                self.logger.info("🤖 Activando corrección inteligente con IA")
                
                correction_timer = self.instrumentor.start_timer("ai_intelligent_correction")
                
                # Corrección individual de quinielas problemáticas
                portafolio_corregido_individual = self._corregir_quinielas_individuales(
                    portafolio_trabajando, partidos
                )
                
                if portafolio_corregido_individual:
                    portafolio_trabajando = portafolio_corregido_individual
                    ai_fue_utilizada = True
                    self.strategy_usage["ai_corrections"] += 1
                
                # Corrección global si es necesaria
                if self._requiere_correccion_global(portafolio_trabajando):
                    self.logger.info("🌐 Aplicando corrección global")
                    portafolio_global = self.ai_assistant.optimizar_distribucion_global_con_safeguards(
                        portafolio_trabajando, partidos
                    )
                    
                    if portafolio_global:
                        portafolio_trabajando = portafolio_global
                
                self.instrumentor.end_timer(correction_timer, success=True, metrics={
                    "ai_individual_corrections": ai_fue_utilizada,
                    "ai_global_optimization": self._requiere_correccion_global(portafolio_trabajando)
                })
            
            # 3.3: Validación final
            validacion_final = self.portfolio_validator.validar_portafolio_completo(portafolio_trabajando)
            
            self.instrumentor.log_validation_result(
                component="portfolio_final",
                rules_passed=validacion_final["detalle_validaciones"],
                metrics=validacion_final["metricas"]
            )
            
            # 3.4: Añadir metadatos de corrección
            validacion_final["ai_utilizada"] = ai_fue_utilizada
            validacion_final["correccion_exitosa"] = (
                validacion_final["es_valido"] or 
                (not es_valido_inicial and validacion_final["es_valido"])
            )
            
            self.instrumentor.end_timer(phase_timer, success=validacion_final["es_valido"], metrics={
                "initial_valid": es_valido_inicial,
                "final_valid": validacion_final["es_valido"],
                "ai_used": ai_fue_utilizada
            })
            
            if validacion_final["es_valido"]:
                self.logger.info("✅ Fase 3 completada: portafolio válido obtenido")
            else:
                self.logger.warning("⚠️ Fase 3: portafolio sigue teniendo problemas de validación")
            
            return portafolio_trabajando, validacion_final
            
        except Exception as e:
            self.instrumentor.end_timer(phase_timer, success=False)
            self.logger.error(f"❌ Error en validación y corrección: {e}")
            return None, {"es_valido": False, "error": str(e)}
    
    def _fase_exportacion_y_finalizacion(self, portafolio: List[Dict[str, Any]], 
                                        partidos: List[Dict[str, Any]], 
                                        validacion: Dict[str, Any], 
                                        concurso_id: str) -> Dict[str, Any]:
        """
        Fase 4: Exportación de resultados y finalización con metadatos completos
        """
        phase_timer = self.instrumentor.start_timer("phase_export_finalization")
        
        try:
            self.logger.info("💾 FASE 4: Exportación y finalización")
            
            # 4.1: Exportar archivos
            archivos_exportados = self.exporter.exportar_portafolio_completo(
                portafolio,
                partidos,
                validacion["metricas"],
                concurso_id
            )
            
            # 4.2: Recopilar estadísticas de sesión
            session_summary = self.instrumentor.get_session_summary()
            ai_stats = self.ai_assistant.get_usage_stats() if self.ai_assistant.enabled else {}
            
            # 4.3: Preparar resultado final completo
            resultado_final = {
                "success": True,
                "portafolio": portafolio,
                "partidos": partidos,
                "validacion": validacion,
                "metricas": validacion["metricas"],
                "archivos_exportados": archivos_exportados,
                "concurso_id": concurso_id,
                
                # Metadatos de la sesión
                "ai_utilizada": validacion.get("ai_utilizada", False),
                "estrategia_final": self._determinar_estrategia_utilizada(),
                "estadisticas_estrategias": self.strategy_usage,
                "session_summary": session_summary,
                "ai_stats": ai_stats,
                
                # Información de trazabilidad
                "version": "2.0-enhanced",
                "timestamp": time.time()
            }
            
            self.instrumentor.end_timer(phase_timer, success=True, metrics={
                "files_exported": len(archivos_exportados),
                "final_validation": validacion["es_valido"]
            })
            
            self.logger.info("✅ Fase 4 completada: resultados exportados exitosamente")
            return resultado_final
            
        except Exception as e:
            self.instrumentor.end_timer(phase_timer, success=False)
            self.logger.error(f"❌ Error en exportación: {e}")
            return self._generar_resultado_error(concurso_id, f"Error en exportación: {e}")
    
    def _ejecutar_estrategia_hibrida_mejorada(self, partidos: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Ejecuta la estrategia híbrida mejorada con CP-SAT + GRASP-Annealing
        """
        strategy_timer = self.instrumentor.start_timer("strategy_enhanced_hybrid")
        
        try:
            self.logger.info("🔬 Ejecutando estrategia híbrida mejorada (CP-SAT + GRASP)")
            
            partidos_df = pd.DataFrame(partidos)
            hybrid_optimizer = EnhancedHybridOptimizer(partidos_df)
            portafolio = hybrid_optimizer.generate_portfolio()
            
            self.instrumentor.end_timer(strategy_timer, success=portafolio is not None, metrics={
                "portfolio_generated": portafolio is not None,
                "portfolio_size": len(portafolio) if portafolio else 0
            })
            
            return portafolio
            
        except Exception as e:
            self.instrumentor.end_timer(strategy_timer, success=False)
            self.logger.error(f"Error en estrategia híbrida: {e}")
            return None
    
    def _ejecutar_estrategia_legacy(self, partidos: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Ejecuta la estrategia legacy Core + Satélites + GRASP-Annealing
        """
        strategy_timer = self.instrumentor.start_timer("strategy_legacy")
        
        try:
            self.logger.info("🔧 Ejecutando estrategia legacy (Core + Satélites + GRASP)")
            
            # Generador de Core
            core_generator = CoreGenerator()
            quinielas_core = core_generator.generar_quinielas_core(partidos)
            
            if not quinielas_core or len(quinielas_core) != 4:
                self.logger.error("❌ No se pudieron generar quinielas Core")
                return None
            
            # Generador de Satélites
            satellite_generator = SatelliteGenerator()
            quinielas_satelites = satellite_generator.generar_satelites_optimizados(
                partidos, quinielas_core
            )
            
            if not quinielas_satelites or len(quinielas_satelites) != 26:
                self.logger.error("❌ No se pudieron generar quinielas Satélite")
                return None
            
            # Combinar y optimizar con GRASP-Annealing
            portafolio_inicial = quinielas_core + quinielas_satelites
            portafolio_optimizado = self.legacy_optimizer.optimizar_portafolio_grasp_annealing(
                portafolio_inicial, partidos
            )
            
            self.instrumentor.end_timer(strategy_timer, success=True, metrics={
                "core_generated": len(quinielas_core),
                "satellites_generated": len(quinielas_satelites),
                "portfolio_optimized": portafolio_optimizado is not None
            })
            
            return portafolio_optimizado
            
        except Exception as e:
            self.instrumentor.end_timer(strategy_timer, success=False)
            self.logger.error(f"Error en estrategia legacy: {e}")
            return None
    
    def _ejecutar_fallback_emergencia(self, partidos: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Fallback de emergencia con generación determinística simple
        """
        fallback_timer = self.instrumentor.start_timer("strategy_emergency_fallback")
        
        try:
            self.logger.warning("🚨 Ejecutando fallback de emergencia")
            
            # Generación simple y determinística
            portafolio_emergencia = []
            
            for i in range(30):
                quiniela_simple = {
                    "id": i,
                    "tipo": "Emergency",
                    "quiniela": [],
                    "distribución": {"L": 0, "E": 0, "V": 0}
                }
                
                # Asignación simple basada en probabilidades más altas
                for partido in partidos:
                    prob_local = partido.get("prob_local", 0.33)
                    prob_empate = partido.get("prob_empate", 0.33)
                    prob_visitante = partido.get("prob_visitante", 0.33)
                    
                    if prob_local >= prob_empate and prob_local >= prob_visitante:
                        resultado = "L"
                    elif prob_empate >= prob_visitante:
                        resultado = "E"
                    else:
                        resultado = "V"
                    
                    quiniela_simple["quiniela"].append(resultado)
                    quiniela_simple["distribución"][resultado] += 1
                
                portafolio_emergencia.append(quiniela_simple)
            
            self.instrumentor.end_timer(fallback_timer, success=True, metrics={
                "emergency_portfolio_generated": len(portafolio_emergencia)
            })
            
            return portafolio_emergencia
            
        except Exception as e:
            self.instrumentor.end_timer(fallback_timer, success=False)
            self.logger.error(f"Error en fallback de emergencia: {e}")
            return None
    
    def _corregir_quinielas_individuales(self, portafolio: List[Dict[str, Any]], 
                                       partidos: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Corrección individual de quinielas con problemas usando IA
        """
        correction_timer = self.instrumentor.start_timer("individual_ai_corrections")
        
        try:
            portafolio_corregido = []
            correcciones_aplicadas = 0
            
            for quiniela in portafolio:
                # Validar quiniela individual
                validacion_individual = self.portfolio_validator.validar_quiniela_individual(quiniela)
                
                if not validacion_individual["es_valida"] and self.ai_assistant.enabled:
                    # Intentar corrección con IA
                    reglas_violadas = validacion_individual.get("reglas_violadas", [])
                    quiniela_corregida = self.ai_assistant.corregir_quiniela_con_safeguards(
                        quiniela, partidos, reglas_violadas
                    )
                    
                    if quiniela_corregida:
                        portafolio_corregido.append(quiniela_corregida)
                        correcciones_aplicadas += 1
                    else:
                        portafolio_corregido.append(quiniela)  # Conservar original si no se pudo corregir
                else:
                    portafolio_corregido.append(quiniela)  # Quiniela válida, conservar
            
            self.instrumentor.end_timer(correction_timer, success=True, metrics={
                "individual_corrections_applied": correcciones_aplicadas
            })
            
            self.logger.info(f"🔧 Aplicadas {correcciones_aplicadas} correcciones individuales")
            return portafolio_corregido
            
        except Exception as e:
            self.instrumentor.end_timer(correction_timer, success=False)
            self.logger.error(f"Error en correcciones individuales: {e}")
            return None
    
    def _requiere_correccion_global(self, portafolio: List[Dict[str, Any]]) -> bool:
        """
        Determina si se requiere corrección global del portafolio
        """
        # Verificar distribución global
        total_L = sum(q.get("distribución", {}).get("L", 0) for q in portafolio)
        total_E = sum(q.get("distribución", {}).get("E", 0) for q in portafolio)
        total_V = sum(q.get("distribución", {}).get("V", 0) for q in portafolio)
        
        total = total_L + total_E + total_V
        if total == 0:
            return True
        
        porc_L = total_L / total
        porc_E = total_E / total
        porc_V = total_V / total
        
        # Rangos históricos esperados
        return not (0.35 <= porc_L <= 0.41 and 0.25 <= porc_E <= 0.33 and 0.30 <= porc_V <= 0.36)
    
    def _forzar_anclas_minimas(self, partidos: List[Dict[str, Any]]):
        """Fuerza al menos 3 partidos como Ancla si hay muy pocos"""
        partidos_ordenados = sorted(partidos, 
                                   key=lambda p: max(p["prob_local"], p["prob_empate"], p["prob_visitante"]), 
                                   reverse=True)
        
        anclas_forzadas = 0
        for partido in partidos_ordenados:
            if anclas_forzadas >= 3:
                break
            if partido["clasificacion"] != "Ancla":
                max_prob = max(partido["prob_local"], partido["prob_empate"], partido["prob_visitante"])
                if max_prob > 0.55:  # Criterio relajado
                    partido["clasificacion"] = "Ancla"
                    anclas_forzadas += 1
        
        self.logger.info(f"🔒 Forzadas {anclas_forzadas} Anclas adicionales")
    
    def _determinar_estrategia_utilizada(self) -> str:
        """Determina qué estrategia fue finalmente exitosa"""
        if self.strategy_usage["enhanced_hybrid"] > 0:
            return "enhanced_hybrid"
        elif self.strategy_usage["legacy_grasp"] > 0:
            return "legacy_grasp"
        elif self.strategy_usage["emergency_fallbacks"] > 0:
            return "emergency_fallback"
        else:
            return "unknown"
    
    def _setup_api_keys(self):
        """Configura API keys desde múltiples fuentes"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                self.logger.debug("🔑 API key configurada desde Streamlit secrets")
        except ImportError:
            pass
    
    def _generar_resultado_error(self, concurso_id: str, error_msg: str) -> Dict[str, Any]:
        """Genera resultado de error estructurado"""
        return {
            "success": False,
            "error": error_msg,
            "concurso_id": concurso_id,
            "portafolio": [],
            "validacion": {"es_valido": False},
            "ai_utilizada": False,
            "estrategia_final": "none",
            "timestamp": time.time(),
            "version": "2.0-enhanced"
        }


# Clase de compatibilidad hacia atrás
class ProgolOptimizer(EnhancedProgolOptimizer):
    """Wrapper para mantener compatibilidad con código existente"""
    
    def __init__(self):
        super().__init__()
        self.logger.warning("⚠️ Usando ProgolOptimizer legacy - migrar a EnhancedProgolOptimizer")
    
    def procesar_concurso(self, archivo_datos: str = None, concurso_id: str = "2283", 
                         forzar_ai: bool = False, method: str = "enhanced_hybrid") -> Dict[str, Any]:
        """Wrapper para mantener compatibilidad de método"""
        return self.procesar_concurso_completo(
            archivo_datos=archivo_datos,
            concurso_id=concurso_id,
            metodo_preferido=method,
            forzar_ai=forzar_ai
        )


def main():
    """Función principal para uso por línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Progol Optimizer v2.0")
    parser.add_argument("--archivo", "-f", help="Archivo CSV con datos de partidos")
    parser.add_argument("--concurso", "-c", default="2283", help="ID del concurso")
    parser.add_argument("--metodo", "-m", default="enhanced_hybrid", 
                       choices=['enhanced_hybrid', 'legacy', 'auto'], 
                       help="Método de optimización")
    parser.add_argument("--debug", "-d", action="store_true", help="Modo debug")
    parser.add_argument("--debug-ai", action="store_true", help="Debug específico de IA")
    parser.add_argument("--api-key", "-k", help="OpenAI API key")
    parser.add_argument("--forzar-ai", "-ai", action="store_true", help="Forzar uso de IA")
    parser.add_argument("--max-intentos", type=int, default=3, help="Máximo número de intentos")
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = "DEBUG" if args.debug else "INFO"
    
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    try:
        # Inicializar optimizador mejorado
        optimizer = EnhancedProgolOptimizer(log_level=log_level, debug_ai=args.debug_ai)
        
        # Procesar concurso
        resultado = optimizer.procesar_concurso_completo(
            archivo_datos=args.archivo,
            concurso_id=args.concurso,
            metodo_preferido=args.metodo,
            forzar_ai=args.forzar_ai,
            max_intentos=args.max_intentos
        )
        
        # Mostrar resultados
        if resultado["success"]:
            print(f"✅ Concurso {args.concurso} procesado exitosamente")
            print(f"   Estrategia utilizada: {resultado['estrategia_final']}")
            print(f"   AI utilizada: {'Sí' if resultado.get('ai_utilizada') else 'No'}")
            print(f"   Portafolio válido: {'Sí' if resultado['validacion']['es_valido'] else 'No'}")
            print(f"   Archivos generados: {len(resultado.get('archivos_exportados', {}))}")
            print(f"   Ubicación: outputs/")
        else:
            print(f"❌ Error procesando concurso {args.concurso}")
            print(f"   Error: {resultado['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()