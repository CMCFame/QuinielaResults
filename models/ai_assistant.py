# progol_optimizer/models/ai_assistant.py
"""
Sistema de IA MEJORADO con safeguards contra sobrescritura y delta tracking
Implementa corrección inteligente con límites de distancia Hamming y rollback automático
"""

import os
import json
import logging
import re
import time
import hashlib
import copy
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from logging_setup import get_instrumentor


class EnhancedProgolAIAssistant:
    """
    Asistente IA MEJORADO con safeguards completos:
    1. Delta tracking para evitar cambios masivos  
    2. Rollback automático si la distancia Hamming > umbral
    3. Logging estructurado de todas las interacciones
    4. Validación previa y posterior de cambios
    5. Límites de tokens y tiempo para evitar bucles
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.instrumentor = get_instrumentor()
        
        # Configuración de seguridad
        self.max_hamming_distance = 8  # Máximo 8 partidos cambiados de 14
        self.max_global_changes = 20   # Máximo 20 cambios en todo el portafolio
        self.max_retries = 3           # Máximo 3 intentos por corrección
        self.timeout_seconds = 30      # Timeout por llamada
        
        # Configurar API key con múltiples fuentes
        self.api_key = self._setup_api_key(api_key)
        
        if not self.api_key or not OPENAI_AVAILABLE:
            self.logger.warning("⚠️ IA no disponible: API key faltante o OpenAI no instalado")
            self.enabled = False
        else:
            try:
                self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
                self.enabled = True
                self.logger.info("✅ Sistema IA mejorado inicializado con safeguards")
                
                # Contadores de uso para monitoring
                self.usage_stats = {
                    "total_calls": 0,
                    "successful_corrections": 0,
                    "rejected_by_hamming": 0,
                    "rollbacks_performed": 0,
                    "timeouts": 0
                }
                
            except Exception as e:
                self.logger.error(f"Error inicializando OpenAI: {e}")
                self.enabled = False
    
    def _setup_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """Configura API key con múltiples fuentes de fallback"""
        
        # Prioridad 1: Parámetro directo
        if api_key:
            return api_key
        
        # Prioridad 2: Variable de entorno
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key
        
        # Prioridad 3: Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
                return st.secrets["OPENAI_API_KEY"]
        except ImportError:
            pass
        
        return None
    
    def corregir_quiniela_con_safeguards(self, quiniela: Dict[str, Any], 
                                        partidos: List[Dict[str, Any]], 
                                        reglas_violadas: List[str]) -> Optional[Dict[str, Any]]:
        """
        Corrige quiniela individual con safeguards completos contra cambios masivos
        """
        if not self.enabled:
            return None
        
        timer_id = self.instrumentor.start_timer("ai_correccion_individual")
        quiniela_id = quiniela.get('id', 'Unknown')
        
        try:
            self.usage_stats["total_calls"] += 1
            
            # Crear backup del estado original
            quiniela_original = copy.deepcopy(quiniela)
            
            self.logger.info(f"🤖 Iniciando corrección IA con safeguards para {quiniela_id}")
            self.logger.debug(f"Reglas violadas: {reglas_violadas}")
            
            # Pre-validación: verificar que realmente necesita corrección
            if not self._requiere_correccion(quiniela, reglas_violadas):
                self.logger.info(f"✅ {quiniela_id} no requiere corrección IA")
                self.instrumentor.end_timer(timer_id, success=True, metrics={"action": "no_correction_needed"})
                return quiniela
            
            mejor_correccion = None
            menor_distancia = float('inf')
            
            # Intentar corrección con múltiples enfoques
            for intento in range(self.max_retries):
                attempt_timer = self.instrumentor.start_timer(f"ai_attempt_{intento}")
                
                try:
                    self.logger.debug(f"🔄 Intento {intento + 1}/{self.max_retries}")
                    
                    # Generar corrección con variación en temperatura
                    temperatura = 0.2 + (intento * 0.15)  # Aumentar randomización en intentos
                    
                    correccion_candidata = self._ejecutar_correccion_con_timeout(
                        quiniela_original, partidos, reglas_violadas, temperatura
                    )
                    
                    if correccion_candidata is None:
                        self.instrumentor.end_timer(attempt_timer, success=False)
                        continue
                    
                    # Validar safeguards: calcular distancia Hamming
                    distancia = self._calcular_distancia_hamming(
                        quiniela_original["resultados"],
                        correccion_candidata["resultados"]
                    )
                    
                    self.logger.debug(f"Intento {intento + 1}: distancia Hamming = {distancia}")
                    
                    # Verificar límites de seguridad
                    if distancia > self.max_hamming_distance:
                        self.logger.warning(f"⚠️ Corrección rechazada: distancia {distancia} > límite {self.max_hamming_distance}")
                        self.usage_stats["rejected_by_hamming"] += 1
                        self.instrumentor.end_timer(attempt_timer, success=False, metrics={
                            "rejection_reason": "hamming_distance_exceeded",
                            "distance": distancia
                        })
                        continue
                    
                    # Validar que la corrección efectivamente mejora
                    problemas_nuevos = self._detectar_problemas_quiniela(correccion_candidata)
                    if len(problemas_nuevos) >= len(reglas_violadas):
                        self.logger.warning(f"⚠️ Corrección no mejora: {len(problemas_nuevos)} problemas vs {len(reglas_violadas)} originales")
                        self.instrumentor.end_timer(attempt_timer, success=False)
                        continue
                    
                    # Esta corrección es válida, evaluar si es la mejor
                    if distancia < menor_distancia:
                        mejor_correccion = correccion_candidata
                        menor_distancia = distancia
                    
                    self.instrumentor.end_timer(attempt_timer, success=True, metrics={
                        "hamming_distance": distancia,
                        "problems_before": len(reglas_violadas),
                        "problems_after": len(problemas_nuevos)
                    })
                    
                    # Si encontramos una corrección perfecta, salir temprano
                    if len(problemas_nuevos) == 0:
                        break
                        
                except Exception as e:
                    self.instrumentor.end_timer(attempt_timer, success=False)
                    self.logger.error(f"Error en intento {intento + 1}: {e}")
                    continue
            
            # Evaluar resultado final
            if mejor_correccion is not None:
                # Log de cambios realizados para auditoría
                self._log_cambios_realizados(quiniela_original, mejor_correccion)
                
                self.usage_stats["successful_corrections"] += 1
                self.instrumentor.end_timer(timer_id, success=True, metrics={
                    "final_hamming_distance": menor_distancia,
                    "successful_attempts": sum(1 for i in range(self.max_retries) if mejor_correccion)
                })
                
                self.logger.info(f"✅ {quiniela_id} corregida exitosamente (distancia: {menor_distancia})")
                return mejor_correccion
            else:
                self.logger.warning(f"⚠️ No se pudo corregir {quiniela_id} dentro de los límites de seguridad")
                self.instrumentor.end_timer(timer_id, success=False)
                return None
                
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"❌ Error en corrección con safeguards para {quiniela_id}: {e}")
            return None
    
    def _ejecutar_correccion_con_timeout(self, quiniela: Dict[str, Any], 
                                        partidos: List[Dict[str, Any]], 
                                        reglas_violadas: List[str], 
                                        temperatura: float) -> Optional[Dict[str, Any]]:
        """
        Ejecuta corrección individual con timeout y parsing robusto
        """
        try:
            # Preparar contexto optimizado
            contexto = self._preparar_contexto_safeguards(quiniela, partidos, reglas_violadas)
            prompt_hash = hashlib.md5(contexto.encode()).hexdigest()[:8]
            
            # Llamada a OpenAI con timeout
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Modelo más rápido y barato
                messages=[
                    {"role": "system", "content": self._get_system_prompt_safeguards()},
                    {"role": "user", "content": contexto}
                ],
                temperature=temperatura,
                max_tokens=400,  # Límite reducido para forzar respuestas concisas
                timeout=self.timeout_seconds
            )
            
            ai_response = response.choices[0].message.content
            response_hash = hashlib.md5(ai_response.encode()).hexdigest()[:8]
            
            # Log de interacción para debug
            self.instrumentor.log_ai_interaction(
                operation="correccion_individual",
                prompt_hash=prompt_hash,
                model="gpt-4o-mini",
                temperature=temperatura,
                success=True,
                response_hash=response_hash
            )
            
            # Parsing robusto con múltiples métodos
            resultado_parseado = self._parsing_robusto_respuesta(ai_response, quiniela)
            
            return resultado_parseado
            
        except Exception as e:
            self.usage_stats["timeouts"] += 1
            self.logger.error(f"Error en ejecución con timeout: {e}")
            return None
    
    def _preparar_contexto_safeguards(self, quiniela: Dict[str, Any], 
                                    partidos: List[Dict[str, Any]], 
                                    reglas_violadas: List[str]) -> str:
        """
        Prepara contexto optimizado con énfasis en cambios mínimos
        """
        # Información de partidos crítica
        info_partidos = []
        anclas_indices = []
        
        for i, partido in enumerate(partidos):
            clasificacion = partido.get("clasificacion", "Neutro")
            if clasificacion == "Ancla":
                anclas_indices.append(i+1)
                info_partidos.append(f"P{i+1}: {partido['home'][:10]} vs {partido['away'][:10]} [ANCLA] ⚠️ NUNCA CAMBIAR")
            else:
                info_partidos.append(f"P{i+1}: {partido['home'][:10]} vs {partido['away'][:10]} [{clasificacion}]")
        
        # Análisis específico de problemas
        problemas_detallados = []
        resultados_actuales = quiniela['resultados']
        
        for regla in reglas_violadas:
            if "empates" in regla.lower():
                empates_actual = quiniela['empates']
                if empates_actual < 4:
                    problemas_detallados.append(f"CRÍTICO: Solo {empates_actual} empates, necesita mínimo 4")
                elif empates_actual > 6:
                    problemas_detallados.append(f"CRÍTICO: {empates_actual} empates, máximo permitido 6")
            elif "concentracion" in regla.lower():
                for signo in ['L', 'E', 'V']:
                    count = quiniela['distribución'][signo]
                    if count > 9:
                        problemas_detallados.append(f"CRÍTICO: {signo} aparece {count} veces (>70% concentración)")
        
        contexto = f"""TAREA CRÍTICA: Corregir quiniela {quiniela['id']} con CAMBIOS MÍNIMOS

⚠️ RESTRICCIÓN ABSOLUTA: Máximo {self.max_hamming_distance} partidos cambiados de 14 total

PROBLEMAS A CORREGIR:
{chr(10).join(f"- {p}" for p in problemas_detallados)}

QUINIELA ACTUAL: {','.join(resultados_actuales)}
Distribución: L:{quiniela['distribución']['L']}, E:{quiniela['distribución']['E']}, V:{quiniela['distribución']['V']}

PARTIDOS (NUNCA cambiar los marcados ⚠️):
{chr(10).join(info_partidos[:7])}  # Solo primeros 7 para ahorrar tokens

ESTRATEGIA OBLIGATORIA:
1. Hacer el MÍNIMO de cambios posible (máximo {self.max_hamming_distance})
2. NUNCA tocar partidos ANCLA (posiciones: {anclas_indices})
3. Priorizar cambios que resuelvan múltiples problemas
4. Mantener empates entre 4-6
5. Evitar concentración >70%

FORMATO DE RESPUESTA (solo JSON, sin explicaciones):
{{"resultados": ["L","E","V","L","E","V","L","E","V","L","E","V","L","E"]}}

IMPORTANTE: La respuesta debe tener exactamente 14 elementos."""

        return contexto
    
    def _get_system_prompt_safeguards(self) -> str:
        """System prompt optimizado para correcciones con safeguards"""
        return """Eres un experto en optimización de quinielas Progol. Tu tarea es hacer correcciones MÍNIMAS y PRECISAS.

REGLAS CRÍTICAS:
1. NUNCA cambies más de 8 partidos de los 14 totales
2. NUNCA toques partidos marcados como ANCLA
3. Haz solo los cambios estrictamente necesarios
4. Empates válidos: 4-6 por quiniela
5. Concentración máxima: 9 de cualquier signo (L/E/V)

FORMATO OBLIGATORIO:
{"resultados": ["L","E","V",...]} con exactamente 14 elementos

¡RESPONDE SOLO CON EL JSON, SIN EXPLICACIONES!"""
    
    def _parsing_robusto_respuesta(self, respuesta: str, quiniela_original: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parsing robusto que prueba múltiples métodos de extracción
        """
        respuesta_limpia = respuesta.strip()
        
        # Método 1: JSON completo
        try:
            json_match = re.search(r'\{[^}]*"resultados"[^}]*\[[^\]]*\][^}]*\}', respuesta_limpia, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "resultados" in data and len(data["resultados"]) == 14:
                    resultados = [str(r).upper() for r in data["resultados"]]
                    if all(r in ['L', 'E', 'V'] for r in resultados):
                        return self._crear_quiniela_corregida(resultados, quiniela_original)
        except json.JSONDecodeError:
            pass
        
        # Método 2: Array de resultados
        try:
            array_match = re.search(r'\[([^\]]+)\]', respuesta_limpia)
            if array_match:
                array_content = array_match.group(1)
                # Extraer solo L, E, V válidos
                resultados = re.findall(r'[LEV]', array_content.upper())
                if len(resultados) == 14:
                    return self._crear_quiniela_corregida(resultados, quiniela_original)
        except:
            pass
        
        # Método 3: Secuencia separada por comas o espacios
        try:
            # Buscar patrón de 14 L/E/V
            sequence_matches = re.findall(r'[LEV]', respuesta_limpia.upper())
            if len(sequence_matches) >= 14:
                resultados = sequence_matches[:14]
                return self._crear_quiniela_corregida(resultados, quiniela_original)
        except:
            pass
        
        # Método 4: Fallback - buscar cualquier secuencia de 14 caracteres válidos
        try:
            clean_response = re.sub(r'[^LEV]', '', respuesta_limpia.upper())
            if len(clean_response) >= 14:
                resultados = list(clean_response[:14])
                return self._crear_quiniela_corregida(resultados, quiniela_original)
        except:
            pass
        
        self.logger.warning(f"⚠️ No se pudo parsear respuesta IA: {respuesta_limpia[:100]}...")
        return None
    
    def _crear_quiniela_corregida(self, resultados: List[str], quiniela_original: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea quiniela corregida preservando metadata original
        """
        quiniela_corregida = copy.deepcopy(quiniela_original)
        quiniela_corregida["resultados"] = resultados
        quiniela_corregida["empates"] = resultados.count("E")
        quiniela_corregida["distribución"] = {
            "L": resultados.count("L"),
            "E": resultados.count("E"),
            "V": resultados.count("V")
        }
        
        # Marcar como corregida por IA para trazabilidad
        quiniela_corregida["_ai_corrected"] = True
        quiniela_corregida["_correction_timestamp"] = time.time()
        
        return quiniela_corregida
    
    def _calcular_distancia_hamming(self, resultados_a: List[str], resultados_b: List[str]) -> int:
        """
        Calcula distancia de Hamming entre dos secuencias de resultados
        """
        if len(resultados_a) != len(resultados_b):
            return float('inf')
        
        return sum(1 for a, b in zip(resultados_a, resultados_b) if a != b)
    
    def _log_cambios_realizados(self, original: Dict[str, Any], corregida: Dict[str, Any]):
        """
        Log detallado de cambios para auditoría
        """
        cambios = []
        resultados_orig = original["resultados"]
        resultados_corr = corregida["resultados"]
        
        for i, (orig, corr) in enumerate(zip(resultados_orig, resultados_corr)):
            if orig != corr:
                cambios.append(f"P{i+1}: {orig}→{corr}")
        
        self.logger.info(f"🔄 Cambios realizados en {original['id']}: {', '.join(cambios) if cambios else 'ninguno'}")
        
        # Log estructurado para análisis
        self.instrumentor.log_state_change(
            component="ai_correction",
            old_state={
                "quiniela_id": original["id"],
                "resultados": resultados_orig,
                "empates": original["empates"],
                "distribución": original["distribución"]
            },
            new_state={
                "quiniela_id": corregida["id"],
                "resultados": resultados_corr,
                "empates": corregida["empates"],
                "distribución": corregida["distribución"],
                "cambios_realizados": cambios
            }
        )
    
    def optimizar_distribucion_global_con_safeguards(self, portafolio: List[Dict[str, Any]], 
                                                    partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimización global del portafolio con límites estrictos de cambios
        """
        if not self.enabled or len(portafolio) < 10:
            return portafolio
        
        timer_id = self.instrumentor.start_timer("ai_optimizacion_global")
        
        try:
            self.logger.info("🌐 Iniciando optimización global con safeguards...")
            
            # Analizar problemas actuales
            problemas_globales = self._analizar_problemas_globales(portafolio)
            
            if not problemas_globales:
                self.logger.info("✅ Portafolio no requiere optimización global")
                self.instrumentor.end_timer(timer_id, success=True, metrics={"action": "no_optimization_needed"})
                return portafolio
            
            # Calcular presupuesto de cambios globales
            cambios_permitidos = min(self.max_global_changes, len(portafolio) // 3)
            
            self.logger.info(f"📊 Problemas detectados: {len(problemas_globales)}, cambios permitidos: {cambios_permitidos}")
            
            # Generar plan de optimización
            plan_optimizacion = self._generar_plan_optimizacion_global(
                portafolio, partidos, problemas_globales, cambios_permitidos
            )
            
            if not plan_optimizacion:
                self.logger.warning("⚠️ No se pudo generar plan de optimización válido")
                self.instrumentor.end_timer(timer_id, success=False)
                return portafolio
            
            # Aplicar plan con validación continua
            portafolio_optimizado = self._aplicar_plan_con_rollback(
                portafolio, plan_optimizacion, partidos
            )
            
            # Validar que la optimización mejoró realmente
            problemas_finales = self._analizar_problemas_globales(portafolio_optimizado)
            
            if len(problemas_finales) < len(problemas_globales):
                self.logger.info(f"✅ Optimización global exitosa: {len(problemas_globales)} → {len(problemas_finales)} problemas")
                
                self.instrumentor.end_timer(timer_id, success=True, metrics={
                    "problems_before": len(problemas_globales),
                    "problems_after": len(problemas_finales),
                    "changes_applied": len(plan_optimizacion)
                })
                
                return portafolio_optimizado
            else:
                self.logger.warning("⚠️ Optimización global no mejoró, revirtiendo cambios")
                self.usage_stats["rollbacks_performed"] += 1
                self.instrumentor.end_timer(timer_id, success=False)
                return portafolio
                
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"❌ Error en optimización global: {e}")
            return portafolio
    
    def _requiere_correccion(self, quiniela: Dict[str, Any], reglas_violadas: List[str]) -> bool:
        """
        Verifica si la quiniela realmente requiere corrección IA
        """
        # Si no hay reglas violadas, no se requiere corrección
        if not reglas_violadas:
            return False
        
        # Verificar si las violaciones son menores y pueden ser ignoradas
        violaciones_menores = ["distribución_equilibrada", "correlación_jaccard"]
        violaciones_serias = [r for r in reglas_violadas if not any(menor in r.lower() for menor in violaciones_menores)]
        
        return len(violaciones_serias) > 0
    
    def _detectar_problemas_quiniela(self, quiniela: Dict[str, Any]) -> List[str]:
        """
        Detecta problemas específicos en una quiniela individual
        """
        problemas = []
        
        # Verificar empates
        empates = quiniela.get("empates", 0)
        if empates < 4:
            problemas.append(f"empates_insuficientes_{empates}")
        elif empates > 6:
            problemas.append(f"empates_exceso_{empates}")
        
        # Verificar concentración
        if "distribución" in quiniela:
            distribución = quiniela["distribución"]
            max_conc = max(distribución.values())
            
            if max_conc > 9:  # >70% de 14
                signo_concentrado = max(distribución, key=distribución.get)
                problemas.append(f"concentracion_excesiva_{signo_concentrado}_{max_conc}")
        
        # Verificar concentración inicial
        if "resultados" in quiniela and len(quiniela["resultados"]) >= 3:
            primeros_3 = quiniela["resultados"][:3]
            for signo in ["L", "E", "V"]:
                if primeros_3.count(signo) > 2:  # >60% de 3
                    problemas.append(f"concentracion_inicial_{signo}_{primeros_3.count(signo)}")
        
        return problemas
    
    def _analizar_problemas_globales(self, portafolio: List[Dict[str, Any]]) -> List[str]:
        """
        Analiza problemas a nivel de todo el portafolio
        """
        problemas = []
        
        if not portafolio:
            return ["portafolio_vacio"]
        
        # Calcular distribución global
        total_L = sum(q.get("distribución", {}).get("L", 0) for q in portafolio)
        total_E = sum(q.get("distribución", {}).get("E", 0) for q in portafolio)
        total_V = sum(q.get("distribución", {}).get("V", 0) for q in portafolio)
        total = total_L + total_E + total_V
        
        if total > 0:
            porc_L = total_L / total
            porc_E = total_E / total
            porc_V = total_V / total
            
            # Verificar rangos históricos
            if not (0.35 <= porc_L <= 0.41):
                problemas.append(f"distribucion_L_fuera_rango_{porc_L:.3f}")
            if not (0.25 <= porc_E <= 0.33):
                problemas.append(f"distribucion_E_fuera_rango_{porc_E:.3f}")
            if not (0.30 <= porc_V <= 0.36):
                problemas.append(f"distribucion_V_fuera_rango_{porc_V:.3f}")
        
        # Verificar arquitectura
        cores = len([q for q in portafolio if q.get("tipo") == "Core"])
        satelites = len([q for q in portafolio if q.get("tipo") == "Satelite"])
        
        if cores != 4:
            problemas.append(f"cores_incorrectos_{cores}")
        if satelites != 26:
            problemas.append(f"satelites_incorrectos_{satelites}")
        
        return problemas
    
    def _generar_plan_optimizacion_global(self, portafolio: List[Dict[str, Any]], 
                                        partidos: List[Dict[str, Any]], 
                                        problemas: List[str], 
                                        max_cambios: int) -> List[Dict[str, Any]]:
        """
        Genera plan de optimización limitado por presupuesto de cambios
        """
        # Por simplicidad, generar plan heurístico básico
        # En implementación completa, esto usaría IA para generar el plan
        
        plan = []
        cambios_usados = 0
        
        # Priorizar problemas por severidad
        problemas_priorizados = sorted(problemas, key=lambda p: (
            "distribucion" in p,  # Distribución es más crítica
            "cores" in p or "satelites" in p,  # Arquitectura es crítica
            "concentracion" in p  # Concentración es moderada
        ), reverse=True)
        
        for problema in problemas_priorizados[:3]:  # Solo top 3 problemas
            if cambios_usados >= max_cambios:
                break
            
            if "distribucion_L" in problema and "fuera_rango" in problema:
                # Buscar quinielas satélite para ajustar L
                candidatos = [i for i, q in enumerate(portafolio) 
                             if q.get("tipo") == "Satelite" and cambios_usados < max_cambios]
                
                for candidato_idx in candidatos[:2]:  # Máximo 2 cambios
                    plan.append({
                        "accion": "ajustar_distribucion_L",
                        "quiniela_idx": candidato_idx,
                        "cambios_estimados": 2
                    })
                    cambios_usados += 2
        
        return plan
    
    def _aplicar_plan_con_rollback(self, portafolio: List[Dict[str, Any]], 
                                  plan: List[Dict[str, Any]], 
                                  partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aplica plan de optimización con capacidad de rollback
        """
        portafolio_trabajando = copy.deepcopy(portafolio)
        cambios_aplicados = []
        
        for accion in plan:
            try:
                # Crear backup antes del cambio
                estado_previo = copy.deepcopy(portafolio_trabajando[accion["quiniela_idx"]])
                
                # Aplicar acción específica
                if accion["accion"] == "ajustar_distribucion_L":
                    self._ajustar_distribucion_l_quiniela(
                        portafolio_trabajando[accion["quiniela_idx"]], 
                        partidos
                    )
                
                # Registrar cambio para posible rollback
                cambios_aplicados.append({
                    "quiniela_idx": accion["quiniela_idx"],
                    "estado_previo": estado_previo,
                    "accion": accion["accion"]
                })
                
            except Exception as e:
                self.logger.error(f"Error aplicando acción {accion}: {e}")
                # Rollback de todos los cambios aplicados
                for cambio in reversed(cambios_aplicados):
                    portafolio_trabajando[cambio["quiniela_idx"]] = cambio["estado_previo"]
                return portafolio  # Retornar original en caso de error
        
        return portafolio_trabajando
    
    def _ajustar_distribucion_l_quiniela(self, quiniela: Dict[str, Any], partidos: List[Dict[str, Any]]):
        """
        Ajusta una quiniela específica para mejorar distribución de L
        """
        resultados = quiniela["resultados"]
        
        # Buscar partidos modificables (no anclas) que se puedan cambiar a L
        modificables = []
        for i, resultado in enumerate(resultados):
            if (resultado != "L" and 
                partidos[i].get("clasificacion") != "Ancla" and
                partidos[i]["prob_local"] > 0.3):  # Solo si L tiene chance razonable
                modificables.append(i)
        
        # Cambiar hasta 2 partidos a L
        cambios_realizados = 0
        for i in modificables[:2]:
            if cambios_realizados < 2:
                resultados[i] = "L"
                cambios_realizados += 1
        
        # Actualizar metadata
        quiniela["resultados"] = resultados
        quiniela["empates"] = resultados.count("E")
        quiniela["distribución"] = {
            "L": resultados.count("L"),
            "E": resultados.count("E"),
            "V": resultados.count("V")
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de uso del sistema IA
        """
        return {
            **self.usage_stats,
            "enabled": self.enabled,
            "api_key_configured": self.api_key is not None,
            "safeguards_config": {
                "max_hamming_distance": self.max_hamming_distance,
                "max_global_changes": self.max_global_changes,
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds
            }
        }


# Clase de compatibilidad hacia atrás
class ProgolAIAssistant(EnhancedProgolAIAssistant):
    """Wrapper para mantener compatibilidad con código existente"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.logger.warning("⚠️ Usando ProgolAIAssistant legacy - migrar a EnhancedProgolAIAssistant")
    
    def corregir_quiniela_invalida(self, quiniela: Dict[str, Any], 
                                  partidos: List[Dict[str, Any]], 
                                  reglas_violadas: List[str]) -> Optional[Dict[str, Any]]:
        """Wrapper para mantener compatibilidad de método"""
        return self.corregir_quiniela_con_safeguards(quiniela, partidos, reglas_violadas)
    
    def optimizar_distribucion_global(self, portafolio: List[Dict[str, Any]], 
                                    partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Wrapper para mantener compatibilidad de método"""
        return self.optimizar_distribucion_global_con_safeguards(portafolio, partidos)