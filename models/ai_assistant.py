# progol_optimizer/models/ai_assistant.py
"""
Asistente AI para corrección y optimización de quinielas usando OpenAI API
Con conocimiento completo de la Metodología Definitiva Progol
VERSIÓN MEJORADA: Con debug logging completo y mejor parsing
"""

import os
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

class ProgolAIAssistant:
    """
    Asistente inteligente que usa GPT-4 con conocimiento completo de la metodología Progol
    MEJORADO: Con logging detallado para debug completo
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Intentar obtener API key en orden de prioridad:
        # 1. Parámetro directo
        # 2. Variable de entorno
        # 3. Streamlit secrets
        self.api_key = api_key
        
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            try:
                import streamlit as st
                if "OPENAI_API_KEY" in st.secrets:
                    self.api_key = st.secrets["OPENAI_API_KEY"]
            except:
                pass
        
        if not self.api_key:
            self.logger.warning("⚠️ OpenAI API key no configurada. Funcionalidad AI deshabilitada.")
            self.enabled = False
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.enabled = True
                self.logger.info("✅ Asistente AI inicializado correctamente")
                
                # Inicializar tracking de debug
                self._init_debug_tracking()
                
            except Exception as e:
                self.logger.error(f"Error inicializando OpenAI: {e}")
                self.enabled = False
    
    def _init_debug_tracking(self):
        """Inicializa el tracking de debug en session_state"""
        try:
            import streamlit as st
            
            if 'ai_debug_responses' not in st.session_state:
                st.session_state.ai_debug_responses = []
            
            if 'ai_usage_stats' not in st.session_state:
                st.session_state.ai_usage_stats = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'parsing_failures': 0
                }
        except:
            pass  # Si no estamos en Streamlit, ignorar
    
    def _log_ai_interaction(self, quiniela_id: str, problemas: List[str], prompt: str, 
                           ai_response: str, success: bool, parsed_result: Any = None):
        """Registra interacción con AI para debug"""
        try:
            import streamlit as st
            
            interaction = {
                'timestamp': time.strftime('%H:%M:%S'),
                'quiniela_id': quiniela_id,
                'problemas': problemas,
                'prompt': prompt[:500] + "..." if len(prompt) > 500 else prompt,  # Truncar prompts largos
                'ai_response': ai_response,
                'success': success,
                'parsed_result': str(parsed_result) if parsed_result else None
            }
            
            st.session_state.ai_debug_responses.append(interaction)
            
            # Mantener solo los últimos 20 para no consumir mucha memoria
            if len(st.session_state.ai_debug_responses) > 20:
                st.session_state.ai_debug_responses = st.session_state.ai_debug_responses[-20:]
            
            # Actualizar estadísticas
            stats = st.session_state.ai_usage_stats
            stats['total_calls'] += 1
            if success:
                stats['successful_calls'] += 1
            else:
                stats['failed_calls'] += 1
                if parsed_result is None:
                    stats['parsing_failures'] += 1
                    
        except Exception as e:
            self.logger.debug(f"Error logging AI interaction: {e}")

    def corregir_quiniela_invalida(self, quiniela: Dict[str, Any], partidos: List[Dict[str, Any]], 
                                  reglas_violadas: List[str]) -> Optional[Dict[str, Any]]:
        """
        Usa GPT-4 para corregir una quiniela que viola reglas - CON DEBUG COMPLETO
        """
        if not self.enabled:
            return None
            
        quiniela_id = quiniela.get('id', 'Unknown')
        
        try:
            # Preparar contexto MEJORADO
            contexto = self._preparar_contexto_correccion_mejorado(quiniela, partidos, reglas_violadas)
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self._get_system_prompt_metodologia_completa()},
                    {"role": "user", "content": contexto}
                ],
                temperature=0.2,  # Más determinístico
                max_tokens=600
            )
            
            ai_response_text = response.choices[0].message.content
            
            # Parsear respuesta con múltiples métodos
            resultado = self._parsear_respuesta_correccion_mejorado(ai_response_text, quiniela)
            
            success = resultado is not None
            
            # Log SIEMPRE para debug
            self._log_ai_interaction(quiniela_id, reglas_violadas, contexto, 
                                   ai_response_text, success, resultado)
            
            if resultado:
                self.logger.info(f"✅ Quiniela {quiniela_id} corregida exitosamente por AI")
                return resultado
            else:
                self.logger.warning(f"⚠️ No se pudo parsear la corrección de AI para {quiniela_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error en corrección AI para {quiniela_id}: {e}")
            
            # Log error para debug
            self._log_ai_interaction(quiniela_id, reglas_violadas, "ERROR", str(e), False)
            
            return None

    def _preparar_contexto_correccion_mejorado(self, quiniela: Dict[str, Any], 
                                             partidos: List[Dict[str, Any]], 
                                             reglas_violadas: List[str]) -> str:
        """Contexto MEJORADO más específico"""
        
        # Información de partidos más clara
        info_partidos = []
        anclas_indices = []
        
        for i, partido in enumerate(partidos):
            clasificacion = partido.get("clasificacion", "Neutro")
            es_ancla = clasificacion == "Ancla"
            
            if es_ancla:
                anclas_indices.append(i+1)
            
            info = f"P{i+1}: {partido['home'][:12]} vs {partido['away'][:12]}"
            info += f" [{clasificacion}]"
            info += f" (L:{partido['prob_local']:.2f}, E:{partido['prob_empate']:.2f}, V:{partido['prob_visitante']:.2f})"
            
            if es_ancla:
                info += " ⚠️ NUNCA CAMBIAR"
                
            info_partidos.append(info)
        
        # Análisis ESPECÍFICO de problemas
        problemas_detallados = []
        resultados_actuales = quiniela['resultados']
        
        for regla in reglas_violadas:
            if "empates" in regla:
                empates_actual = quiniela['empates']
                if empates_actual < 4:
                    problemas_detallados.append(f"- EMPATES: tiene {empates_actual}, necesita al menos 4")
                elif empates_actual > 6:
                    problemas_detallados.append(f"- EMPATES: tiene {empates_actual}, máximo permitido 6")
            elif "concentracion" in regla:
                # Detectar QUÉ signo está concentrado
                for signo in ['L', 'E', 'V']:
                    count = quiniela['distribución'][signo]
                    if count > 9:  # >70%
                        problemas_detallados.append(f"- CONCENTRACIÓN: Demasiados {signo}: {count}/14 (>70%)")
        
        contexto = f"""TAREA: Corregir quiniela {quiniela['id']} ({quiniela['tipo']})

PROBLEMAS ESPECÍFICOS:
{chr(10).join(problemas_detallados)}

QUINIELA ACTUAL: {','.join(resultados_actuales)}
Distribución: L:{quiniela['distribución']['L']}, E:{quiniela['distribución']['E']}, V:{quiniela['distribución']['V']}

PARTIDOS (NUNCA cambiar los marcados con ⚠️):
{chr(10).join(info_partidos)}

REGLAS OBLIGATORIAS:
1. NUNCA cambiar partidos Ancla (posiciones: {anclas_indices})
2. Debe tener entre 4-6 empates
3. Máximo 9 de cualquier signo (≤70% de 14)
4. Máximo 2 iguales en primeros 3 partidos

ESTRATEGIA DE CORRECCIÓN:
- Haz el MÍNIMO de cambios necesarios
- Prioriza cambiar partidos con probabilidades bajas del resultado actual
- Si hay demasiados L, cambia algunos L por E o V en partidos donde L tenga baja probabilidad

RESPUESTA REQUERIDA (solo JSON, sin explicaciones):
{{"resultados": ["L", "E", "V", "L", "E", "V", "L", "E", "V", "L", "E", "V", "L", "E"]}}"""

        return contexto

    def _parsear_respuesta_correccion_mejorado(self, respuesta: str, quiniela_original: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parsing MEJORADO con múltiples patrones de fallback"""
        
        # Limpiar respuesta
        respuesta_limpia = respuesta.strip()
        
        # Método 1: JSON completo
        try:
            json_pattern = r'\{[^}]*"resultados"[^}]*\[[^\]]*\][^}]*\}'
            json_match = re.search(json_pattern, respuesta_limpia, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                if "resultados" in data and len(data["resultados"]) == 14:
                    resultados = [str(r).upper() for r in data["resultados"]]
                    if all(r in ['L', 'E', 'V'] for r in resultados):
                        return self._crear_quiniela_corregida(resultados, quiniela_original)
        except:
            pass
        
        # Método 2: Array de resultados
        try:
            array_pattern = r'\[([^]]*)\]'
            array_match = re.search(array_pattern, respuesta_limpia)
            
            if array_match:
                array_content = array_match.group(1)
                # Extraer solo L, E, V
                resultados = re.findall(r'[LEV]', array_content.upper())
                if len(resultados) == 14:
                    return self._crear_quiniela_corregida(resultados, quiniela_original)
        except:
            pass
        
        # Método 3: Secuencia separada por comas
        try:
            # Buscar patrón L,E,V,L,E,V...
            sequence_pattern = r'[LEV](?:\s*,\s*[LEV]){13}'
            sequence_match = re.search(sequence_pattern, respuesta_limpia.upper())
            
            if sequence_match:
                sequence = sequence_match.group()
                resultados = re.findall(r'[LEV]', sequence)
                if len(resultados) == 14:
                    return self._crear_quiniela_corregida(resultados, quiniela_original)
        except:
            pass
        
        # Método 4: Extraer todas las L/E/V y tomar las primeras 14
        try:
            all_letters = re.findall(r'[LEV]', respuesta_limpia.upper())
            if len(all_letters) >= 14:
                resultados = all_letters[:14]
                return self._crear_quiniela_corregida(resultados, quiniela_original)
        except:
            pass
        
        return None
    
    def _crear_quiniela_corregida(self, resultados: List[str], quiniela_original: Dict[str, Any]) -> Dict[str, Any]:
        """Crea la quiniela corregida con estructura completa"""
        quiniela_corregida = quiniela_original.copy()
        quiniela_corregida["resultados"] = resultados
        quiniela_corregida["empates"] = resultados.count("E")
        quiniela_corregida["distribución"] = {
            "L": resultados.count("L"),
            "E": resultados.count("E"),
            "V": resultados.count("V")
        }
        return quiniela_corregida

    def optimizar_distribucion_global(self, portafolio: List[Dict[str, Any]], 
                                    partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Usa GPT-4 para optimizar la distribución global del portafolio
        """
        if not self.enabled or len(portafolio) < 10:
            return portafolio
            
        try:
            # Analizar problemas actuales
            problemas = self._analizar_problemas_portafolio(portafolio)
            
            if not problemas:
                return portafolio
            
            # Pedir sugerencias a GPT-4
            contexto = self._preparar_contexto_optimizacion(portafolio, partidos, problemas)
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self._get_system_prompt_metodologia_completa()},
                    {"role": "user", "content": contexto}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            # Log para debug
            self._log_ai_interaction("GLOBAL", problemas, contexto, 
                                   response.choices[0].message.content, True, "Optimización global")
            
            # Aplicar sugerencias
            portafolio_optimizado = self._aplicar_sugerencias_optimizacion(
                response.choices[0].message.content, portafolio, partidos
            )
            
            return portafolio_optimizado
            
        except Exception as e:
            self.logger.error(f"Error en optimización AI: {e}")
            return portafolio

    def _get_system_prompt_metodologia_completa(self) -> str:
        """Prompt con la metodología completa del documento técnico"""
        return """Eres un experto en la Metodología Definitiva Progol, basada en el documento técnico de 7 partes.

CONTEXTO HISTÓRICO Y ESTADÍSTICO:
- Distribución histórica de 1,497 concursos: 38% Locales, 29% Empates, 33% Visitantes
- Promedio histórico: 4.33 empates por quiniela
- Los rangos aceptables son: L[35-41%], E[25-33%], V[30-36%]

ARQUITECTURA DEL PORTAFOLIO (OBLIGATORIA):
1. Total: 30 quinielas = 4 Core + 26 Satélites
2. Quinielas Core (4):
   - Fijan el resultado de máxima probabilidad en partidos Ancla
   - Los 4 Core deben tener idénticos resultados en partidos Ancla
   - Representan la base conservadora del portafolio
3. Satélites (26 en 13 pares):
   - Cada par invierte los resultados en partidos Divisores
   - Correlación Jaccard entre pares debe ser ≤ 0.57
   - Generan diversidad controlada

TAXONOMÍA DE PARTIDOS:
1. ANCLA: p_max > 60% con alta confianza
   - NUNCA se modifican en ninguna quiniela
   - Todos los Core y Satélites deben tener el mismo resultado
2. DIVISOR: 40% < p_max < 60% o con volatilidad
   - Se invierten entre pares de satélites
   - Son la fuente principal de diversificación
3. TENDENCIA EMPATE: Empate es favorito o equipos muy parejos
   - Priorizar empate cuando sea razonable
4. NEUTRO: Resto de partidos

REGLAS INDIVIDUALES (TODAS OBLIGATORIAS):
1. Exactamente 14 resultados por quiniela
2. Entre 4-6 empates por quiniela (rango estricto)
3. Concentración máxima 70% del mismo signo (máximo 9 de 14)
4. Concentración máxima 60% en primeros 3 partidos (máximo 2 de 3)
5. Los partidos ANCLA nunca se modifican

CUANDO CORRIJAS:
- Haz el MÍNIMO de cambios necesarios
- NUNCA toques partidos Ancla
- Prioriza cambiar resultados con baja probabilidad
- Mantén la coherencia estadística

FORMATO DE RESPUESTA:
{"resultados": ["L", "E", "V", "L", "E", "V", "L", "E", "V", "L", "E", "V", "L", "E"]}

¡RESPONDE SOLO CON EL JSON, SIN EXPLICACIONES!"""

    def _preparar_contexto_optimizacion(self, portafolio: List[Dict[str, Any]], 
                                      partidos: List[Dict[str, Any]], problemas: List[str]) -> str:
        """Prepara contexto para optimización global con información de arquitectura"""
        
        # Análisis de arquitectura
        cores = [q for q in portafolio if q["tipo"] == "Core"]
        satelites = [q for q in portafolio if q["tipo"] == "Satelite"]
        
        # Análisis de distribución
        total_L = sum(q["distribución"]["L"] for q in portafolio)
        total_E = sum(q["distribución"]["E"] for q in portafolio)
        total_V = sum(q["distribución"]["V"] for q in portafolio)
        total = total_L + total_E + total_V
        
        # Análisis de partidos
        anclas = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        divisores = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Divisor"]
        
        # Quinielas más problemáticas
        quinielas_criticas = []
        for q in portafolio:
            problemas_q = []
            max_conc = max(q["distribución"].values()) / 14
            
            if max_conc > 0.70:
                problemas_q.append(f"concentración {max_conc:.1%}")
            if q["empates"] < 4 or q["empates"] > 6:
                problemas_q.append(f"empates={q['empates']}")
                
            if problemas_q:
                quinielas_criticas.append(f"{q['id']}: {', '.join(problemas_q)}")
        
        contexto = f"""TAREA: Optimizar portafolio completo de 30 quinielas

ARQUITECTURA ACTUAL:
- {len(cores)} Core + {len(satelites)} Satélites (debe ser 4 + 26)
- Partidos Ancla: {len(anclas)} en posiciones {[p+1 for p in anclas]}
- Partidos Divisores: {len(divisores)} en posiciones {[p+1 for p in divisores[:5]]}...

PROBLEMAS DETECTADOS:
{chr(10).join(f"- {p}" for p in problemas)}

DISTRIBUCIÓN ACTUAL vs OBJETIVO:
L: {total_L} ({(total_L/total)*100:.1f}%) - Objetivo: 35-41% (147-172 de 420)
E: {total_E} ({(total_E/total)*100:.1f}%) - Objetivo: 25-33% (105-139 de 420)
V: {total_V} ({(total_V/total)*100:.1f}%) - Objetivo: 30-36% (126-151 de 420)

QUINIELAS CRÍTICAS ({len(quinielas_criticas)} de 30):
{chr(10).join(quinielas_criticas[:10])}

INSTRUCCIONES PARA OPTIMIZACIÓN:
1. Sugiere cambios ESPECÍFICOS: ID de quiniela + posición + nuevo resultado
2. Prioriza cambios en Satélites (no tocar Core si es posible)
3. NUNCA sugieras cambiar partidos Ancla
4. Busca balancear la distribución por posición
5. Mantén correlación baja entre pares de satélites

Formato de respuesta esperado:
"Cambio 1: Sat-1A posición 5 cambiar de L a E (reduce exceso de L, balancea posición 5)
Cambio 2: Sat-3B posición 8 cambiar de V a L (aumenta L hacia objetivo)
..."

Sugiere entre 10-15 cambios específicos para corregir los problemas."""

        return contexto

    def _analizar_problemas_portafolio(self, portafolio: List[Dict[str, Any]]) -> List[str]:
        """Analiza qué problemas tiene el portafolio actual"""
        problemas = []
        
        # Calcular distribución global
        total_L = sum(q["distribución"]["L"] for q in portafolio)
        total_E = sum(q["distribución"]["E"] for q in portafolio)
        total_V = sum(q["distribución"]["V"] for q in portafolio)
        total = total_L + total_E + total_V
        
        if total == 0:
            return ["Portafolio vacío"]

        porc_L = total_L / total
        porc_E = total_E / total
        porc_V = total_V / total
        
        # Verificar rangos
        if not (0.35 <= porc_L <= 0.41):
            problemas.append(f"Distribución L fuera de rango: {porc_L:.1%} (debe ser 35-41%)")
        if not (0.25 <= porc_E <= 0.33):
            problemas.append(f"Distribución E fuera de rango: {porc_E:.1%} (debe ser 25-33%)")
        if not (0.30 <= porc_V <= 0.36):
            problemas.append(f"Distribución V fuera de rango: {porc_V:.1%} (debe ser 30-36%)")
            
        # Verificar concentraciones individuales
        quinielas_con_problemas = 0
        for q in portafolio:
            max_conc = max(q["distribución"].values()) / 14
            if max_conc > 0.70:
                quinielas_con_problemas += 1
                
        if quinielas_con_problemas > 0:
            problemas.append(f"{quinielas_con_problemas} quinielas con concentración >70%")
            
        return problemas

    def _aplicar_sugerencias_optimizacion(self, sugerencias: str, portafolio: List[Dict[str, Any]], 
                                         partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aplica las sugerencias de optimización de forma segura"""
        portafolio_optimizado = [q.copy() for q in portafolio]
        
        try:
            # Buscar patrones de cambios sugeridos
            # Patrón: "Sat-1A posición 5 cambiar a E"
            patron_cambio = r"(Sat-\d+[AB]|Core-\d+).*?posici[óo]n\s*(\d+).*?cambiar\s*(?:de\s*[LEV]\s*)?a\s*([LEV])"
            cambios = re.findall(patron_cambio, sugerencias, re.IGNORECASE)
            
            cambios_aplicados = 0
            for quiniela_id, posicion_str, nuevo_resultado in cambios:
                try:
                    posicion = int(posicion_str) - 1  # Convertir a índice 0
                    
                    # Buscar la quiniela
                    for i, q in enumerate(portafolio_optimizado):
                        if q["id"] == quiniela_id and 0 <= posicion < 14:
                            # Verificar que no es ancla
                            if partidos[posicion].get("clasificacion") != "Ancla":
                                # Aplicar cambio
                                nuevos_resultados = q["resultados"].copy()
                                nuevos_resultados[posicion] = nuevo_resultado.upper()
                                
                                # Actualizar quiniela
                                portafolio_optimizado[i]["resultados"] = nuevos_resultados
                                portafolio_optimizado[i]["empates"] = nuevos_resultados.count("E")
                                portafolio_optimizado[i]["distribución"] = {
                                    "L": nuevos_resultados.count("L"),
                                    "E": nuevos_resultados.count("E"),
                                    "V": nuevos_resultados.count("V")
                                }
                                cambios_aplicados += 1
                                break
                                
                except Exception as e:
                    self.logger.debug(f"Error aplicando cambio individual: {e}")
                    continue
            
            self.logger.info(f"✅ Aplicados {cambios_aplicados} cambios sugeridos por AI")
            
        except Exception as e:
            self.logger.error(f"Error aplicando sugerencias: {e}")
            
        return portafolio_optimizado

# Añadir estas funciones al final de models/ai_assistant.py

    def debug_correccion_detallada(self, quiniela: Dict[str, Any], partidos: List[Dict[str, Any]], 
                                  reglas_violadas: List[str]) -> Dict[str, Any]:
        """
        NUEVA FUNCIÓN: Debug detallado paso a paso de corrección de IA
        """
        if not self.enabled:
            return {"status": "AI_DISABLED", "details": "OpenAI API key no disponible"}
        
        debug_info = {
            "quiniela_original": quiniela.copy(),
            "problemas_detectados": reglas_violadas,
            "prompt_enviado": "",
            "respuesta_ai_raw": "",
            "respuesta_parseada": None,
            "exito": False,
            "errores": []
        }
        
        try:
            # 1. Preparar contexto detallado
            contexto = self._preparar_contexto_debug_detallado(quiniela, partidos, reglas_violadas)
            debug_info["prompt_enviado"] = contexto
            
            # 2. Llamada a la IA con logging detallado
            self.logger.info(f"🤖 Enviando prompt a GPT-4 para {quiniela['id']}")
            self.logger.debug(f"Prompt completo: {contexto[:500]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self._get_system_prompt_debug()},
                    {"role": "user", "content": contexto}
                ],
                temperature=0.1,  # Muy determinístico para debug
                max_tokens=800
            )
            
            ai_response_text = response.choices[0].message.content
            debug_info["respuesta_ai_raw"] = ai_response_text
            
            self.logger.info(f"✅ Respuesta de IA recibida para {quiniela['id']}")
            self.logger.debug(f"Respuesta completa: {ai_response_text}")
            
            # 3. Parsear respuesta con debug
            resultado = self._parsear_respuesta_con_debug(ai_response_text, quiniela, debug_info)
            debug_info["respuesta_parseada"] = resultado
            
            if resultado:
                debug_info["exito"] = True
                debug_info["quiniela_corregida"] = resultado
                
                # Validar que realmente mejora
                problemas_antes = len(reglas_violadas)
                problemas_despues = len(self._detectar_problemas_quiniela_debug(resultado))
                debug_info["mejora"] = problemas_antes - problemas_despues
                
                self.logger.info(f"✅ Corrección exitosa: {problemas_antes} → {problemas_despues} problemas")
            else:
                debug_info["exito"] = False
                debug_info["errores"].append("No se pudo parsear la respuesta")
                self.logger.error(f"❌ No se pudo parsear respuesta para {quiniela['id']}")
            
        except Exception as e:
            debug_info["exito"] = False
            debug_info["errores"].append(str(e))
            self.logger.error(f"❌ Error en debug de corrección: {e}")
        
        return debug_info

    def _preparar_contexto_debug_detallado(self, quiniela: Dict[str, Any], 
                                         partidos: List[Dict[str, Any]], 
                                         reglas_violadas: List[str]) -> str:
        """
        Contexto SÚPER detallado para debug
        """
        # Información de partidos con nombres y clasificaciones
        info_partidos = []
        anclas_indices = []
        
        for i, partido in enumerate(partidos):
            clasificacion = partido.get("clasificacion", "Neutro")
            es_ancla = clasificacion == "Ancla"
            
            if es_ancla:
                anclas_indices.append(i+1)
            
            info = f"P{i+1}: {partido['home'][:15]} vs {partido['away'][:15]}"
            info += f" [{clasificacion}]"
            info += f" (L:{partido['prob_local']:.3f}, E:{partido['prob_empate']:.3f}, V:{partido['prob_visitante']:.3f})"
            
            if es_ancla:
                info += " ⚠️ NUNCA CAMBIAR - ES ANCLA"
            elif clasificacion == "Divisor":
                info += " 🔄 DIVISOR - Bueno para cambiar"
            elif clasificacion == "TendenciaEmpate":
                info += " ⚖️ TENDENCIA EMPATE"
                
            info_partidos.append(info)
        
        # Análisis DETALLADO de problemas
        resultados_actuales = quiniela['resultados']
        distribucion_actual = quiniela.get('distribución', {})
        
        problemas_detallados = []
        
        for regla in reglas_violadas:
            if "empates" in regla.lower():
                empates_actual = quiniela['empates']
                if empates_actual < 4:
                    problemas_detallados.append(f"EMPATES INSUFICIENTES: tiene {empates_actual}, necesita mínimo 4")
                    problemas_detallados.append(f"  → Debe cambiar {4-empates_actual} resultados L/V a E")
                elif empates_actual > 6:
                    problemas_detallados.append(f"EMPATES EXCESIVOS: tiene {empates_actual}, máximo 6")
                    problemas_detallados.append(f"  → Debe cambiar {empates_actual-6} empates E a L/V")
                    
            elif "concentracion" in regla.lower():
                # Analizar QUÉ está concentrado
                for signo in ['L', 'E', 'V']:
                    count = distribucion_actual.get(signo, 0)
                    porcentaje = (count / 14) * 100
                    
                    if count > 9:  # >70%
                        problemas_detallados.append(f"CONCENTRACIÓN EXCESIVA de {signo}: {count}/14 ({porcentaje:.1f}%)")
                        problemas_detallados.append(f"  → Máximo permitido: 9 ({signo}), debe cambiar {count-9} posiciones")
                        
                        # Sugerir qué cambiar
                        posiciones_este_signo = [i+1 for i, r in enumerate(resultados_actuales) if r == signo]
                        if len(posiciones_este_signo) > 9:
                            problemas_detallados.append(f"  → {signo} está en posiciones: {posiciones_este_signo}")
                
                # Concentración inicial
                primeros_3 = resultados_actuales[:3]
                for signo in ['L', 'E', 'V']:
                    count_inicial = primeros_3.count(signo)
                    if count_inicial > 2:  # >60% de los primeros 3
                        problemas_detallados.append(f"CONCENTRACIÓN INICIAL: {signo} aparece {count_inicial}/3 veces en primeros 3 partidos")
                        problemas_detallados.append(f"  → Máximo 2 iguales en primeros 3 partidos")
        
        # Estado actual detallado
        estado_actual = f"""
QUINIELA ACTUAL: {','.join(resultados_actuales)}
Distribución actual: L:{distribucion_actual.get('L', 0)}, E:{distribucion_actual.get('E', 0)}, V:{distribucion_actual.get('V', 0)}
Empates actuales: {quiniela['empates']}

RESULTADOS POR POSICIÓN:
{chr(10).join(f"P{i+1}: {r} ({partidos[i]['home'][:12]} vs {partidos[i]['away'][:12]})" for i, r in enumerate(resultados_actuales))}
"""

        contexto = f"""TAREA DE CORRECCIÓN DETALLADA PARA DEBUG

QUINIELA A CORREGIR: {quiniela['id']} (Tipo: {quiniela['tipo']})

PROBLEMAS ESPECÍFICOS DETECTADOS:
{chr(10).join(f"❌ {p}" for p in problemas_detallados)}

{estado_actual}

INFORMACIÓN DE PARTIDOS (NUNCA cambiar los marcados con ⚠️):
{chr(10).join(info_partidos)}

REGLAS OBLIGATORIAS:
1. NUNCA cambiar partidos Ancla (posiciones: {anclas_indices})
2. Debe tener entre 4-6 empates total
3. Máximo 9 de cualquier signo L/E/V (≤70% de 14)
4. Máximo 2 iguales en primeros 3 partidos (≤60%)

ESTRATEGIA RECOMENDADA:
- Identificar el problema principal (¿demasiados L?, ¿pocos empates?)
- Buscar partidos NO-Ancla con probabilidad baja del resultado actual
- Hacer cambios mínimos y específicos
- Priorizar partidos Divisor para cambios

EJEMPLO DE RAZONAMIENTO:
"El problema es concentración de L (10/14). Debo cambiar 1 L a E o V.
Mirando las probabilidades, P5 tiene L con solo 0.35 de prob_local.
P5 no es Ancla, así que puedo cambiarlo. Como necesito más empates, cambio P5 de L a E."

RESPUESTA REQUERIDA (JSON estricto, sin explicaciones):
{{"resultados": ["L", "E", "V", "L", "E", "V", "L", "E", "V", "L", "E", "V", "L", "E"], "razonamiento": "Breve explicación del cambio realizado"}}"""

        return contexto

    def _get_system_prompt_debug(self) -> str:
        """
        System prompt específico para debug
        """
        return """Eres un experto en optimización de quinielas deportivas. Tu tarea es corregir quinielas que violan reglas específicas.

CONTEXTO METODOLÓGICO:
- Una quiniela son 14 resultados: L (Local gana), E (Empate), V (Visitante gana)
- Distribución histórica objetivo: 38% L, 29% E, 33% V
- Cada quiniela debe tener 4-6 empates
- Concentración máxima: 70% del mismo signo (máximo 9 de 14)
- Concentración inicial: máximo 60% iguales en primeros 3 (máximo 2 de 3)

PARTIDOS ESPECIALES:
- ANCLA: Partidos con alta probabilidad (>60%) - NUNCA se cambian
- DIVISOR: Partidos equilibrados (40-60%) - Buenos para cambiar
- TENDENCIA EMPATE: Favorecen el empate

INSTRUCCIONES:
1. Identifica el problema principal
2. Encuentra partidos NO-Ancla para cambiar
3. Haz el MÍNIMO de cambios necesarios
4. Prioriza partidos con baja probabilidad del resultado actual
5. Responde SOLO con JSON válido

FORMATO DE RESPUESTA OBLIGATORIO:
{"resultados": ["L", "E", "V", ...], "razonamiento": "Explicación breve"}

¡NO agregues texto fuera del JSON!"""

    def _parsear_respuesta_con_debug(self, respuesta: str, quiniela_original: Dict[str, Any], 
                                   debug_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parsing con debug detallado
        """
        debug_info["parsing_attempts"] = []
        
        # Limpiar respuesta
        respuesta_limpia = respuesta.strip()
        debug_info["respuesta_limpia"] = respuesta_limpia
        
        # Método 1: JSON completo con razonamiento
        try:
            import json
            import re
            
            json_pattern = r'\{[^}]*"resultados"[^}]*\[[^\]]*\][^}]*\}'
            json_match = re.search(json_pattern, respuesta_limpia, re.DOTALL)
            
            if json_match:
                json_text = json_match.group()
                debug_info["parsing_attempts"].append({"method": "json_completo", "text": json_text, "success": False})
                
                data = json.loads(json_text)
                if "resultados" in data and len(data["resultados"]) == 14:
                    resultados = [str(r).upper() for r in data["resultados"]]
                    if all(r in ['L', 'E', 'V'] for r in resultados):
                        debug_info["parsing_attempts"][-1]["success"] = True
                        debug_info["razonamiento_ai"] = data.get("razonamiento", "Sin razonamiento")
                        
                        return self._crear_quiniela_corregida_debug(resultados, quiniela_original, debug_info)
        except Exception as e:
            debug_info["parsing_attempts"].append({"method": "json_completo", "error": str(e), "success": False})
        
        # Método 2: Solo array de resultados
        try:
            import re
            array_pattern = r'\[([^\]]*)\]'
            array_match = re.search(array_pattern, respuesta_limpia)
            
            if array_match:
                array_content = array_match.group(1)
                debug_info["parsing_attempts"].append({"method": "array_simple", "text": array_content, "success": False})
                
                # Extraer solo L, E, V
                resultados = re.findall(r'[LEV]', array_content.upper())
                if len(resultados) == 14:
                    debug_info["parsing_attempts"][-1]["success"] = True
                    return self._crear_quiniela_corregida_debug(resultados, quiniela_original, debug_info)
        except Exception as e:
            debug_info["parsing_attempts"].append({"method": "array_simple", "error": str(e), "success": False})
        
        # Método 3: Extraer todas las letras L/E/V
        try:
            all_letters = re.findall(r'[LEV]', respuesta_limpia.upper())
            debug_info["parsing_attempts"].append({"method": "todas_las_letras", "resultados": all_letters, "success": False})
            
            if len(all_letters) >= 14:
                resultados = all_letters[:14]
                debug_info["parsing_attempts"][-1]["success"] = True
                return self._crear_quiniela_corregida_debug(resultados, quiniela_original, debug_info)
        except Exception as e:
            debug_info["parsing_attempts"].append({"method": "todas_las_letras", "error": str(e), "success": False})
        
        return None

    def _crear_quiniela_corregida_debug(self, resultados: List[str], quiniela_original: Dict[str, Any], 
                                      debug_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crear quiniela corregida con información de debug
        """
        quiniela_corregida = quiniela_original.copy()
        quiniela_corregida["resultados"] = resultados
        quiniela_corregida["empates"] = resultados.count("E")
        quiniela_corregida["distribución"] = {
            "L": resultados.count("L"),
            "E": resultados.count("E"),
            "V": resultados.count("V")
        }
        
        # Análisis de cambios
        resultados_originales = quiniela_original["resultados"]
        cambios = []
        
        for i, (orig, nuevo) in enumerate(zip(resultados_originales, resultados)):
            if orig != nuevo:
                cambios.append(f"P{i+1}: {orig}→{nuevo}")
        
        debug_info["cambios_realizados"] = cambios
        debug_info["num_cambios"] = len(cambios)
        
        return quiniela_corregida

    def _detectar_problemas_quiniela_debug(self, quiniela: Dict[str, Any]) -> List[str]:
        """
        Detectar problemas con información detallada para debug
        """
        problemas = []
        
        # Empates
        empates = quiniela.get("empates", 0)
        if not (4 <= empates <= 6):
            if empates < 4:
                problemas.append(f"empates_insuficientes: {empates} < 4")
            else:
                problemas.append(f"empates_excesivos: {empates} > 6")
        
        # Concentración general
        if "distribución" in quiniela:
            distribucion = quiniela["distribución"]
            for signo, count in distribucion.items():
                if count > 9:  # >70%
                    problemas.append(f"concentracion_general_{signo}: {count}/14 > 9")
        
        # Concentración inicial
        if "resultados" in quiniela and len(quiniela["resultados"]) >= 3:
            primeros_3 = quiniela["resultados"][:3]
            for signo in ["L", "E", "V"]:
                count_inicial = primeros_3.count(signo)
                if count_inicial > 2:  # >60%
                    problemas.append(f"concentracion_inicial_{signo}: {count_inicial}/3 > 2")
        
        return problemas