# progol_optimizer/models/ai_assistant.py
"""
Asistente AI para corrección y optimización de quinielas usando OpenAI API
Con conocimiento completo de la Metodología Definitiva Progol
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

class ProgolAIAssistant:
    """
    Asistente inteligente que usa GPT-4 con conocimiento completo de la metodología Progol
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
            except Exception as e:
                self.logger.error(f"Error inicializando OpenAI: {e}")
                self.enabled = False
    
    def corregir_quiniela_invalida(self, quiniela: Dict[str, Any], partidos: List[Dict[str, Any]], 
                                  reglas_violadas: List[str]) -> Optional[Dict[str, Any]]:
        """
        Usa GPT-4 para corregir una quiniela que viola reglas
        """
        if not self.enabled:
            return None
            
        try:
            # Preparar contexto para GPT-4
            contexto = self._preparar_contexto_correccion(quiniela, partidos, reglas_violadas)
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self._get_system_prompt_metodologia_completa()},
                    {"role": "user", "content": contexto}
                ],
                temperature=0.3,  # Baja temperatura para respuestas más determinísticas
                max_tokens=500
            )
            
            # Parsear respuesta
            resultado = self._parsear_respuesta_correccion(response.choices[0].message.content, quiniela)
            
            if resultado:
                self.logger.info("✅ Quiniela corregida exitosamente por AI")
                return resultado
            else:
                self.logger.warning("⚠️ No se pudo parsear la corrección de AI")
                return None
                
        except Exception as e:
            self.logger.error(f"Error en corrección AI: {e}")
            return None
    
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

CALIBRACIÓN BAYESIANA:
- Las probabilidades ya están calibradas con factores k1(forma), k2(lesiones), k3(contexto)
- Draw-Propensity Rule: Si |p_L - p_V| < 0.08 y p_E > max(p_L, p_V), boost +6pp al empate

OBJETIVO DE OPTIMIZACIÓN:
Maximizar F = 1 - ∏(1 - Pr[≥11 aciertos]) sobre las 30 quinielas

CRITERIOS DE ÉXITO:
- Distribución global dentro de rangos históricos
- Alta diversidad entre quinielas (baja correlación)
- Respeto absoluto de partidos Ancla
- Balance entre seguridad (Core) y cobertura (Satélites)

Cuando corrijas o sugieras cambios, SIEMPRE verifica que se mantengan estas reglas."""

    def _preparar_contexto_correccion(self, quiniela: Dict[str, Any], partidos: List[Dict[str, Any]], 
                                     reglas_violadas: List[str]) -> str:
        """Prepara el contexto para la corrección incluyendo clasificación de partidos"""
        
        # Información detallada de partidos
        info_partidos = []
        for i, partido in enumerate(partidos):
            clasificacion = partido.get("clasificacion", "Neutro")
            es_ancla = clasificacion == "Ancla"
            
            info = f"P{i+1}: {partido['home'][:15]} vs {partido['away'][:15]} "
            info += f"[{clasificacion}] "
            info += f"(L:{partido['prob_local']:.2f}, E:{partido['prob_empate']:.2f}, V:{partido['prob_visitante']:.2f})"
            
            if es_ancla:
                info += " ⚠️ NO MODIFICAR"
                
            info_partidos.append(info)
        
        # Análisis de la quiniela actual
        primeros_3 = quiniela['resultados'][:3]
        conc_inicial = max(primeros_3.count(s) for s in ['L', 'E', 'V'])
        
        contexto = f"""TAREA: Corregir quiniela {quiniela['id']} ({quiniela['tipo']})

VIOLACIONES DETECTADAS: {', '.join(reglas_violadas)}

QUINIELA ACTUAL:
Resultados: {','.join(quiniela['resultados'])}
Empates: {quiniela['empates']} (debe ser 4-6)
Distribución: L={quiniela['distribución']['L']}, E={quiniela['distribución']['E']}, V={quiniela['distribución']['V']}
Concentración general: {max(quiniela['distribución'].values())}/14 = {max(quiniela['distribución'].values())/14:.1%}
Concentración inicial (primeros 3): {conc_inicial}/3 = {conc_inicial/3:.1%}

INFORMACIÓN DE PARTIDOS:
{chr(10).join(info_partidos)}

INSTRUCCIONES:
1. Corrige SOLO lo necesario para cumplir las reglas
2. NUNCA modifiques partidos marcados como [Ancla]
3. Mantén 4-6 empates
4. Asegura concentración ≤70% general y ≤60% en primeros 3
5. Responde con JSON: {{"resultados": ["L", "E", "V", ...]}}"""

        return contexto

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
        
        # Análisis por posición
        posiciones_desbalanceadas = []
        for pos in range(14):
            if partidos[pos].get("clasificacion") != "Ancla":
                conteos = {"L": 0, "E": 0, "V": 0}
                for q in portafolio:
                    conteos[q["resultados"][pos]] += 1
                
                max_apariciones = max(conteos.values())
                if max_apariciones > 20:  # Más del 67%
                    posiciones_desbalanceadas.append(f"P{pos+1}: {conteos}")
        
        contexto = f"""TAREA: Optimizar portafolio completo de 30 quinielas

ARQUITECTURA ACTUAL:
- {len(cores)} Core + {len(satelites)} Satélites (debe ser 4 + 26)
- Partidos Ancla: {len(anclas)} en posiciones {[p+1 for p in anclas]}
- Partidos Divisores: {len(divisores)} en posiciones {[p+1 for p in divisores[:5]]}...

PROBLEMAS DETECTADOS:
{chr(10).join(f"- {p}" for p in problemas)}

DISTRIBUCIÓN ACTUAL vs OBJETIVO:
L: {total_L} ({(total_L/total)*100:.1f}%) - Objetivo: 35-41% (490-574 de 1400)
E: {total_E} ({(total_E/total)*100:.1f}%) - Objetivo: 25-33% (350-462 de 1400)
V: {total_V} ({(total_V/total)*100:.1f}%) - Objetivo: 30-36% (420-504 de 1400)

QUINIELAS CRÍTICAS ({len(quinielas_criticas)} de 30):
{chr(10).join(quinielas_criticas[:10])}

POSICIONES DESBALANCEADAS:
{chr(10).join(posiciones_desbalanceadas[:5])}

INSTRUCCIONES PARA OPTIMIZACIÓN:
1. Sugiere cambios ESPECÍFICOS: ID de quiniela + posición + nuevo resultado
2. Prioriza cambios en Satélites (no tocar Core si es posible)
3. NUNCA sugieras cambiar partidos Ancla
4. Busca balancear la distribución por posición
5. Mantén correlación baja entre pares de satélites
6. Cada cambio debe acercar la distribución global a los rangos objetivo

Formato de respuesta esperado:
"Cambio 1: Sat-1A posición 5 cambiar de L a E (reduce exceso de L, balancea posición 5)
Cambio 2: Sat-3B posición 8 cambiar de V a L (aumenta L hacia objetivo)
..."

Sugiere entre 10-20 cambios específicos para corregir los problemas."""

        return contexto

    def generar_satelite_inteligente(self, partidos: List[Dict[str, Any]], 
                                   satelites_existentes: List[Dict[str, Any]],
                                   tipo_par: str = "A") -> Optional[List[str]]:
        """
        Genera un satélite nuevo usando el conocimiento completo de la metodología
        """
        if not self.enabled:
            return None
            
        try:
            # Analizar distribución actual
            total_L = sum(s["distribución"]["L"] for s in satelites_existentes)
            total_E = sum(s["distribución"]["E"] for s in satelites_existentes)
            total_V = sum(s["distribución"]["V"] for s in satelites_existentes)
            
            # Determinar qué necesitamos más
            necesidades = {
                "L": 13 * 38 - total_L,  # Lo que falta para llegar al 38%
                "E": 13 * 29 - total_E,  # Lo que falta para llegar al 29%
                "V": 13 * 33 - total_V   # Lo que falta para llegar al 33%
            }
            
            contexto = self._preparar_contexto_generacion_satelite(partidos, necesidades, tipo_par)
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self._get_system_prompt_metodologia_completa()},
                    {"role": "user", "content": contexto}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Parsear y validar resultado
            quiniela = self._parsear_quiniela_generada(response.choices[0].message.content)
            
            if quiniela and len(quiniela) == 14:
                return quiniela
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error generando satélite con AI: {e}")
            return None

    def validar_y_explicar_portafolio(self, portafolio: List[Dict[str, Any]], 
                                     partidos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Usa AI para validar y explicar el estado del portafolio
        """
        if not self.enabled:
            return {"explicacion": "AI no disponible"}
            
        try:
            # Preparar análisis completo
            from validation.portfolio_validator import PortfolioValidator
            validator = PortfolioValidator()
            validacion = validator.validar_portafolio_completo(portafolio)
            
            contexto = f"""Analiza este portafolio de Progol y explica su estado:

VALIDACIÓN: {'✅ VÁLIDO' if validacion['es_valido'] else '❌ INVÁLIDO'}

MÉTRICAS:
{json.dumps(validacion['metricas'], indent=2)}

REGLAS CUMPLIDAS/FALLADAS:
{json.dumps(validacion['detalle_validaciones'], indent=2)}

Proporciona:
1. Resumen ejecutivo del estado
2. Principales fortalezas
3. Principales debilidades
4. Recomendaciones específicas para mejorar

Sé conciso pero completo."""

            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self._get_system_prompt_metodologia_completa()},
                    {"role": "user", "content": contexto}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            return {
                "valido": validacion['es_valido'],
                "explicacion": response.choices[0].message.content,
                "metricas": validacion['metricas']
            }
            
        except Exception as e:
            self.logger.error(f"Error en validación AI: {e}")
            return {"explicacion": f"Error: {str(e)}"}

    def _parsear_respuesta_correccion(self, respuesta: str, quiniela_original: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parsea la respuesta de corrección de GPT-4"""
        try:
            # Intentar extraer JSON
            json_match = re.search(r'\{.*\}', respuesta, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "resultados" in data and len(data["resultados"]) == 14:
                    quiniela_corregida = quiniela_original.copy()
                    quiniela_corregida["resultados"] = data["resultados"]
                    quiniela_corregida["empates"] = data["resultados"].count("E")
                    quiniela_corregida["distribución"] = {
                        "L": data["resultados"].count("L"),
                        "E": data["resultados"].count("E"),
                        "V": data["resultados"].count("V")
                    }
                    return quiniela_corregida
            return None
        except:
            return None

    def _analizar_problemas_portafolio(self, portafolio: List[Dict[str, Any]]) -> List[str]:
        """Analiza qué problemas tiene el portafolio actual"""
        problemas = []
        
        # Calcular distribución global
        total_L = sum(q["distribución"]["L"] for q in portafolio)
        total_E = sum(q["distribución"]["E"] for q in portafolio)
        total_V = sum(q["distribución"]["V"] for q in portafolio)
        total = total_L + total_E + total_V
        
        porc_L = total_L / total
        porc_E = total_E / total
        porc_V = total_V / total
        
        # Verificar rangos
        if not (0.35 <= porc_L <= 0.41):
            problemas.append(f"Distribución L fuera de rango: {porc_L:.1%}")
        if not (0.25 <= porc_E <= 0.33):
            problemas.append(f"Distribución E fuera de rango: {porc_E:.1%}")
        if not (0.30 <= porc_V <= 0.36):
            problemas.append(f"Distribución V fuera de rango: {porc_V:.1%}")
            
        # Verificar concentraciones individuales
        quinielas_con_problemas = 0
        for q in portafolio:
            max_conc = max(q["distribución"].values()) / 14
            if max_conc > 0.70:
                quinielas_con_problemas += 1
                
        if quinielas_con_problemas > 0:
            problemas.append(f"{quinielas_con_problemas} quinielas con concentración >70%")
            
        return problemas

    def _preparar_contexto_generacion_satelite(self, partidos: List[Dict[str, Any]], 
                                             necesidades: Dict[str, str], tipo_par: str) -> str:
        """Prepara contexto para generar un satélite nuevo"""
        
        # Información de partidos con clasificación
        info_partidos = []
        anclas_indices = []
        divisores_indices = []
        
        for i, partido in enumerate(partidos):
            clasificacion = partido.get("clasificacion", "Neutro")
            
            if clasificacion == "Ancla":
                anclas_indices.append(i)
            elif clasificacion == "Divisor":
                divisores_indices.append(i)
                
            info = f"P{i+1}: [{clasificacion}] "
            info += f"L:{partido['prob_local']:.2f}, E:{partido['prob_empate']:.2f}, V:{partido['prob_visitante']:.2f}"
            info_partidos.append(info)
        
        # Determinar qué resultado necesitamos más
        mas_necesitado = max(necesidades, key=necesidades.get)
        
        contexto = f"""TAREA: Generar Satélite {tipo_par} que ayude a balancear el portafolio

NECESIDADES DE DISTRIBUCIÓN:
- Necesitamos más {mas_necesitado} (faltan {necesidades[mas_necesitado]} apariciones)
- L actual necesita {necesidades['L']} más
- E actual necesita {necesidades['E']} más  
- V actual necesita {necesidades['V']} más

PARTIDOS Y CLASIFICACIÓN:
{chr(10).join(info_partidos)}

REGLAS ESTRICTAS:
1. Partidos ANCLA (posiciones {[p+1 for p in anclas_indices]}): usar SIEMPRE resultado de máxima probabilidad
2. Partidos DIVISOR (posiciones {[p+1 for p in divisores_indices[:3]]}...): variar para diversidad
3. Debe tener entre 4-6 empates
4. Máximo 70% concentración general (9 de 14)
5. Máximo 60% concentración en primeros 3 (2 de 3)

ESTRATEGIA:
- Priorizar {mas_necesitado} en partidos no-Ancla donde sea razonable
- Mantener coherencia con las probabilidades
- Si es tipo "B", invertir algunos Divisores respecto al tipo "A"

Responde SOLO con los 14 resultados separados por comas: L,E,V,L,E,V..."""

        return contexto

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

    def _parsear_quiniela_generada(self, respuesta: str) -> Optional[List[str]]:
        """Parsea una quiniela generada por GPT-4"""
        try:
            # Buscar patrón de 14 resultados
            # Limpiar respuesta
            respuesta_limpia = respuesta.strip().upper()
            
            # Buscar secuencia de L,E,V
            patron = r'[LEV](?:\s*,\s*[LEV]){13}'
            match = re.search(patron, respuesta_limpia)
            
            if match:
                resultados_str = match.group()
                resultados = [r.strip() for r in resultados_str.split(',')]
                
                if len(resultados) == 14 and all(r in ['L', 'E', 'V'] for r in resultados):
                    return resultados
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error parseando quiniela generada: {e}")
            return None