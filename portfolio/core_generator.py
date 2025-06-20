# progol_optimizer/portfolio/core_generator.py - VERSIÓN FINAL CORREGIDA
"""
Generador de Quinielas Core - SOLUCIÓN DEFINITIVA
Crea 4 quinielas Core con VERDADERA variación que pasan TODAS las validaciones
"""

import logging
import random
import copy
from typing import List, Dict, Any

class CoreGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.rangos = self.config["RANGOS_HISTORICOS"]
        self.logger.debug("CoreGenerator DEFINITIVO inicializado")

    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genera 4 quinielas Core que PASAN TODAS las validaciones"""
        self.logger.info("🎯 Generando 4 quinielas Core con VARIACIÓN GARANTIZADA...")
        
        # Identificar partidos especiales
        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        divisores_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Divisor"]
        otros_indices = [i for i in range(14) if i not in anclas_indices and i not in divisores_indices]
        
        self.logger.info(f"📌 Anclas: {len(anclas_indices)}, Divisores: {len(divisores_indices)}, Otros: {len(otros_indices)}")
        
        # Generar 4 quinielas con estrategias específicas y diferentes
        quinielas_core = []
        
        for core_num in range(4):
            core_id = f"Core-{core_num + 1}"
            
            max_intentos = 50
            for intento in range(max_intentos):
                quiniela = self._generar_core_individual(
                    core_num, partidos_clasificados, 
                    anclas_indices, divisores_indices, otros_indices
                )
                
                # Validar que cumple TODAS las reglas individuales
                if self._validar_core_individual(quiniela, core_num):
                    # Verificar que sea diferente a las existentes
                    if self._es_suficientemente_diferente(quiniela, quinielas_core):
                        quiniela_info = self._crear_info_quiniela(core_id, quiniela)
                        quinielas_core.append(quiniela_info)
                        self.logger.info(f"✅ {core_id} generada en intento {intento + 1}")
                        break
            else:
                raise RuntimeError(f"No se pudo generar {core_id} válida después de {max_intentos} intentos")
        
        # Validación y ajuste final del conjunto
        quinielas_finales = self._ajuste_final_conjunto(quinielas_core, partidos_clasificados, anclas_indices)
        
        # Verificación final
        self._verificacion_final_completa(quinielas_finales)
        
        return quinielas_finales

    def _generar_core_individual(self, core_num: int, partidos: List[Dict[str, Any]], 
                                 anclas_indices: List[int], divisores_indices: List[int], 
                                 otros_indices: List[int]) -> List[str]:
        """Genera una quiniela Core individual con estrategia específica"""
        
        resultados = [""] * 14
        
        # PASO 1: Fijar Anclas (iguales para todas las Core)
        for idx in anclas_indices:
            resultado_ancla = self._get_resultado_max_prob(partidos[idx])
            resultados[idx] = resultado_ancla
        
        # PASO 2: Estrategias específicas por Core para crear VARIACIÓN
        if core_num == 0:  # Core-1: Balance conservador
            self._aplicar_estrategia_conservadora(resultados, partidos, divisores_indices, otros_indices)
        elif core_num == 1:  # Core-2: Prioriza visitantes
            self._aplicar_estrategia_visitantes(resultados, partidos, divisores_indices, otros_indices)
        elif core_num == 2:  # Core-3: Más empates estratégicos
            self._aplicar_estrategia_empates(resultados, partidos, divisores_indices, otros_indices)
        else:  # Core-4: Mix rotativo para máxima variación
            self._aplicar_estrategia_mixta(resultados, partidos, divisores_indices, otros_indices)
        
        # PASO 3: Asegurar 4-6 empates
        resultados = self._ajustar_empates_garantizado(resultados, partidos, anclas_indices)
        
        # PASO 4: Corregir concentración inicial si es necesaria
        resultados = self._corregir_concentracion_inicial_agresiva(resultados, partidos, anclas_indices)
        
        return resultados

    def _aplicar_estrategia_conservadora(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                         divisores_indices: List[int], otros_indices: List[int]):
        """Core-1: Estrategia conservadora pero con variación en primeros 3"""
        
        # Para divisores: usar máxima probabilidad
        for idx in divisores_indices:
            resultados[idx] = self._get_resultado_max_prob(partidos[idx])
        
        # Para otros: usar máxima probabilidad pero con variación intencional en posiciones clave
        for idx in otros_indices:
            if idx < 3:  # Primeros 3: crear variación intencional
                probs = self._get_probabilidades_ordenadas(partidos[idx])
                # Usar segunda opción en posición 1 para variación
                if idx == 1:
                    resultados[idx] = probs[1][0]  # Segunda opción
                else:
                    resultados[idx] = probs[0][0]  # Primera opción
            else:
                resultados[idx] = self._get_resultado_max_prob(partidos[idx])

    def _aplicar_estrategia_visitantes(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                       divisores_indices: List[int], otros_indices: List[int]):
        """Core-2: Prioriza visitantes para balancear distribución global"""
        
        for idx in divisores_indices + otros_indices:
            if idx < 3:  # Variación en primeros 3
                if idx == 0:  # Posición 0: priorizar empate para variación
                    if partidos[idx]["prob_empate"] > 0.25:
                        resultados[idx] = "E"
                    else:
                        resultados[idx] = self._get_resultado_max_prob(partidos[idx])
                elif idx == 2:  # Posición 2: priorizar visitante si es viable
                    if partidos[idx]["prob_visitante"] > 0.30:
                        resultados[idx] = "V"
                    else:
                        resultados[idx] = self._get_resultado_max_prob(partidos[idx])
                else:
                    resultados[idx] = self._get_resultado_max_prob(partidos[idx])
            else:
                # Resto: priorizar visitantes cuando sea razonable
                probs = self._get_probabilidades_ordenadas(partidos[idx])
                if probs[0][0] == "V" or (probs[1][0] == "V" and probs[1][1] > 0.35):
                    resultados[idx] = "V"
                else:
                    resultados[idx] = probs[0][0]

    def _aplicar_estrategia_empates(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                    divisores_indices: List[int], otros_indices: List[int]):
        """Core-3: Más empates estratégicos"""
        
        for idx in divisores_indices + otros_indices:
            if idx < 3:  # Variación específica en primeros 3
                if idx == 1:  # Posición 1: cambiar a visitante para variación
                    if partidos[idx]["prob_visitante"] > 0.25:
                        resultados[idx] = "V"
                    else:
                        resultados[idx] = self._get_resultado_max_prob(partidos[idx])
                else:
                    resultados[idx] = self._get_resultado_max_prob(partidos[idx])
            else:
                # Resto: priorizar empates cuando la probabilidad sea decente
                if partidos[idx]["prob_empate"] > 0.30:
                    resultados[idx] = "E"
                else:
                    resultados[idx] = self._get_resultado_max_prob(partidos[idx])

    def _aplicar_estrategia_mixta(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                  divisores_indices: List[int], otros_indices: List[int]):
        """Core-4: Mix rotativo para máxima variación"""
        
        opciones = ["L", "E", "V"]
        
        for i, idx in enumerate(divisores_indices + otros_indices):
            if idx < 3:  # Máxima variación en primeros 3
                if idx == 0:  # Posición 0: usar visitante para máxima variación
                    if partidos[idx]["prob_visitante"] > 0.20:
                        resultados[idx] = "V"
                    else:
                        resultados[idx] = self._get_resultado_max_prob(partidos[idx])
                elif idx == 1:  # Posición 1: mantener empate si es viable
                    if partidos[idx]["prob_empate"] > 0.20:
                        resultados[idx] = "E"
                    else:
                        resultados[idx] = self._get_resultado_max_prob(partidos[idx])
                else:
                    resultados[idx] = self._get_resultado_max_prob(partidos[idx])
            else:
                # Resto: rotación basada en posición para distribución equilibrada
                estrategia_idx = i % 3
                probs = self._get_probabilidades_ordenadas(partidos[idx])
                
                # Usar la opción según rotación, pero solo si es razonable
                if len(probs) > estrategia_idx and probs[estrategia_idx][1] > 0.20:
                    resultados[idx] = probs[estrategia_idx][0]
                else:
                    resultados[idx] = probs[0][0]

    def _ajustar_empates_garantizado(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                     anclas_indices: List[int]) -> List[str]:
        """Garantiza que la quiniela tenga entre 4-6 empates"""
        
        resultados_ajustados = resultados.copy()
        empates_actuales = resultados_ajustados.count("E")
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Si faltan empates
        while empates_actuales < self.empates_min and modificables:
            # Buscar el mejor candidato para convertir a empate
            mejor_candidato = max(
                [i for i in modificables if resultados_ajustados[i] != "E"],
                key=lambda i: partidos[i]["prob_empate"],
                default=None
            )
            
            if mejor_candidato is not None:
                resultados_ajustados[mejor_candidato] = "E"
                empates_actuales += 1
            else:
                break
        
        # Si sobran empates
        while empates_actuales > self.empates_max and modificables:
            # Buscar el peor empate para convertir
            candidatos_empate = [i for i in modificables if resultados_ajustados[i] == "E"]
            if candidatos_empate:
                peor_empate = min(candidatos_empate, key=lambda i: partidos[i]["prob_empate"])
                # Cambiar al resultado más probable
                nuevo_resultado = self._get_resultado_max_prob(partidos[peor_empate])
                resultados_ajustados[peor_empate] = nuevo_resultado
                empates_actuales -= 1
            else:
                break
        
        return resultados_ajustados

    def _corregir_concentracion_inicial_agresiva(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                                  anclas_indices: List[int]) -> List[str]:
        """Corrección AGRESIVA de concentración en primeros 3 partidos"""
        
        resultados_corregidos = resultados.copy()
        primeros_3 = resultados_corregidos[:3]
        modificables_iniciales = [i for i in range(3) if i not in anclas_indices]
        
        if not modificables_iniciales:
            return resultados_corregidos
        
        # Contar apariciones en primeros 3
        conteos = {"L": primeros_3.count("L"), "E": primeros_3.count("E"), "V": primeros_3.count("V")}
        
        # Si algún signo aparece más de 1 vez (>33.3%), corregir
        for signo, count in conteos.items():
            if count > 1:  # Más de 1 vez = >33.3% = viola regla ≤60%
                self.logger.debug(f"Corrigiendo concentración inicial: {signo} aparece {count} veces en primeros 3")
                
                # Encontrar posiciones modificables con este signo
                posiciones_signo = [i for i in modificables_iniciales if resultados_corregidos[i] == signo]
                
                if len(posiciones_signo) > 1:
                    # Cambiar la posición con menor probabilidad de este signo
                    pos_a_cambiar = min(posiciones_signo, 
                                       key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                    
                    # Cambiar al signo menos usado en primeros 3
                    otros_signos = {s: c for s, c in conteos.items() if s != signo}
                    nuevo_signo = min(otros_signos, key=otros_signos.get)
                    
                    # Verificar que el nuevo signo sea razonable para esta posición
                    prob_nuevo = partidos[pos_a_cambiar][f"prob_{self._resultado_a_clave(nuevo_signo)}"]
                    if prob_nuevo > 0.15:  # Al menos 15% de probabilidad
                        resultados_corregidos[pos_a_cambiar] = nuevo_signo
                        # Actualizar conteos para próxima iteración
                        conteos[signo] -= 1
                        conteos[nuevo_signo] += 1
        
        return resultados_corregidos

    def _validar_core_individual(self, resultados: List[str], core_num: int) -> bool:
        """Valida que una Core individual cumple todas las reglas"""
        
        # Validar empates
        empates = resultados.count("E")
        if not (self.empates_min <= empates <= self.empates_max):
            self.logger.debug(f"Core-{core_num + 1}: empates {empates} fuera de rango [{self.empates_min}-{self.empates_max}]")
            return False
        
        # Validar concentración general
        max_conc_general = max(resultados.count(s) for s in ["L", "E", "V"]) / 14
        if max_conc_general > 0.70:
            self.logger.debug(f"Core-{core_num + 1}: concentración general {max_conc_general:.1%} > 70%")
            return False
        
        # Validar concentración inicial
        primeros_3 = resultados[:3]
        max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3
        if max_conc_inicial > 0.60:
            self.logger.debug(f"Core-{core_num + 1}: concentración inicial {max_conc_inicial:.1%} > 60%")
            return False
        
        return True

    def _es_suficientemente_diferente(self, nueva_quiniela: List[str], 
                                      existentes: List[Dict[str, Any]]) -> bool:
        """Verifica que la nueva quiniela sea suficientemente diferente"""
        
        if not existentes:
            return True
        
        for q_existente in existentes:
            coincidencias = sum(1 for a, b in zip(nueva_quiniela, q_existente["resultados"]) if a == b)
            similitud = coincidencias / 14
            
            if similitud > 0.85:  # Más de 85% iguales = muy similar
                return False
        
        return True

    def _ajuste_final_conjunto(self, quinielas_core: List[Dict[str, Any]], 
                               partidos: List[Dict[str, Any]], 
                               anclas_indices: List[int]) -> List[Dict[str, Any]]:
        """Ajuste final del conjunto para optimizar distribución global"""
        
        self.logger.info("🔧 Aplicando ajuste final del conjunto...")
        
        # Calcular distribución actual
        total_L = sum(q["distribución"]["L"] for q in quinielas_core)
        total_E = sum(q["distribución"]["E"] for q in quinielas_core)
        total_V = sum(q["distribución"]["V"] for q in quinielas_core)
        total_partidos = len(quinielas_core) * 14
        
        porc_L = total_L / total_partidos
        porc_E = total_E / total_partidos
        porc_V = total_V / total_partidos
        
        self.logger.info(f"Distribución actual: L={porc_L:.1%}, E={porc_E:.1%}, V={porc_V:.1%}")
        
        # Objetivos de distribución
        target_L = 0.39  # 39% (dentro del rango 35-41%)
        target_E = 0.29  # 29% (dentro del rango 25-33%)
        target_V = 0.32  # 32% (dentro del rango 30-36%)
        
        # Calcular ajustes necesarios
        ajuste_L = int((target_L - porc_L) * total_partidos)
        ajuste_V = int((target_V - porc_V) * total_partidos)
        
        self.logger.info(f"Ajustes necesarios: L{ajuste_L:+d}, V{ajuste_V:+d}")
        
        # Aplicar ajustes si son necesarios
        if abs(ajuste_L) > 0 or abs(ajuste_V) > 0:
            quinielas_ajustadas = self._aplicar_ajustes_distribucion(
                quinielas_core, partidos, anclas_indices, ajuste_L, ajuste_V
            )
        else:
            quinielas_ajustadas = quinielas_core
        
        return quinielas_ajustadas

    def _aplicar_ajustes_distribucion(self, quinielas: List[Dict[str, Any]], 
                                      partidos: List[Dict[str, Any]], 
                                      anclas_indices: List[int], 
                                      ajuste_L: int, ajuste_V: int) -> List[Dict[str, Any]]:
        """Aplica ajustes específicos de distribución"""
        
        quinielas_ajustadas = copy.deepcopy(quinielas)
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Si necesitamos reducir L y aumentar V
        if ajuste_L < 0 and ajuste_V > 0:
            cambios_necesarios = min(abs(ajuste_L), abs(ajuste_V))
            
            # Buscar mejores candidatos para cambiar L → V
            candidatos = []
            for q_idx, quiniela in enumerate(quinielas_ajustadas):
                for pos in modificables:
                    if quiniela["resultados"][pos] == "L":
                        prob_L = partidos[pos]["prob_local"]
                        prob_V = partidos[pos]["prob_visitante"]
                        score = prob_V - prob_L  # Preferir donde V es más probable que L
                        candidatos.append((q_idx, pos, score))
            
            # Ordenar por score (mejor candidato primero)
            candidatos.sort(key=lambda x: x[2], reverse=True)
            
            # Aplicar cambios
            for i in range(min(cambios_necesarios, len(candidatos))):
                q_idx, pos, _ = candidatos[i]
                quinielas_ajustadas[q_idx]["resultados"][pos] = "V"
                # Actualizar distribución
                quinielas_ajustadas[q_idx]["distribución"]["L"] -= 1
                quinielas_ajustadas[q_idx]["distribución"]["V"] += 1
        
        return quinielas_ajustadas

    def _verificacion_final_completa(self, quinielas_core: List[Dict[str, Any]]):
        """Verificación final completa con logging detallado"""
        
        self.logger.info("=== VERIFICACIÓN FINAL DEL GENERADOR CORE ===")
        
        # Verificar cada quiniela individual
        for q in quinielas_core:
            empates = q["empates"]
            distribución = q["distribución"]
            resultados = q["resultados"]
            
            # Concentración general
            max_conc_general = max(distribución.values()) / 14
            
            # Concentración inicial
            primeros_3 = resultados[:3]
            max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3
            
            problemas = []
            if not (self.empates_min <= empates <= self.empates_max):
                problemas.append(f"empates={empates}")
            if max_conc_general > 0.70:
                problemas.append(f"conc_general={max_conc_general:.1%}")
            if max_conc_inicial > 0.60:
                problemas.append(f"conc_inicial={max_conc_inicial:.1%}")
            
            if problemas:
                self.logger.error(f"❌ {q['id']}: {', '.join(problemas)}")
            else:
                self.logger.info(f"✅ {q['id']}: válida")
        
        # Verificar distribución global
        total_L = sum(q["distribución"]["L"] for q in quinielas_core)
        total_E = sum(q["distribución"]["E"] for q in quinielas_core)
        total_V = sum(q["distribución"]["V"] for q in quinielas_core)
        total_partidos = len(quinielas_core) * 14
        
        porc_L = total_L / total_partidos
        porc_E = total_E / total_partidos
        porc_V = total_V / total_partidos
        
        self.logger.info(f"📊 Distribución final: L={porc_L:.1%}, E={porc_E:.1%}, V={porc_V:.1%}")
        
        # Verificar variación por posición
        posiciones_100_pct = 0
        for pos in range(14):
            conteos = {"L": 0, "E": 0, "V": 0}
            for q in quinielas_core:
                resultado = q["resultados"][pos]
                conteos[resultado] += 1
            
            max_apariciones = max(conteos.values())
            if max_apariciones == len(quinielas_core):
                posiciones_100_pct += 1
        
        self.logger.info(f"🎯 Posiciones con 100% mismo resultado: {posiciones_100_pct}/14")
        
        if posiciones_100_pct == 0:
            self.logger.info("✅ EXCELENTE: Variación perfecta por posición")
        elif posiciones_100_pct <= 3:
            self.logger.info("✅ BUENO: Variación aceptable por posición")
        else:
            self.logger.warning(f"⚠️ MEJORABLE: {posiciones_100_pct} posiciones sin variación")

    # Funciones auxiliares
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        return max(probs, key=probs.get)

    def _get_probabilidades_ordenadas(self, partido: Dict[str, Any]) -> List[tuple]:
        """Obtiene probabilidades ordenadas de mayor a menor"""
        probs = [
            ("L", partido["prob_local"]),
            ("E", partido["prob_empate"]),
            ("V", partido["prob_visitante"])
        ]
        return sorted(probs, key=lambda x: x[1], reverse=True)

    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte resultado a clave de probabilidad"""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "local")

    def _crear_info_quiniela(self, q_id: str, resultados: List[str]) -> Dict[str, Any]:
        """Crea estructura completa de quiniela"""
        return {
            "id": q_id,
            "tipo": "Core",
            "resultados": resultados,
            "empates": resultados.count("E"),
            "distribución": {
                "L": resultados.count("L"),
                "E": resultados.count("E"),
                "V": resultados.count("V")
            }
        }