# progol_optimizer/portfolio/optimizer.py - CORRECCIÓN FINAL ESPECÍFICA
"""
Optimizador GRASP-Annealing - CORRECCIÓN FINAL
Soluciona específicamente los 2 problemas restantes:
1. Concentración Máxima en quinielas individuales
2. Distribución Divisores por posición
"""

import logging
import random
import math
import numpy as np
from typing import List, Dict, Any, Tuple

class GRASPAnnealing:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.opt_config = self.config["OPTIMIZACION"]
        
        self.max_iteraciones = self.opt_config["max_iteraciones"]
        self.temperatura_inicial = self.opt_config["temperatura_inicial"]
        self.factor_enfriamiento = self.opt_config["factor_enfriamiento"]
        self.alpha_grasp = self.opt_config["alpha_grasp"]
        self.iteraciones_sin_mejora_max = self.opt_config["iteraciones_sin_mejora"]

        self.cache_probabilidades = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        from validation.portfolio_validator import PortfolioValidator
        self.validator = PortfolioValidator()

    def _resultado_a_clave(self, resultado: str) -> str:
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "")

    def optimizar_portafolio_grasp_annealing(self, quinielas_iniciales: List[Dict[str, Any]],
                                           partidos: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
        self.logger.info("🚀 Iniciando optimización GRASP-Annealing FINAL...")
        self._precalcular_matrices_probabilidades(partidos)

        portafolio_actual = [q.copy() for q in quinielas_iniciales]
        score_actual = self._calcular_objetivo_f_optimizado(portafolio_actual, partidos)
        
        mejor_portafolio = portafolio_actual
        mejor_score = score_actual

        temperatura = self.temperatura_inicial
        iteraciones_sin_mejora = 0
        self.logger.info(f"Score inicial: F={score_actual:.6f}")

        for iteracion in range(self.max_iteraciones):
            nuevo_portafolio_candidato = self._generar_movimiento_valido(portafolio_actual, partidos)

            if nuevo_portafolio_candidato is None:
                continue

            nuevo_score = self._calcular_objetivo_f_optimizado(nuevo_portafolio_candidato, partidos)
            delta = nuevo_score - score_actual

            if delta > 0 or (temperatura > 0 and random.random() < math.exp(delta / temperatura)):
                portafolio_actual = nuevo_portafolio_candidato
                score_actual = nuevo_score
                
                if nuevo_score > mejor_score:
                    mejor_portafolio = portafolio_actual
                    mejor_score = nuevo_score
                    iteraciones_sin_mejora = 0
                    self.logger.debug(f"Iter {iteracion}: Nueva mejor solución -> Score {mejor_score:.6f}")
                else:
                    iteraciones_sin_mejora += 1
            else:
                iteraciones_sin_mejora += 1

            if progress_callback and iteracion % 10 == 0:
                progreso = iteracion / self.max_iteraciones
                texto = f"Iter. {iteracion}/{self.max_iteraciones} | Score: {mejor_score:.5f}"
                progress_callback(progreso, texto)

            temperatura *= self.factor_enfriamiento

            if iteraciones_sin_mejora >= self.iteraciones_sin_mejora_max:
                self.logger.info(f"⏹️ Parada temprana en iteración {iteracion} (sin mejora)")
                break
        
        self.logger.info("Fase final ESPECÍFICA: Corrigiendo concentración y distribución...")
        mejor_portafolio = self._ajuste_final_especifico(mejor_portafolio, partidos)

        score_final = self._calcular_objetivo_f_optimizado(mejor_portafolio, partidos)
        self.logger.info(f"✅ Optimización FINAL completada: F={score_final:.6f}")
        return mejor_portafolio

    def _generar_movimiento_valido(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        intentos = 0
        while intentos < 20:
            intentos += 1
            
            nuevo_portafolio = [q.copy() for q in portafolio]
            quiniela_idx = random.randrange(len(nuevo_portafolio))
            
            if random.random() < 0.8:
                satelite_indices = [i for i, q in enumerate(nuevo_portafolio) if q['tipo'] == 'Satelite']
                if satelite_indices:
                    quiniela_idx = random.choice(satelite_indices)
            
            quiniela_original = nuevo_portafolio[quiniela_idx]
            nuevos_resultados = quiniela_original["resultados"].copy()
            
            partidos_modificables = [i for i, p in enumerate(partidos) if p.get("clasificacion") != "Ancla"]
            if not partidos_modificables: continue
            
            num_cambios = random.choice([1, 2])
            indices_a_cambiar = random.sample(partidos_modificables, min(num_cambios, len(partidos_modificables)))

            for idx in indices_a_cambiar:
                resultado_actual = nuevos_resultados[idx]
                opciones = ["L", "E", "V"]
                opciones.remove(resultado_actual)
                nuevos_resultados[idx] = random.choice(opciones)
            
            if self._es_movimiento_valido(nuevo_portafolio, quiniela_idx, nuevos_resultados):
                quiniela_modificada = quiniela_original.copy()
                quiniela_modificada["resultados"] = nuevos_resultados
                quiniela_modificada["empates"] = nuevos_resultados.count("E")
                quiniela_modificada["distribución"] = {"L": nuevos_resultados.count("L"), "E": nuevos_resultados.count("E"), "V": nuevos_resultados.count("V")}
                nuevo_portafolio[quiniela_idx] = quiniela_modificada
                return nuevo_portafolio
                
        return None

    def _es_movimiento_valido(self, portafolio: List[Dict[str, Any]], quiniela_idx: int, nuevos_resultados: List[str]) -> bool:
        empates = nuevos_resultados.count("E")
        if not (self.config["EMPATES_MIN"] <= empates <= self.config["EMPATES_MAX"]):
            return False

        max_conc = max(nuevos_resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc > self.config["CONCENTRACION_MAX_GENERAL"]:
            return False
            
        max_conc_inicial = max(nuevos_resultados[:3].count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > self.config["CONCENTRACION_MAX_INICIAL"]:
            return False

        quiniela_modificada = portafolio[quiniela_idx]
        if quiniela_modificada["tipo"] == "Satelite":
            par_id = quiniela_modificada.get("par_id")
            par_quiniela = next((q for i, q in enumerate(portafolio) if q.get("par_id") == par_id and i != quiniela_idx), None)
            if par_quiniela:
                coincidencias = sum(1 for a, b in zip(nuevos_resultados, par_quiniela["resultados"]) if a == b)
                jaccard = coincidencias / 14.0
                if jaccard > self.config["ARQUITECTURA_PORTAFOLIO"]["correlacion_jaccard_max"]:
                    return False
        
        return True

    def _ajuste_final_especifico(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CORRECCIÓN FINAL ESPECÍFICA: Ataca directamente los 2 problemas restantes
        """
        self.logger.info("🎯 Iniciando ajuste ESPECÍFICO para concentración y distribución...")
        portafolio_ajustado = [q.copy() for q in portafolio]

        # PROBLEMA 1: CONCENTRACIÓN MÁXIMA EN QUINIELAS INDIVIDUALES
        self.logger.info("🔧 Paso 1: Corrigiendo concentración individual...")
        portafolio_ajustado = self._corregir_concentracion_individual_agresivo(portafolio_ajustado, partidos)

        # PROBLEMA 2: DISTRIBUCIÓN DIVISORES POR POSICIÓN  
        self.logger.info("🔧 Paso 2: Corrigiendo distribución por posición...")
        portafolio_ajustado = self._corregir_distribucion_posicion_agresivo(portafolio_ajustado, partidos)

        # VERIFICACIÓN FINAL
        concentracion_ok = self.validator._validar_concentracion_70_60(portafolio_ajustado)
        distribucion_ok = self.validator._validar_distribucion_equilibrada(portafolio_ajustado)
        
        self.logger.info(f"📊 Resultado final: Concentración={concentracion_ok}, Distribución={distribucion_ok}")
        
        return portafolio_ajustado

    def _corregir_concentracion_individual_agresivo(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CORRECCIÓN AGRESIVA: Fuerza que cada quiniela cumpla concentración ≤70% y ≤60% primeros 3
        """
        anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        
        for i, quiniela in enumerate(portafolio):
            if quiniela["tipo"] == "Core":  # No tocar Core
                continue
                
            resultados = quiniela["resultados"].copy()
            modificado = False
            
            # CORRECCIÓN 1: Concentración general ≤70%
            max_permitido_general = int(14 * self.config["CONCENTRACION_MAX_GENERAL"])  # 9
            for signo in ["L", "E", "V"]:
                count_signo = resultados.count(signo)
                if count_signo > max_permitido_general:
                    # Cambiar los menos probables de este signo
                    exceso = count_signo - max_permitido_general
                    indices_signo = [idx for idx, r in enumerate(resultados) if r == signo and idx not in anclas_indices]
                    
                    # Ordenar por probabilidad (cambiar los menos probables)
                    indices_signo.sort(key=lambda idx: partidos[idx][f"prob_{self._resultado_a_clave(signo)}"])
                    
                    for j in range(min(exceso, len(indices_signo))):
                        idx = indices_signo[j]
                        # Cambiar al resultado más probable que no sea el actual
                        opciones = [(s, partidos[idx][f"prob_{self._resultado_a_clave(s)}"]) for s in ["L", "E", "V"] if s != signo]
                        mejor_opcion = max(opciones, key=lambda x: x[1])[0]
                        resultados[idx] = mejor_opcion
                        modificado = True
                        self.logger.debug(f"Corregido concentración general en {quiniela['id']}, posición {idx}: {signo}→{mejor_opcion}")

            # CORRECCIÓN 2: Concentración inicial ≤60%
            primeros_3 = resultados[:3]
            max_permitido_inicial = int(3 * self.config["CONCENTRACION_MAX_INICIAL"])  # 1 (60% de 3 = 1.8, redondeado a 1)
            
            for signo in ["L", "E", "V"]:
                count_inicial = primeros_3.count(signo)
                if count_inicial > max_permitido_inicial:
                    # Cambiar en los primeros 3 partidos
                    exceso = count_inicial - max_permitido_inicial
                    indices_inicial = [idx for idx in range(3) if resultados[idx] == signo and idx not in anclas_indices]
                    
                    # Ordenar por probabilidad
                    indices_inicial.sort(key=lambda idx: partidos[idx][f"prob_{self._resultado_a_clave(signo)}"])
                    
                    for j in range(min(exceso, len(indices_inicial))):
                        idx = indices_inicial[j]
                        opciones = [(s, partidos[idx][f"prob_{self._resultado_a_clave(s)}"]) for s in ["L", "E", "V"] if s != signo]
                        mejor_opcion = max(opciones, key=lambda x: x[1])[0]
                        resultados[idx] = mejor_opcion
                        modificado = True
                        self.logger.debug(f"Corregido concentración inicial en {quiniela['id']}, posición {idx}: {signo}→{mejor_opcion}")

            # Actualizar quiniela si se modificó
            if modificado:
                portafolio[i]["resultados"] = resultados
                portafolio[i]["empates"] = resultados.count("E")
                portafolio[i]["distribución"] = {
                    "L": resultados.count("L"),
                    "E": resultados.count("E"),
                    "V": resultados.count("V")
                }

        return portafolio

    def _corregir_distribucion_posicion_agresivo(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CORRECCIÓN AGRESIVA: Balancea cada posición para que ningún resultado domine excesivamente
        """
        total_quinielas = len(portafolio)
        max_apariciones_posicion = int(total_quinielas * 0.67)  # 67% máximo por posición
        
        for posicion in range(14):
            # Contar resultados en esta posición
            conteos = {"L": 0, "E": 0, "V": 0}
            for q in portafolio:
                conteos[q["resultados"][posicion]] += 1
            
            # Encontrar violaciones
            for signo, count in conteos.items():
                if count > max_apariciones_posicion:
                    exceso = count - max_apariciones_posicion
                    self.logger.debug(f"Posición {posicion+1}: {signo} aparece {count} veces (máx {max_apariciones_posicion})")
                    
                    # Buscar quinielas Satélite que tengan este signo en esta posición
                    candidatos = []
                    for i, q in enumerate(portafolio):
                        if (q["tipo"] == "Satelite" and 
                            q["resultados"][posicion] == signo and 
                            partidos[posicion].get("clasificacion") != "Ancla"):
                            
                            # Priorizar las menos probables
                            prob_actual = partidos[posicion][f"prob_{self._resultado_a_clave(signo)}"]
                            candidatos.append((i, prob_actual))
                    
                    # Ordenar por probabilidad (cambiar primero los menos probables)
                    candidatos.sort(key=lambda x: x[1])
                    
                    # Cambiar hasta corregir el exceso
                    cambios_realizados = 0
                    for q_idx, _ in candidatos:
                        if cambios_realizados >= exceso:
                            break
                            
                        quiniela = portafolio[q_idx]
                        resultados_nuevos = quiniela["resultados"].copy()
                        
                        # Intentar cambiar a un resultado menos usado en esta posición
                        menos_usado = min(conteos, key=conteos.get)
                        resultados_test = resultados_nuevos.copy()
                        resultados_test[posicion] = menos_usado
                        
                        # Verificar que el cambio sea válido
                        if self._es_movimiento_valido(portafolio, q_idx, resultados_test):
                            # Aplicar el cambio
                            portafolio[q_idx]["resultados"] = resultados_test
                            portafolio[q_idx]["empates"] = resultados_test.count("E")
                            portafolio[q_idx]["distribución"] = {
                                "L": resultados_test.count("L"),
                                "E": resultados_test.count("E"),
                                "V": resultados_test.count("V")
                            }
                            
                            # Actualizar conteos
                            conteos[signo] -= 1
                            conteos[menos_usado] += 1
                            cambios_realizados += 1
                            
                            self.logger.debug(f"Posición {posicion+1}: Cambiado {quiniela['id']} de {signo} a {menos_usado}")

        return portafolio

    def _precalcular_matrices_probabilidades(self, partidos: List[Dict[str, Any]]):
        self.probabilidades_matrix = np.zeros((14, 3))
        for i, partido in enumerate(partidos):
            self.probabilidades_matrix[i, 0] = partido["prob_local"]
            self.probabilidades_matrix[i, 1] = partido["prob_empate"]
            self.probabilidades_matrix[i, 2] = partido["prob_visitante"]
            
    def _calcular_objetivo_f_optimizado(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> float:
        cache_key = self._crear_cache_key(portafolio)
        if cache_key in self.cache_probabilidades:
            self.cache_hits += 1
            return self.cache_probabilidades[cache_key]
        self.cache_misses += 1
        
        producto = 1.0
        for quiniela in portafolio:
            prob_11_plus = self._calcular_prob_11_montecarlo_rapido(quiniela["resultados"], partidos)
            producto *= (1 - prob_11_plus)
        resultado = 1 - producto
        
        self.cache_probabilidades[cache_key] = resultado
        if len(self.cache_probabilidades) > 2000:
            self.cache_probabilidades.clear()
        return resultado

    def _calcular_prob_11_montecarlo_rapido(self, resultados: List[str], partidos: List[Dict[str, Any]]) -> float:
        num_simulaciones = 1000
        prob_acierto = np.array([partidos[i][f"prob_{self._resultado_a_clave(res)}"] for i, res in enumerate(resultados)])
        simulaciones = np.random.rand(num_simulaciones, 14)
        aciertos = np.sum(simulaciones < prob_acierto, axis=1)
        aciertos_11_plus = np.sum(aciertos >= 11)
        return aciertos_11_plus / num_simulaciones

    def _crear_cache_key(self, portafolio: List[Dict[str, Any]]) -> int:
        return hash(tuple("".join(q["resultados"]) for q in portafolio))