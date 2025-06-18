# progol_optimizer/portfolio/optimizer.py - CORRECCIÓN DEFINITIVA
"""
Optimizador GRASP-Annealing - CORRECCIÓN DEFINITIVA
Menos restrictivo durante optimización, más agresivo en ajuste final
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
        self.logger.info("🚀 Iniciando optimización GRASP-Annealing DEFINITIVA...")
        self._precalcular_matrices_probabilidades(partidos)

        portafolio_actual = [q.copy() for q in quinielas_iniciales]
        score_actual = self._calcular_objetivo_f_optimizado(portafolio_actual, partidos)
        
        mejor_portafolio = portafolio_actual
        mejor_score = score_actual

        temperatura = self.temperatura_inicial
        iteraciones_sin_mejora = 0
        self.logger.info(f"Score inicial: F={score_actual:.6f}")

        for iteracion in range(self.max_iteraciones):
            nuevo_portafolio_candidato = self._generar_movimiento_valido_mejorado(portafolio_actual, partidos)

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
                self.logger.info(f"⏹️ Parada temprana en iteración {iteracion} (sin mejora por {iteraciones_sin_mejora} iteraciones)")
                break
        
        self.logger.info("🔧 Fase final: Ajuste agresivo de concentración y distribución...")
        mejor_portafolio = self._ajuste_final_definitivo(mejor_portafolio, partidos)

        score_final = self._calcular_objetivo_f_optimizado(mejor_portafolio, partidos)
        self.logger.info(f"✅ Optimización DEFINITIVA completada: F={score_final:.6f}")
        return mejor_portafolio

    def _generar_movimiento_valido_mejorado(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generación de movimientos MENOS restrictiva durante optimización"""
        intentos = 0
        while intentos < 50:  # Más intentos
            intentos += 1
            
            nuevo_portafolio = [q.copy() for q in portafolio]
            quiniela_idx = random.randrange(len(nuevo_portafolio))
            
            # Priorizar satélites (80% de probabilidad)
            if random.random() < 0.8:
                satelite_indices = [i for i, q in enumerate(nuevo_portafolio) if q['tipo'] == 'Satelite']
                if satelite_indices:
                    quiniela_idx = random.choice(satelite_indices)
            
            quiniela_original = nuevo_portafolio[quiniela_idx]
            nuevos_resultados = quiniela_original["resultados"].copy()
            
            partidos_modificables = [i for i, p in enumerate(partidos) if p.get("clasificacion") != "Ancla"]
            if not partidos_modificables: 
                continue
            
            # Cambios más pequeños para mayor éxito
            num_cambios = random.choice([1, 1, 2])  # Más probabilidad de 1 cambio
            indices_a_cambiar = random.sample(partidos_modificables, min(num_cambios, len(partidos_modificables)))

            for idx in indices_a_cambiar:
                resultado_actual = nuevos_resultados[idx]
                opciones = ["L", "E", "V"]
                opciones.remove(resultado_actual)
                nuevos_resultados[idx] = random.choice(opciones)
            
            # Validación MENOS estricta durante optimización
            if self._es_movimiento_valido_permisivo(nuevo_portafolio, quiniela_idx, nuevos_resultados):
                quiniela_modificada = quiniela_original.copy()
                quiniela_modificada["resultados"] = nuevos_resultados
                quiniela_modificada["empates"] = nuevos_resultados.count("E")
                quiniela_modificada["distribución"] = {
                    "L": nuevos_resultados.count("L"), 
                    "E": nuevos_resultados.count("E"), 
                    "V": nuevos_resultados.count("V")
                }
                nuevo_portafolio[quiniela_idx] = quiniela_modificada
                return nuevo_portafolio
                
        return None

    def _es_movimiento_valido_permisivo(self, portafolio: List[Dict[str, Any]], quiniela_idx: int, nuevos_resultados: List[str]) -> bool:
        """Validación PERMISIVA durante optimización (el ajuste final corregirá problemas)"""
        
        # 1. Validar empates (obligatorio)
        empates = nuevos_resultados.count("E")
        if not (self.config["EMPATES_MIN"] <= empates <= self.config["EMPATES_MAX"]):
            return False

        # 2. Concentración MÁS PERMISIVA durante optimización (85% vs 70%)
        max_conc = max(nuevos_resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc > 0.85:  # 85% en lugar de 70%
            return False
            
        # 3. Concentración inicial MÁS PERMISIVA (85% vs 60%)
        max_conc_inicial = max(nuevos_resultados[:3].count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > 0.85:  # 85% en lugar de 60%
            return False

        # 4. Validar correlación Jaccard para satélites
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

    def _ajuste_final_definitivo(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        AJUSTE FINAL DEFINITIVO: Corrige TODO lo necesario para pasar validación
        """
        self.logger.info("🔧 Iniciando ajuste final DEFINITIVO...")
        portafolio_ajustado = [q.copy() for q in portafolio]

        # PASO 1: Corrección AGRESIVA de concentración individual
        self.logger.info("🎯 Paso 1: Corrección AGRESIVA de concentración individual...")
        for iter_conc in range(3):  # Múltiples pasadas
            cambios_realizados = 0
            for i, quiniela in enumerate(portafolio_ajustado):
                if quiniela["tipo"] == "Core":
                    continue
                    
                resultados_originales = quiniela["resultados"].copy()
                resultados_corregidos = self._forzar_concentracion_valida(resultados_originales, partidos)
                
                if resultados_originales != resultados_corregidos:
                    portafolio_ajustado[i]["resultados"] = resultados_corregidos
                    portafolio_ajustado[i]["empates"] = resultados_corregidos.count("E")
                    portafolio_ajustado[i]["distribución"] = {
                        "L": resultados_corregidos.count("L"),
                        "E": resultados_corregidos.count("E"),
                        "V": resultados_corregidos.count("V")
                    }
                    cambios_realizados += 1
            
            self.logger.debug(f"Iteración concentración {iter_conc+1}: {cambios_realizados} cambios")
            if cambios_realizados == 0:
                break

        # PASO 2: Corrección AGRESIVA de distribución por posición
        self.logger.info("🎯 Paso 2: Corrección AGRESIVA de distribución por posición...")
        for iter_dist in range(5):  # Múltiples pasadas
            cambios_realizados = 0
            
            # Analizar cada posición
            for posicion in range(14):
                cambios_realizados += self._balancear_posicion_agresivo(portafolio_ajustado, posicion, partidos)
            
            self.logger.debug(f"Iteración distribución {iter_dist+1}: {cambios_realizados} cambios")
            if cambios_realizados == 0:
                break

        # PASO 3: Verificación final
        concentracion_ok = self.validator._validar_concentracion_70_60(portafolio_ajustado)
        distribucion_ok = self.validator._validar_distribucion_equilibrada(portafolio_ajustado)
        
        self.logger.info(f"📊 Resultado final: Concentración={concentracion_ok}, Distribución={distribucion_ok}")
        
        return portafolio_ajustado

    def _forzar_concentracion_valida(self, resultados: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """
        FUERZA que una quiniela tenga concentración válida ≤70% general y ≤60% primeros 3
        """
        resultados_corregidos = resultados.copy()
        anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Corrección 1: Concentración general ≤70% (máximo 9 de 14)
        max_permitido_general = int(14 * 0.70)  # 9
        for signo in ["L", "E", "V"]:
            count_signo = resultados_corregidos.count(signo)
            if count_signo > max_permitido_general:
                # Encontrar índices de este signo que se pueden modificar
                indices_modificables = [i for i in modificables if resultados_corregidos[i] == signo]
                # Ordenar por probabilidad (cambiar los menos probables primero)
                indices_modificables.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                
                # Cambiar hasta que sea válido
                exceso = count_signo - max_permitido_general
                for j in range(min(exceso, len(indices_modificables))):
                    idx = indices_modificables[j]
                    # Cambiar al resultado que menos tengamos
                    otros_signos = [s for s in ["L", "E", "V"] if s != signo]
                    mejor_cambio = min(otros_signos, key=lambda s: resultados_corregidos.count(s))
                    resultados_corregidos[idx] = mejor_cambio

        # Corrección 2: Concentración inicial ≤60% (máximo 1 de 3)
        max_permitido_inicial = int(3 * 0.60)  # 1
        for signo in ["L", "E", "V"]:
            count_inicial = resultados_corregidos[:3].count(signo)
            if count_inicial > max_permitido_inicial:
                # Encontrar índices en primeros 3 partidos
                indices_inicial = [i for i in range(3) if resultados_corregidos[i] == signo and i not in anclas_indices]
                # Ordenar por probabilidad
                indices_inicial.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                
                # Cambiar hasta que sea válido
                exceso = count_inicial - max_permitido_inicial
                for j in range(min(exceso, len(indices_inicial))):
                    idx = indices_inicial[j]
                    # Cambiar al resultado que menos tengamos en los primeros 3
                    otros_signos = [s for s in ["L", "E", "V"] if s != signo]
                    mejor_cambio = min(otros_signos, key=lambda s: resultados_corregidos[:3].count(s))
                    resultados_corregidos[idx] = mejor_cambio

        return resultados_corregidos

    def _balancear_posicion_agresivo(self, portafolio: List[Dict[str, Any]], posicion: int, partidos: List[Dict[str, Any]]) -> int:
        """
        Balancea AGRESIVAMENTE una posición específica
        """
        if partidos[posicion].get("clasificacion") == "Ancla":
            return 0  # No tocar anclas
            
        total_quinielas = len(portafolio)
        max_apariciones = int(total_quinielas * 0.67)  # 67% máximo
        
        # Contar apariciones actuales
        conteos = {"L": 0, "E": 0, "V": 0}
        for q in portafolio:
            conteos[q["resultados"][posicion]] += 1
        
        cambios_realizados = 0
        
        # Encontrar signos que exceden el límite
        for signo, count in conteos.items():
            if count > max_apariciones:
                exceso = count - max_apariciones
                
                # Encontrar candidatos para cambiar (solo satélites)
                candidatos = []
                for i, q in enumerate(portafolio):
                    if q["tipo"] == "Satelite" and q["resultados"][posicion] == signo:
                        prob_actual = partidos[posicion][f"prob_{self._resultado_a_clave(signo)}"]
                        candidatos.append((i, prob_actual))
                
                # Ordenar por probabilidad (cambiar los menos probables)
                candidatos.sort(key=lambda x: x[1])
                
                # Realizar cambios
                for j in range(min(exceso, len(candidatos))):
                    q_idx, _ = candidatos[j]
                    
                    # Encontrar el signo menos usado en esta posición
                    menos_usado = min(conteos, key=conteos.get)
                    
                    # Aplicar cambio
                    quiniela = portafolio[q_idx]
                    resultados_nuevos = quiniela["resultados"].copy()
                    resultados_nuevos[posicion] = menos_usado
                    
                    # Actualizar quiniela
                    portafolio[q_idx]["resultados"] = resultados_nuevos
                    portafolio[q_idx]["empates"] = resultados_nuevos.count("E")
                    portafolio[q_idx]["distribución"] = {
                        "L": resultados_nuevos.count("L"),
                        "E": resultados_nuevos.count("E"),
                        "V": resultados_nuevos.count("V")
                    }
                    
                    # Actualizar conteos
                    conteos[signo] -= 1
                    conteos[menos_usado] += 1
                    cambios_realizados += 1
        
        return cambios_realizados

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