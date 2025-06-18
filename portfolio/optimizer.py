# progol_optimizer/portfolio/optimizer.py
"""
Optimizador GRASP-Annealing OPTIMIZADO y CORREGIDO
AHORA VALIDA TODAS LAS REGLAS DURANTE LA OPTIMIZACIÓN
- Previene que los swaps invaliden el portafolio
- Añade fase de ajuste final para garantizar distribución global
"""

import logging
import random
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import combinations

class GRASPAnnealing:
    """
    Implementa optimización GRASP-Annealing que respeta todas las reglas de validación
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.opt_config = self.config["OPTIMIZACION"]
        
        # Parámetros de optimización
        self.max_iteraciones = self.opt_config["max_iteraciones"]
        self.temperatura_inicial = self.opt_config["temperatura_inicial"]
        self.factor_enfriamiento = self.opt_config["factor_enfriamiento"]
        self.alpha_grasp = self.opt_config["alpha_grasp"]
        self.iteraciones_sin_mejora_max = self.opt_config["iteraciones_sin_mejora"]

        # Cache para probabilidades
        self.cache_probabilidades = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cargar el validador para usar sus reglas
        from validation.portfolio_validator import PortfolioValidator
        self.validator = PortfolioValidator()

    def optimizar_portafolio_grasp_annealing(self, quinielas_iniciales: List[Dict[str, Any]],
                                           partidos: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
        """
        GRASP-Annealing que valida reglas en cada paso
        """
        self.logger.info("🚀 Iniciando optimización GRASP-Annealing CORREGIDA...")
        self._precalcular_matrices_probabilidades(partidos)

        portafolio_actual = [q.copy() for q in quinielas_iniciales]
        score_actual = self._calcular_objetivo_f_optimizado(portafolio_actual, partidos)
        
        mejor_portafolio = portafolio_actual
        mejor_score = score_actual

        temperatura = self.temperatura_inicial
        iteraciones_sin_mejora = 0
        self.logger.info(f"Score inicial: F={score_actual:.6f}")

        # Loop principal
        for iteracion in range(self.max_iteraciones):
            # Generar un movimiento válido (un cambio en el portafolio)
            nuevo_portafolio_candidato = self._generar_movimiento_valido(portafolio_actual, partidos)

            if nuevo_portafolio_candidato is None:
                continue

            nuevo_score = self._calcular_objetivo_f_optimizado(nuevo_portafolio_candidato, partidos)
            delta = nuevo_score - score_actual

            # Criterio de aceptación de Simulated Annealing
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
        
        # FASE FINAL: AJUSTE DE DISTRIBUCIÓN GLOBAL
        self.logger.info("Fase final: Ajustando distribución global del portafolio...")
        mejor_portafolio = self._ajustar_distribucion_global_final(mejor_portafolio, partidos)

        score_final = self._calcular_objetivo_f_optimizado(mejor_portafolio, partidos)
        self.logger.info(f"✅ Optimización completada: F={score_final:.6f}")
        return mejor_portafolio

    def _generar_movimiento_valido(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera un nuevo portafolio candidato que cumpla las reglas individuales y de par.
        """
        intentos = 0
        while intentos < 20:
            intentos += 1
            
            nuevo_portafolio = [q.copy() for q in portafolio]
            
            # Elegir una quiniela al azar para modificar (priorizar satélites)
            quiniela_idx = random.randrange(len(nuevo_portafolio))
            if random.random() < 0.8: # 80% de probabilidad de elegir un satélite
                satelite_indices = [i for i, q in enumerate(nuevo_portafolio) if q['tipo'] == 'Satelite']
                if satelite_indices:
                    quiniela_idx = random.choice(satelite_indices)
            
            quiniela_original = nuevo_portafolio[quiniela_idx]
            nuevos_resultados = quiniela_original["resultados"].copy()
            
            # Elegir 1 o 2 partidos para cambiar (no anclas)
            partidos_modificables = [i for i, p in enumerate(partidos) if p.get("clasificacion") != "Ancla"]
            if not partidos_modificables: continue
            
            num_cambios = random.choice([1, 2])
            indices_a_cambiar = random.sample(partidos_modificables, min(num_cambios, len(partidos_modificables)))

            for idx in indices_a_cambiar:
                resultado_actual = nuevos_resultados[idx]
                opciones = ["L", "E", "V"]
                opciones.remove(resultado_actual)
                nuevos_resultados[idx] = random.choice(opciones)
            
            # AHORA VALIDAMOS EL CAMBIO PROPUESTO
            if self._es_movimiento_valido(nuevo_portafolio, quiniela_idx, nuevos_resultados):
                # Aplicar el cambio
                quiniela_modificada = quiniela_original.copy()
                quiniela_modificada["resultados"] = nuevos_resultados
                quiniela_modificada["empates"] = nuevos_resultados.count("E")
                quiniela_modificada["distribución"] = {"L": nuevos_resultados.count("L"), "E": nuevos_resultados.count("E"), "V": nuevos_resultados.count("V")}
                nuevo_portafolio[quiniela_idx] = quiniela_modificada
                return nuevo_portafolio
                
        return None # No se encontró un movimiento válido

    def _es_movimiento_valido(self, portafolio: List[Dict[str, Any]], quiniela_idx: int, nuevos_resultados: List[str]) -> bool:
        """
        Función CRÍTICA: Valida un cambio contra las reglas individuales y de pares.
        """
        # 1. Validar empates
        empates = nuevos_resultados.count("E")
        if not (self.config["EMPATES_MIN"] <= empates <= self.config["EMPATES_MAX"]):
            return False

        # 2. Validar concentración
        max_conc = max(nuevos_resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc > self.config["CONCENTRACION_MAX_GENERAL"]:
            return False
            
        max_conc_inicial = max(nuevos_resultados[:3].count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > self.config["CONCENTRACION_MAX_INICIAL"]:
            return False

        # 3. Validar correlación Jaccard si es un satélite
        quiniela_modificada = portafolio[quiniela_idx]
        if quiniela_modificada["tipo"] == "Satelite":
            par_id = quiniela_modificada.get("par_id")
            # Encontrar su par
            par_quiniela = next((q for i, q in enumerate(portafolio) if q.get("par_id") == par_id and i != quiniela_idx), None)
            if par_quiniela:
                coincidencias = sum(1 for a, b in zip(nuevos_resultados, par_quiniela["resultados"]) if a == b)
                jaccard = coincidencias / 14.0
                if jaccard > self.config["ARQUITECTURA_PORTAFOLIO"]["correlacion_jaccard_max"]:
                    return False
        
        return True

    def _ajustar_distribucion_global_final(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Hook post-optimización que ajusta el portafolio para cumplir con la distribución global.
        """
        portafolio_ajustado = [q.copy() for q in portafolio]
        max_ajustes = 50
        
        for _ in range(max_ajustes):
            total_L = sum(q["resultados"].count("L") for q in portafolio_ajustado)
            total_E = sum(q["resultados"].count("E") for q in portafolio_ajustado)
            total_V = sum(q["resultados"].count("V") for q in portafolio_ajustado)
            total_partidos = len(portafolio_ajustado) * 14
            
            dist = {"L": total_L/total_partidos, "E": total_E/total_partidos, "V": total_V/total_partidos}
            rangos = self.config["RANGOS_HISTORICOS"]
            
            # Checar qué regla se viola
            violacion = None
            if dist["L"] < rangos["L"][0]: violacion = ("L", "bajo")
            elif dist["L"] > rangos["L"][1]: violacion = ("L", "alto")
            elif dist["E"] < rangos["E"][0]: violacion = ("E", "bajo")
            elif dist["E"] > rangos["E"][1]: violacion = ("E", "alto")
            elif dist["V"] < rangos["V"][0]: violacion = ("V", "bajo")
            elif dist["V"] > rangos["V"][1]: violacion = ("V", "alto")
                
            if violacion is None:
                self.logger.info("✅ Distribución global final está dentro de los rangos.")
                return portafolio_ajustado

            # Lógica de ajuste
            signo, direccion = violacion
            
            if direccion == "alto": # Necesitamos MENOS de este signo
                # Buscar el mejor candidato para cambiar DE este signo A otro
                # Candidato: una quiniela que tenga este signo en un partido de baja probabilidad
                mejor_cambio = None
                menor_prob = float('inf')
                
                for i, q in enumerate(portafolio_ajustado):
                    for j, res in enumerate(q["resultados"]):
                        if res == signo:
                            prob_partido = partidos[j][f"prob_{self.validator._resultado_a_clave(res)}"]
                            if prob_partido < menor_prob:
                                # Proponer un cambio
                                otras_opciones = [s for s in ["L", "E", "V"] if s != signo]
                                nuevo_res = otras_opciones[0] # Simplificado, se puede mejorar
                                
                                # Simular el cambio y ver si es válido
                                resultados_simulados = q["resultados"].copy()
                                resultados_simulados[j] = nuevo_res
                                if self._es_movimiento_valido(portafolio_ajustado, i, resultados_simulados):
                                    menor_prob = prob_partido
                                    mejor_cambio = (i, j, nuevo_res)

                if mejor_cambio:
                    q_idx, p_idx, nuevo_res = mejor_cambio
                    portafolio_ajustado[q_idx]["resultados"][p_idx] = nuevo_res
                else: break # No se encontró cambio válido
            
            else: # Necesitamos MÁS de este signo
                # Inverso: buscar cambiar a este signo desde otro
                mejor_cambio = None
                mayor_prob = 0
                
                for i, q in enumerate(portafolio_ajustado):
                    for j, res in enumerate(q["resultados"]):
                        if res != signo:
                            prob_partido = partidos[j][f"prob_{self.validator._resultado_a_clave(signo)}"]
                            if prob_partido > mayor_prob:
                                resultados_simulados = q["resultados"].copy()
                                resultados_simulados[j] = signo
                                if self._es_movimiento_valido(portafolio_ajustado, i, resultados_simulados):
                                    mayor_prob = prob_partido
                                    mejor_cambio = (i, j, signo)
                
                if mejor_cambio:
                    q_idx, p_idx, nuevo_res = mejor_cambio
                    portafolio_ajustado[q_idx]["resultados"][p_idx] = nuevo_res
                else: break

        self.logger.warning("⚠️ No se pudo ajustar completamente la distribución global tras varios intentos.")
        return portafolio_ajustado


    # --- Las funciones de cálculo de score y pre-cálculo permanecen igual ---
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
        aciertos_11_plus = 0
        
        prob_acierto = np.array([partidos[i][f"prob_{self.validator._resultado_a_clave(res)}"] for i, res in enumerate(resultados)])
        
        simulaciones = np.random.rand(num_simulaciones, 14)
        aciertos = np.sum(simulaciones < prob_acierto, axis=1)
        aciertos_11_plus = np.sum(aciertos >= 11)
                
        return aciertos_11_plus / num_simulaciones

    def _crear_cache_key(self, portafolio: List[Dict[str, Any]]) -> int:
        return hash(tuple("".join(q["resultados"]) for q in portafolio))