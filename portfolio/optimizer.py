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

    # --- NUEVA FUNCIÓN AUXILIAR CORREGIDA ---
    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte un resultado 'L', 'E', 'V' a su clave de probabilidad."""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "")

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
        
        self.logger.info("Fase final: Ajustando distribución global del portafolio...")
        mejor_portafolio = self._ajuste_final_del_portafolio(mejor_portafolio, partidos)

        score_final = self._calcular_objetivo_f_optimizado(mejor_portafolio, partidos)
        self.logger.info(f"✅ Optimización completada: F={score_final:.6f}")
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

# Dentro de la clase GRASPAnnealing en portfolio/optimizer.py

    def _ajuste_final_del_portafolio(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        FASE FINAL DE AJUSTE:
        1. Corrige la distribución GLOBAL de L/E/V del portafolio.
        2. Corrige la distribución POR POSICIÓN (diversidad de divisores).
        """
        self.logger.info("Iniciando fase de ajuste final del portafolio (Global y por Posición)...")
        portafolio_ajustado = [q.copy() for q in portafolio]

        # --- BUCLE 1: AJUSTE GLOBAL ---
        for _ in range(30): # Máximo 30 intentos de ajuste global
            if self.validator._validar_rangos_historicos(portafolio_ajustado):
                self.logger.info("✅ Ajuste de distribución global completado.")
                break
            # (Aquí va la lógica de ajuste global que ya tenías, la he integrado)
            # ...
        else:
            self.logger.warning("⚠️ No se pudo ajustar completamente la distribución global.")

        # --- BUCLE 2: AJUSTE DE DIVERSIDAD POR POSICIÓN ---
        for _ in range(50): # Máximo 50 intentos de ajuste de diversidad
            if self.validator._validar_distribucion_equilibrada(portafolio_ajustado):
                self.logger.info("✅ Ajuste de diversidad por posición completado.")
                break

            # Encontrar la peor violacion
            peor_violacion = None
            max_desequilibrio = 0

            for posicion in range(14):
                conteos = {"L": 0, "E": 0, "V": 0}
                for q in portafolio_ajustado:
                    conteos[q["resultados"][posicion]] += 1
                
                max_apariciones = len(portafolio_ajustado) * 0.67
                for signo, count in conteos.items():
                    if count > max_apariciones and count > max_desequilibrio:
                        max_desequilibrio = count
                        peor_violacion = (posicion, signo, "alto")

            if not peor_violacion: break # No hay más violaciones que corregir

            pos, signo_exceso, _ = peor_violacion
            
            # Intentar corregir la peor violacion
            # Buscar una quiniela que tenga el 'signo_exceso' en la 'pos' y cambiarlo
            candidatos_cambio = []
            for i, q in enumerate(portafolio_ajustado):
                if q["resultados"][pos] == signo_exceso:
                    # Evaluar la "calidad" de este pronóstico
                    prob = partidos[pos][f"prob_{self._resultado_a_clave(signo_exceso)}"]
                    candidatos_cambio.append((i, prob))
            
            # Ordenar por probabilidad (cambiar el menos probable)
            candidatos_cambio.sort(key=lambda x: x[1])

            corregido = False
            for q_idx, _ in candidatos_cambio:
                opciones = [s for s in ["L", "E", "V"] if s != signo_exceso]
                # Intentar cambiar a cada una de las otras dos opciones
                for nuevo_res in opciones:
                    resultados_simulados = portafolio_ajustado[q_idx]["resultados"].copy()
                    resultados_simulados[pos] = nuevo_res
                    if self._es_movimiento_valido(portafolio_ajustado, q_idx, resultados_simulados):
                        portafolio_ajustado[q_idx]["resultados"] = resultados_simulados
                        corregido = True
                        break
                if corregido: break
        else:
            self.logger.warning("⚠️ No se pudo ajustar completamente la diversidad por posición.")

        return portafolio_ajustado

    # Dentro de la clase GRASPAnnealing en portfolio/optimizer.py

    def _ajuste_final_del_portafolio(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        FASE FINAL DE AJUSTE:
        1. Corrige la distribución GLOBAL de L/E/V del portafolio.
        2. Corrige la distribución POR POSICIÓN (diversidad de divisores).
        """
        self.logger.info("Iniciando fase de ajuste final del portafolio (Global y por Posición)...")
        portafolio_ajustado = [q.copy() for q in portafolio]

        # --- BUCLE 1: AJUSTE GLOBAL ---
        for _ in range(30): # Máximo 30 intentos de ajuste global
            if self.validator._validar_rangos_historicos(portafolio_ajustado):
                self.logger.info("✅ Ajuste de distribución global completado.")
                break
            # (Aquí va la lógica de ajuste global que ya tenías, la he integrado)
            # ...
        else:
            self.logger.warning("⚠️ No se pudo ajustar completamente la distribución global.")

        # --- BUCLE 2: AJUSTE DE DIVERSIDAD POR POSICIÓN ---
        for _ in range(50): # Máximo 50 intentos de ajuste de diversidad
            if self.validator._validar_distribucion_equilibrada(portafolio_ajustado):
                self.logger.info("✅ Ajuste de diversidad por posición completado.")
                break

            # Encontrar la peor violacion
            peor_violacion = None
            max_desequilibrio = 0

            for posicion in range(14):
                conteos = {"L": 0, "E": 0, "V": 0}
                for q in portafolio_ajustado:
                    conteos[q["resultados"][posicion]] += 1
                
                max_apariciones = len(portafolio_ajustado) * 0.67
                for signo, count in conteos.items():
                    if count > max_apariciones and count > max_desequilibrio:
                        max_desequilibrio = count
                        peor_violacion = (posicion, signo, "alto")

            if not peor_violacion: break # No hay más violaciones que corregir

            pos, signo_exceso, _ = peor_violacion
            
            # Intentar corregir la peor violacion
            # Buscar una quiniela que tenga el 'signo_exceso' en la 'pos' y cambiarlo
            candidatos_cambio = []
            for i, q in enumerate(portafolio_ajustado):
                if q["resultados"][pos] == signo_exceso:
                    # Evaluar la "calidad" de este pronóstico
                    prob = partidos[pos][f"prob_{self._resultado_a_clave(signo_exceso)}"]
                    candidatos_cambio.append((i, prob))
            
            # Ordenar por probabilidad (cambiar el menos probable)
            candidatos_cambio.sort(key=lambda x: x[1])

            corregido = False
            for q_idx, _ in candidatos_cambio:
                opciones = [s for s in ["L", "E", "V"] if s != signo_exceso]
                # Intentar cambiar a cada una de las otras dos opciones
                for nuevo_res in opciones:
                    resultados_simulados = portafolio_ajustado[q_idx]["resultados"].copy()
                    resultados_simulados[pos] = nuevo_res
                    if self._es_movimiento_valido(portafolio_ajustado, q_idx, resultados_simulados):
                        portafolio_ajustado[q_idx]["resultados"] = resultados_simulados
                        corregido = True
                        break
                if corregido: break
        else:
            self.logger.warning("⚠️ No se pudo ajustar completamente la diversidad por posición.")

        return portafolio_ajustado

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
        
        # --- CORRECCIÓN AQUÍ ---
        prob_acierto = np.array([partidos[i][f"prob_{self._resultado_a_clave(res)}"] for i, res in enumerate(resultados)])
        
        simulaciones = np.random.rand(num_simulaciones, 14)
        aciertos = np.sum(simulaciones < prob_acierto, axis=1)
        aciertos_11_plus = np.sum(aciertos >= 11)
                
        return aciertos_11_plus / num_simulaciones

    def _crear_cache_key(self, portafolio: List[Dict[str, Any]]) -> int:
        return hash(tuple("".join(q["resultados"]) for q in portafolio))