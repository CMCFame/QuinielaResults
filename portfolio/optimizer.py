# progol_optimizer/portfolio/optimizer.py
"""
Optimizador GRASP-Annealing - Implementación EXACTA de la página 5
Maximiza F = 1 - ∏(1 - Pr[≥11]) usando búsqueda GRASP + Simulated Annealing
"""

import logging
import random
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import combinations

# --- INICIO DE CAMBIOS: Implementación de Fallback ---
# Se intenta importar el método preciso. Si falla, se usará la simulación.
try:
    from scipy.stats import poisson_binomial
    POISSON_BINOMIAL_AVAILABLE = True
except ImportError:
    POISSON_BINOMIAL_AVAILABLE = False
# --- FIN DE CAMBIOS ---

class GRASPAnnealing:
    """
    Implementa optimización GRASP-Annealing según especificaciones del documento
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG["OPTIMIZACION"]

        self.max_iteraciones = self.config["max_iteraciones"]
        self.temperatura_inicial = self.config["temperatura_inicial"]
        self.factor_enfriamiento = self.config["factor_enfriamiento"]
        self.alpha_grasp = self.config["alpha_grasp"]

        self.logger.debug(f"Optimizador GRASP-Annealing configurado: "
                         f"max_iter={self.max_iteraciones}, T0={self.temperatura_inicial}")
        
        # --- INICIO DE CAMBIOS: Log para el método de cálculo ---
        if POISSON_BINOMIAL_AVAILABLE:
            self.logger.info("Usando método de cálculo Poisson-Binomial (preciso y rápido).")
        else:
            self.logger.warning("Función 'poisson_binomial' no disponible. Usando simulación Monte Carlo como fallback.")
        # --- FIN DE CAMBIOS ---

    def optimizar_portafolio_grasp_annealing(self, quinielas_iniciales: List[Dict[str, Any]],
                                           partidos: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Implementa GRASP-Annealing de página 5 para maximizar:
        F = 1 - ∏(q=1 to 30)(1 - Pr_q[≥11])
        """
        self.logger.info("Iniciando optimización GRASP-Annealing...")

        if len(quinielas_iniciales) != 30:
            raise ValueError(f"Se requieren exactamente 30 quinielas, recibidas: {len(quinielas_iniciales)}")

        # Configuración inicial
        mejor_portafolio = [q.copy() for q in quinielas_iniciales]
        mejor_score = self._calcular_objetivo_f(mejor_portafolio, partidos)

        temperatura = self.temperatura_inicial
        iteraciones_sin_mejora = 0

        self.logger.info(f"Score inicial: F={mejor_score:.6f}")

        # Loop principal GRASP-Annealing
        for iteracion in range(self.max_iteraciones):
            candidatos = self._generar_candidatos_vecinos(mejor_portafolio, partidos)
            candidatos_top = self._seleccionar_top_alpha(candidatos, partidos)

            if not candidatos_top:
                continue

            nuevo_portafolio = random.choice(candidatos_top)
            nuevo_score = self._calcular_objetivo_f(nuevo_portafolio, partidos)
            delta = nuevo_score - mejor_score

            if delta > 0 or random.random() < math.exp(delta / temperatura):
                if delta > 0:
                    iteraciones_sin_mejora = 0
                mejor_portafolio = nuevo_portafolio
                mejor_score = nuevo_score
            else:
                iteraciones_sin_mejora += 1
            
            if progress_callback and iteracion % 10 == 0:
                progreso_actual = iteracion / self.max_iteraciones
                texto_progreso = f"Iter. {iteracion}/{self.max_iteraciones} | Score: {mejor_score:.5f}"
                progress_callback(progreso_actual, texto_progreso)

            if iteracion % 100 == 0 and iteracion > 0:
                temperatura *= self.factor_enfriamiento

            if iteraciones_sin_mejora > self.config["iteraciones_sin_mejora"]:
                self.logger.info(f"Parada temprana en iteración {iteracion} (sin mejora)")
                break

        score_final = self._calcular_objetivo_f(mejor_portafolio, partidos)
        self.logger.info(f"✅ Optimización completada: F={score_final:.6f}")
        return mejor_portafolio

    def _calcular_objetivo_f(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> float:
        """Calcula F = 1 - ∏(1 - Pr[≥11]) según página 5."""
        producto = 1.0
        for quiniela in portafolio:
            prob_11_plus = self._calcular_prob_11_aciertos(quiniela["resultados"], partidos)
            producto *= (1 - prob_11_plus)
        return 1 - producto

    # --- INICIO DE CAMBIOS: Función unificada con fallback ---
    def _calcular_prob_11_aciertos(self, resultados: List[str], partidos: List[Dict[str, Any]]) -> float:
        """
        Calcula la probabilidad de obtener ≥11 aciertos.
        Usa Poisson-Binomial si está disponible, de lo contrario usa Monte Carlo.
        """
        if POISSON_BINOMIAL_AVAILABLE:
            # MÉTODO PRECISO Y RÁPIDO
            probabilidades_acierto = []
            for i, resultado_predicho in enumerate(resultados):
                partido = partidos[i]
                if resultado_predicho == "L":
                    probabilidades_acierto.append(partido["prob_local"])
                elif resultado_predicho == "E":
                    probabilidades_acierto.append(partido["prob_empate"])
                else:
                    probabilidades_acierto.append(partido["prob_visitante"])

            mu = poisson_binomial(p=np.array(probabilidades_acierto))
            return mu.sf(k=10)  # Pr[X >= 11]
        else:
            # MÉTODO DE RESPALDO (MONTE CARLO)
            num_simulaciones = 1000
            aciertos_11_plus = 0
            for _ in range(num_simulaciones):
                aciertos = 0
                for i, resultado_predicho in enumerate(resultados):
                    partido = partidos[i]
                    rand = random.random()
                    if rand < partido["prob_local"]:
                        resultado_real = "L"
                    elif rand < partido["prob_local"] + partido["prob_empate"]:
                        resultado_real = "E"
                    else:
                        resultado_real = "V"
                    if resultado_predicho == resultado_real:
                        aciertos += 1
                if aciertos >= 11:
                    aciertos_11_plus += 1
            return aciertos_11_plus / num_simulaciones
    # --- FIN DE CAMBIOS ---

    def _generar_candidatos_vecinos(self, portafolio_actual: List[Dict[str, Any]],
                                  partidos: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Genera vecindario de candidatos mediante swaps de 1-3 partidos."""
        candidatos = []
        for quiniela_idx in range(len(portafolio_actual)):
            quiniela = portafolio_actual[quiniela_idx]
            if quiniela["tipo"] == "Core":
                continue

            for num_cambios in [1, 2]:
                for partidos_indices in combinations(range(14), num_cambios):
                    nuevo_portafolio = [q.copy() for q in portafolio_actual]
                    nueva_quiniela = quiniela.copy()
                    nuevos_resultados = nueva_quiniela["resultados"].copy()
                    
                    for partido_idx in partidos_indices:
                        partido = partidos[partido_idx]
                        if partido["clasificacion"] == "Ancla":
                            continue
                        resultado_actual = nuevos_resultados[partido_idx]
                        nuevo_resultado = self._obtener_resultado_alternativo(resultado_actual, partido)
                        nuevos_resultados[partido_idx] = nuevo_resultado

                    if self._es_quiniela_valida(nuevos_resultados):
                        nueva_quiniela["resultados"] = nuevos_resultados
                        nueva_quiniela["empates"] = nuevos_resultados.count("E")
                        nuevo_portafolio[quiniela_idx] = nueva_quiniela
                        candidatos.append(nuevo_portafolio)
                        if len(candidatos) >= 50:
                            return candidatos
        return candidatos

    def _obtener_resultado_alternativo(self, resultado_actual: str, partido: Dict[str, Any]) -> str:
        """Obtiene un resultado alternativo válido diferente al actual."""
        opciones = ["L", "E", "V"]
        opciones.remove(resultado_actual)
        probs_alternativas = {op: partido[f"prob_{op.lower()}"] for op in opciones}
        return max(probs_alternativas, key=probs_alternativas.get)

    def _es_quiniela_valida(self, resultados: List[str]) -> bool:
        """Verifica que una quiniela cumple las reglas básicas."""
        from config.constants import PROGOL_CONFIG
        empates = resultados.count("E")
        return PROGOL_CONFIG["EMPATES_MIN"] <= empates <= PROGOL_CONFIG["EMPATES_MAX"]

    def _seleccionar_top_alpha(self, candidatos: List[List[Dict[str, Any]]],
                              partidos: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Selecciona el top α% de candidatos según valor de función objetivo."""
        if not candidatos:
            return []
        
        candidatos_con_score = [(c, self._calcular_objetivo_f(c, partidos)) for c in candidatos]
        candidatos_con_score.sort(key=lambda x: x[1], reverse=True)
        
        num_top = max(1, int(len(candidatos_con_score) * self.alpha_grasp))
        return [c for c, _ in candidatos_con_score[:num_top]]
