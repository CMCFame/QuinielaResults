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

    def optimizar_portafolio_grasp_annealing(self, quinielas_iniciales: List[Dict[str, Any]],
                                           partidos: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Implementa GRASP-Annealing de página 5 para maximizar:
        F = 1 - ∏(q=1 to 30)(1 - Pr_q[≥11])

        Args:
            quinielas_iniciales: 4 Core + 26 Satélites
            partidos: Partidos con probabilidades calibradas
            progress_callback: Función opcional para reportar el progreso.
            
        Returns:
            List[Dict]: Portafolio optimizado de 30 quinielas
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
            # Fase GRASP: Construcción golosa + aleatoriedad
            candidatos = self._generar_candidatos_vecinos(mejor_portafolio, partidos)
            candidatos_top = self._seleccionar_top_alpha(candidatos, partidos)

            if not candidatos_top:
                self.logger.debug(f"Iteración {iteracion}: sin candidatos válidos")
                continue

            nuevo_portafolio = random.choice(candidatos_top)

            # Fase Annealing: Aceptación probabilística
            nuevo_score = self._calcular_objetivo_f(nuevo_portafolio, partidos)
            delta = nuevo_score - mejor_score

            # Decisión de aceptación
            if delta > 0 or random.random() < math.exp(delta / temperatura):
                if delta > 0:
                    self.logger.debug(f"Iteración {iteracion}: mejora F={nuevo_score:.6f} (Δ={delta:.6f})")
                    iteraciones_sin_mejora = 0
                else:
                    self.logger.debug(f"Iteración {iteracion}: aceptación probabilística (T={temperatura:.4f})")

                mejor_portafolio = nuevo_portafolio
                mejor_score = nuevo_score
            else:
                iteraciones_sin_mejora += 1
            
            # Reportar progreso cada 10 iteraciones para no ralentizar demasiado
            if progress_callback and iteracion % 10 == 0:
                progreso_actual = iteracion / self.max_iteraciones
                texto_progreso = f"Iter. {iteracion}/{self.max_iteraciones} | Score: {mejor_score:.5f}"
                # Llama al callback con el progreso y el texto
                progress_callback(progreso_actual, texto_progreso)

            # Enfriamiento
            if iteracion % 100 == 0 and iteracion > 0:
                temperatura *= self.factor_enfriamiento
                self.logger.debug(f"Enfriamiento: T={temperatura:.4f}")

            # Criterio de parada temprana
            if iteraciones_sin_mejora > self.config["iteraciones_sin_mejora"]:
                self.logger.info(f"Parada temprana en iteración {iteracion} (sin mejora)")
                break

        score_final = self._calcular_objetivo_f(mejor_portafolio, partidos)
        
        self.logger.info(f"✅ Optimización completada: F={score_final:.6f}")
        
        return mejor_portafolio

    def _calcular_objetivo_f(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> float:
        """
        Calcula F = 1 - ∏(1 - Pr[≥11]) según página 5
        """
        producto = 1.0

        for quiniela in portafolio:
            prob_11_plus = self._calcular_prob_11_aciertos(quiniela["resultados"], partidos)
            producto *= (1 - prob_11_plus)

        F = 1 - producto
        return F

    def _calcular_prob_11_aciertos(self, resultados: List[str], partidos: List[Dict[str, Any]]) -> float:
        """
        Calcula la probabilidad de obtener ≥11 aciertos para una quiniela
        usando simulación Monte Carlo
        """
        num_simulaciones = 1000
        aciertos_11_plus = 0

        for _ in range(num_simulaciones):
            aciertos = 0

            for i, resultado_predicho in enumerate(resultados):
                partido = partidos[i]

                # Generar resultado real según probabilidades
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

    def _generar_candidatos_vecinos(self, portafolio_actual: List[Dict[str, Any]],
                                  partidos: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Genera vecindario de candidatos mediante swaps de 1-3 partidos
        """
        candidatos = []

        for quiniela_idx in range(len(portafolio_actual)):
            quiniela = portafolio_actual[quiniela_idx]

            # Solo modificar satélites (preservar Core según sea posible)
            if quiniela["tipo"] == "Core":
                continue

            # Generar variaciones cambiando 1-2 partidos
            for num_cambios in [1, 2]:
                for partidos_indices in combinations(range(14), num_cambios):
                    nuevo_portafolio = [q.copy() for q in portafolio_actual]
                    nueva_quiniela = quiniela.copy()
                    nuevos_resultados = nueva_quiniela["resultados"].copy()

                    # Aplicar cambios
                    for partido_idx in partidos_indices:
                        partido = partidos[partido_idx]

                        # No tocar ANCLAS
                        if partido["clasificacion"] == "Ancla":
                            continue

                        # Cambiar a resultado alternativo
                        resultado_actual = nuevos_resultados[partido_idx]
                        nuevo_resultado = self._obtener_resultado_alternativo(resultado_actual, partido)
                        nuevos_resultados[partido_idx] = nuevo_resultado

                    # Validar que sigue siendo válida
                    if self._es_quiniela_valida(nuevos_resultados):
                        nueva_quiniela["resultados"] = nuevos_resultados
                        nueva_quiniela["empates"] = nuevos_resultados.count("E")
                        nuevo_portafolio[quiniela_idx] = nueva_quiniela
                        candidatos.append(nuevo_portafolio)

                        if len(candidatos) >= 50:  # Limitar candidatos para eficiencia
                            return candidatos

        return candidatos

    def _obtener_resultado_alternativo(self, resultado_actual: str, partido: Dict[str, Any]) -> str:
        """
        Obtiene un resultado alternativo válido diferente al actual
        """
        opciones = ["L", "E", "V"]
        opciones.remove(resultado_actual)

        # Preferir por probabilidad
        probs_alternativas = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }

        # Filtrar y ordenar opciones restantes
        opciones_validas = [(op, probs_alternativas[op]) for op in opciones]
        opciones_validas.sort(key=lambda x: x[1], reverse=True)

        return opciones_validas[0][0]

    def _es_quiniela_valida(self, resultados: List[str]) -> bool:
        """
        Verifica que una quiniela cumple las reglas básicas
        """
        empates = resultados.count("E")

        # Importar límites
        from config.constants import PROGOL_CONFIG
        min_empates = PROGOL_CONFIG["EMPATES_MIN"]
        max_empates = PROGOL_CONFIG["EMPATES_MAX"]

        return min_empates <= empates <= max_empates

    def _seleccionar_top_alpha(self, candidatos: List[List[Dict[str, Any]]],
                              partidos: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Selecciona el top α% de candidatos según valor de función objetivo
        """
        if not candidatos:
            return []

        # Calcular scores para todos los candidatos
        candidatos_con_score = []
        for candidato in candidatos:
            score = self._calcular_objetivo_f(candidato, partidos)
            candidatos_con_score.append((candidato, score))

        # Ordenar por score descendente
        candidatos_con_score.sort(key=lambda x: x[1], reverse=True)

        # Seleccionar top α%
        num_top = max(1, int(len(candidatos_con_score) * self.alpha_grasp))
        top_candidatos = [candidato for candidato, _ in candidatos_con_score[:num_top]]

        return top_candidatos