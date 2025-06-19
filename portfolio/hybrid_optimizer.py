# progol_optimizer/portfolio/hybrid_optimizer.py
"""
Implementa la estrategia de generación de portafolios Híbrida (IP + GRASP-Annealing).

Este módulo contiene la lógica para resolver el problema de generación de quinielas
utilizando un enfoque de dos fases:
1.  Programación Entera (IP) con el solver CP-SAT de OR-Tools para generar un
    conjunto inicial de quinielas que cumplen con todas las restricciones duras.
2.  Una metaheurística de Recocido Simulado (Simulated Annealing) para mejorar
    iterativamente la solución inicial, optimizando la probabilidad de premio.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import math
import random
import copy
from ortools.sat.python import cp_model

from config.constants import PROGOL_CONFIG

# Configuración del logger para este módulo
logging.basicConfig(level=PROGOL_CONFIG.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class HybridOptimizer:
    """
    Gestiona la generación de portafolios de quinielas mediante un enfoque híbrido.
    """

    def __init__(self, match_data: pd.DataFrame):
        """
        Inicializa el optimizador híbrido.

        Args:
            match_data (pd.DataFrame): DataFrame con los datos de los partidos,
                                       incluyendo probabilidades y clasificación.
        """
        self.match_data = match_data
        self.config = PROGOL_CONFIG.get("HYBRID_OPTIMIZER", {})
        self.architecture = PROGOL_CONFIG.get("ARQUITECTURA_PORTAFOLIO", {})
        self.rules = {
            "empates_min": PROGOL_CONFIG.get("EMPATES_MIN"),
            "empates_max": PROGOL_CONFIG.get("EMPATES_MAX"),
            "rangos_historicos": PROGOL_CONFIG.get("RANGOS_HISTORICOS")
        }
        
        self.num_matches = len(match_data)
        self.num_quinielas = self.architecture.get("num_total", 30)
        self.num_cores = self.architecture.get("num_core", 4)
        
        self.resultados_map = {"L": 0, "E": 1, "V": 2}
        self.resultados_rev_map = {0: "L", 1: "E", 2: "V"}
        
        logger.info("Inicializando HybridOptimizer con %d partidos y para %d quinielas.",
                    self.num_matches, self.num_quinielas)

    def generate_portfolio(self) -> List[Dict[str, Any]]:
        """
        Punto de entrada principal para generar el portafolio completo.
        """
        logger.info("Iniciando generación de portafolio con estrategia Híbrida.")
        
        # --- Fase 1: Generación con Programación Entera (IP) ---
        initial_solution = self._solve_ip_model()
        
        if not initial_solution:
            logger.error("La fase de Programación Entera no pudo encontrar una solución inicial válida.")
            return []

        # --- Fase 2: Mejora con Recocido Simulado (Simulated Annealing) ---
        optimized_solution = self._run_simulated_annealing(initial_solution)
        
        logger.info("Generación de portafolio Híbrido completada.")
        return optimized_solution

    def _build_ip_model(self) -> tuple[cp_model.CpModel, dict]:
        """
        Construye el modelo de Programación Entera (IP) con todas las variables y restricciones.
        """
        logger.info("Construyendo el modelo de Programación Entera (IP)...")
        model = cp_model.CpModel()

        x = {}
        for j in range(self.num_quinielas):
            for i in range(self.num_matches):
                for r_idx, r_str in self.resultados_rev_map.items():
                    x[j, i, r_idx] = model.NewBoolVar(f'x_q{j}_p{i}_r{r_str}')

        for j in range(self.num_quinielas):
            for i in range(self.num_matches):
                model.AddExactlyOne([x[j, i, r_idx] for r_idx in self.resultados_rev_map])

        anchor_matches = self.match_data[self.match_data['clasificacion'] == 'Ancla']
        if not anchor_matches.empty:
            for idx, partido in anchor_matches.iterrows():
                probs = {
                    self.resultados_map['L']: partido['prob_local'],
                    self.resultados_map['E']: partido['prob_empate'],
                    self.resultados_map['V']: partido['prob_visitante']
                }
                best_result_idx = max(probs, key=probs.get)
                for j in range(self.num_cores):
                    model.Add(x[j, idx, best_result_idx] == 1)
            logger.info(f"Aplicada restricción de Anclas a {len(anchor_matches)} partidos para las {self.num_cores} quinielas Core.")

        empates_idx = self.resultados_map['E']
        for j in range(self.num_quinielas):
            empates_en_quiniela = [x[j, i, empates_idx] for i in range(self.num_matches)]
            model.Add(sum(empates_en_quiniela) >= self.rules['empates_min'])
            model.Add(sum(empates_en_quiniela) <= self.rules['empates_max'])
        logger.info(f"Aplicada restricción de {self.rules['empates_min']}-{self.rules['empates_max']} empates por quiniela.")

        total_partidos_portafolio = self.num_quinielas * self.num_matches
        for r_idx, r_str in self.resultados_rev_map.items():
            min_range, max_range = self.rules['rangos_historicos'][r_str]
            min_count = int(np.ceil(total_partidos_portafolio * min_range))
            max_count = int(np.floor(total_partidos_portafolio * max_range))
            total_resultado = [x[j, i, r_idx] for j in range(self.num_quinielas) for i in range(self.num_matches)]
            model.Add(sum(total_resultado) >= min_count)
            model.Add(sum(total_resultado) <= max_count)
            logger.debug(f"Restricción global para '{r_str}': entre {min_count} y {max_count} apariciones.")
        logger.info("Aplicadas restricciones de distribución global para todo el portafolio.")

        objective_terms = []
        for j in range(self.num_quinielas):
            for i in range(self.num_matches):
                for r_idx, r_str in self.resultados_rev_map.items():
                    prob = self.match_data.iloc[i][f'prob_{r_str.lower()}']
                    objective_terms.append(x[j, i, r_idx] * int(prob * 10000))
        model.Maximize(sum(objective_terms))
        logger.info("Establecido objetivo: maximizar la suma de probabilidades esperadas.")

        return model, x

    def _solve_ip_model(self) -> List[Dict[str, Any]]:
        """
        Resuelve el modelo IP y traduce el resultado a un portafolio de quinielas.
        """
        logger.info("Resolviendo el modelo IP para obtener la solución inicial.")
        model, x = self._build_ip_model()
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.get("max_time_in_seconds", 120.0)
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logger.info(f"Solución IP encontrada en {solver.WallTime():.2f} segundos. Traduciendo resultados...")
            
            initial_portfolio = []
            for j in range(self.num_quinielas):
                quiniela_results = []
                for i in range(self.num_matches):
                    for r_idx, r_str in self.resultados_rev_map.items():
                        if solver.BooleanValue(x[j, i, r_idx]):
                            quiniela_results.append(r_str)
                            break
                
                quiniela_type = "Core" if j < self.num_cores else "Satelite"
                par_id = (j - self.num_cores) // 2 if quiniela_type == "Satelite" else None
                sub_id = "A" if quiniela_type == "Satelite" and (j - self.num_cores) % 2 == 0 else ("B" if quiniela_type == "Satelite" else "")
                quiniela_id = f"Core-{j+1}" if quiniela_type == "Core" else f"Sat-{par_id+1}{sub_id}"

                empates = quiniela_results.count("E")
                initial_portfolio.append({
                    "id": quiniela_id, "tipo": quiniela_type, "par_id": par_id,
                    "resultados": quiniela_results, "empates": empates,
                    "distribucion": {"L": quiniela_results.count("L"), "E": empates, "V": quiniela_results.count("V")}
                })
            
            logger.info(f"Portafolio inicial de {len(initial_portfolio)} quinielas generado exitosamente desde el solver.")
            return initial_portfolio
        else:
            logger.error("No se pudo encontrar una solución válida para el modelo IP. El modelo puede ser infactible.")
            return []

    def _run_simulated_annealing(self, initial_solution: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ejecuta el algoritmo de Recocido Simulado para mejorar la solución inicial.
        """
        logger.info("Iniciando fase de mejora con Recocido Simulado.")
        
        # Cargar parámetros del config
        temp = self.config.get("temperature_initial", 1.0)
        cooling_factor = self.config.get("temperature_cooling_factor", 0.99)
        iterations = self.config.get("annealing_iterations", 20000)

        current_portfolio = copy.deepcopy(initial_solution)
        best_portfolio = copy.deepcopy(initial_solution)

        current_score = self._calculate_portfolio_score(current_portfolio)
        best_score = current_score
        
        for i in range(iterations):
            neighbor_portfolio = self._generate_neighbor(current_portfolio)
            
            if not neighbor_portfolio:
                continue

            neighbor_score = self._calculate_portfolio_score(neighbor_portfolio)
            
            delta = neighbor_score - current_score
            
            # Decidir si aceptamos el nuevo portafolio
            if delta > 0 or math.exp(delta / temp) > random.random():
                current_portfolio = copy.deepcopy(neighbor_portfolio)
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_portfolio = copy.deepcopy(current_portfolio)
                    best_score = current_score
                    logger.debug(f"Iter {i}: Nueva mejor solución encontrada! Score: {best_score:.6f}")

            # Enfriar la temperatura
            temp *= cooling_factor
            
            if i % 1000 == 0:
                logger.debug(f"Iter {i}/{iterations} - Temp: {temp:.4f}, Best Score: {best_score:.6f}")

        logger.info(f"Optimización finalizada. Mejor score alcanzado: {best_score:.6f}")
        return best_portfolio

    def _calculate_portfolio_score(self, portfolio: List[Dict[str, Any]]) -> float:
        """
        Calcula el score del portafolio completo basado en la fórmula F = 1 - ∏(1 - Pr(≥11)).
        """
        producto = 1.0
        for quiniela in portfolio:
            prob_11_plus = self._calculate_prob_11_or_more(quiniela['resultados'])
            producto *= (1.0 - prob_11_plus)
        return 1.0 - producto

    def _calculate_prob_11_or_more(self, quiniela_results: List[str], num_simulations: int = 2000) -> float:
        """
        Calcula Pr(≥11) para una quiniela usando simulación Monte Carlo.
        """
        prob_acierto = np.array([
            self.match_data.iloc[i][f"prob_{res.lower()}"] for i, res in enumerate(quiniela_results)
        ])
        
        simulaciones = np.random.rand(num_simulations, self.num_matches)
        aciertos_matrix = simulaciones < prob_acierto
        aciertos_por_simulacion = np.sum(aciertos_matrix, axis=1)
        
        aciertos_11_plus = np.sum(aciertos_por_simulacion >= 11)
        
        return aciertos_11_plus / num_simulations

    def _generate_neighbor(self, portfolio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera una solución vecina haciendo un pequeño cambio válido en el portafolio.
        """
        neighbor = copy.deepcopy(portfolio)
        
        for _ in range(10): # Intentar hasta 10 veces encontrar un swap válido
            q_idx = random.randrange(self.num_quinielas)
            m_idx = random.randrange(self.num_matches)

            # No modificar partidos Ancla
            if self.match_data.iloc[m_idx]['clasificacion'] == 'Ancla':
                continue

            quiniela_a_modificar = neighbor[q_idx]
            resultado_actual = quiniela_a_modificar['resultados'][m_idx]
            
            opciones = ['L', 'E', 'V']
            opciones.remove(resultado_actual)
            nuevo_resultado = random.choice(opciones)
            
            # Aplicar el cambio
            quiniela_a_modificar['resultados'][m_idx] = nuevo_resultado
            
            # Verificar si el cambio mantiene la validez de la quiniela (empates)
            empates = quiniela_a_modificar['resultados'].count('E')
            if self.rules['empates_min'] <= empates <= self.rules['empates_max']:
                # Actualizar estadísticas y retornar vecino válido
                quiniela_a_modificar['empates'] = empates
                quiniela_a_modificar['distribucion'] = {
                    "L": quiniela_a_modificar['resultados'].count("L"),
                    "E": empates,
                    "V": quiniela_a_modificar['resultados'].count("V")
                }
                return neighbor

        return None # No se pudo encontrar un vecino válido