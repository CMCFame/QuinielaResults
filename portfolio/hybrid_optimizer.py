# progol_optimizer/portfolio/hybrid_optimizer.py
"""
Optimizador Híbrido MEJORADO con cobertura combinatoria y fallbacks automáticos
Implementa CP-SAT con restricciones de cobertura + GRASP-Annealing robusto
"""

import logging
import time
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import math
import random
import copy

# Importación condicional de OR-Tools con fallback
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from config.constants import PROGOL_CONFIG
from logging_setup import get_instrumentor


class EnhancedHybridOptimizer:
    """
    Optimizador Híbrido MEJORADO que garantiza portafolios válidos
    
    Implementa:
    1. CP-SAT con restricciones de cobertura combinatoria
    2. GRASP adaptativo con múltiples reinicios
    3. Simulated Annealing con cooling schedule dinámico
    4. Fallbacks automáticos robustos
    5. Instrumentación completa para debug
    """
    
    def __init__(self, match_data: pd.DataFrame):
        self.logger = logging.getLogger(__name__)
        self.instrumentor = get_instrumentor()
        
        # Validar datos de entrada
        self._validate_input_data(match_data)
        
        self.match_data = match_data
        self.config = PROGOL_CONFIG.get("HYBRID_OPTIMIZER", {})
        self.architecture = PROGOL_CONFIG.get("ARQUITECTURA_PORTAFOLIO", {})
        self.rules = {
            "empates_min": PROGOL_CONFIG.get("EMPATES_MIN", 4),
            "empates_max": PROGOL_CONFIG.get("EMPATES_MAX", 6),
            "rangos_historicos": PROGOL_CONFIG.get("RANGOS_HISTORICOS"),
            "concentracion_max": PROGOL_CONFIG.get("CONCENTRACION_MAX_GENERAL", 0.70),
            "concentracion_inicial": PROGOL_CONFIG.get("CONCENTRACION_MAX_INICIAL", 0.60)
        }
        
        self.num_matches = len(match_data)
        self.num_quinielas = self.architecture.get("num_total", 30)
        self.num_cores = self.architecture.get("num_core", 4)
        
        # Configuración optimizada para velocidad y robustez
        self.max_ip_time = 45  # Tiempo generoso para CP-SAT
        self.max_annealing_iterations = 10000
        self.adaptive_restarts = 3
        
        # Mapeos de resultados
        self.resultados_map = {"L": 0, "E": 1, "V": 2}
        self.resultados_rev_map = {0: "L", 1: "E", 2: "V"}
        self.prob_columns = {"L": "prob_local", "E": "prob_empate", "V": "prob_visitante"}
        
        # Identificar partidos especiales
        self.anchor_matches = self._identify_anchor_matches()
        self.divisor_matches = self._identify_divisor_matches()
        
        self.logger.info(f"🔧 EnhancedHybridOptimizer inicializado: {self.num_matches} partidos, {self.num_quinielas} quinielas")
        self.logger.info(f"⚓ Anclas identificadas: {len(self.anchor_matches)}, Divisores: {len(self.divisor_matches)}")
    
    def generate_portfolio(self) -> List[Dict[str, Any]]:
        """
        Punto de entrada principal con múltiples estrategias de fallback
        """
        session_timer = self.instrumentor.start_timer("generate_portfolio_complete")
        
        self.logger.info("🚀 Iniciando generación de portafolio con estrategia híbrida MEJORADA")
        
        strategies = [
            ("Enhanced CP-SAT", self._solve_enhanced_cpsat_model),
            ("Adaptive GRASP", self._solve_adaptive_grasp),
            ("Robust Heuristic", self._solve_robust_heuristic),
            ("Emergency Fallback", self._solve_emergency_fallback)
        ]
        
        for strategy_name, strategy_func in strategies:
            strategy_timer = self.instrumentor.start_timer(f"strategy_{strategy_name.replace(' ', '_').lower()}")
            
            try:
                self.logger.info(f"🎯 Intentando estrategia: {strategy_name}")
                portfolio = strategy_func()
                
                if portfolio and len(portfolio) == self.num_quinielas:
                    # Validar portafolio generado
                    validation_result = self._validate_portfolio_fast(portfolio)
                    
                    if validation_result["valid"]:
                        self.instrumentor.end_timer(strategy_timer, success=True, metrics={
                            "strategy": strategy_name,
                            "portfolio_size": len(portfolio),
                            "validation_score": validation_result["score"]
                        })
                        
                        self.instrumentor.end_timer(session_timer, success=True, metrics={
                            "successful_strategy": strategy_name,
                            "final_portfolio_size": len(portfolio)
                        })
                        
                        self.logger.info(f"✅ {strategy_name} exitosa! Portafolio válido generado")
                        return portfolio
                    else:
                        self.logger.warning(f"⚠️ {strategy_name} generó portafolio inválido: {validation_result['issues']}")
                else:
                    self.logger.warning(f"⚠️ {strategy_name} falló en generar portafolio completo")
                
                self.instrumentor.end_timer(strategy_timer, success=False)
                
            except Exception as e:
                self.instrumentor.end_timer(strategy_timer, success=False)
                self.logger.error(f"❌ {strategy_name} falló con excepción: {e}")
                if "CP-SAT" in strategy_name:
                    self.logger.info("ℹ️ OR-Tools puede no estar instalado, continuando con métodos alternativos")
        
        # Si llegamos aquí, todas las estrategias fallaron
        self.instrumentor.end_timer(session_timer, success=False)
        self.logger.error("❌ TODAS las estrategias de optimización fallaron")
        return []
    
    def _solve_enhanced_cpsat_model(self) -> Optional[List[Dict[str, Any]]]:
        """
        Modelo CP-SAT MEJORADO con restricciones de cobertura combinatoria
        """
        if not ORTOOLS_AVAILABLE:
            self.logger.warning("📦 OR-Tools no disponible, saltando CP-SAT")
            return None
        
        timer_id = self.instrumentor.start_timer("cp_sat_enhanced")
        
        try:
            self.logger.info("🏗️ Construyendo modelo CP-SAT mejorado con cobertura combinatoria...")
            
            model = cp_model.CpModel()
            
            # Variables: x[q,m,r] = 1 si quiniela q tiene resultado r en partido m
            x = {}
            for q in range(self.num_quinielas):
                for m in range(self.num_matches):
                    for r in range(3):  # L, E, V
                        x[q, m, r] = model.NewBoolVar(f'x_{q}_{m}_{r}')
            
            # RESTRICCIÓN 1: Exactamente un resultado por partido por quiniela
            for q in range(self.num_quinielas):
                for m in range(self.num_matches):
                    model.AddExactlyOne([x[q, m, r] for r in range(3)])
            
            # RESTRICCIÓN 2: Empates por quiniela (4-6)
            for q in range(self.num_quinielas):
                empates_vars = [x[q, m, 1] for m in range(self.num_matches)]  # 1 = E
                model.Add(sum(empates_vars) >= self.rules['empates_min'])
                model.Add(sum(empates_vars) <= self.rules['empates_max'])
            
            # RESTRICCIÓN 3: Concentración máxima por quiniela
            for q in range(self.num_quinielas):
                for r in range(3):
                    resultado_vars = [x[q, m, r] for m in range(self.num_matches)]
                    model.Add(sum(resultado_vars) <= int(self.num_matches * self.rules['concentracion_max']))
            
            # RESTRICCIÓN 4: Concentración inicial (primeros 3 partidos)
            for q in range(self.num_quinielas):
                for r in range(3):
                    primeros_3 = [x[q, m, r] for m in range(min(3, self.num_matches))]
                    model.Add(sum(primeros_3) <= int(3 * self.rules['concentracion_inicial']))
            
            # RESTRICCIÓN 5: Anclas idénticas en todas las quinielas
            for anchor_idx in self.anchor_matches:
                # Encontrar resultado más probable para esta ancla
                probs = [
                    self.match_data.iloc[anchor_idx]['prob_local'],
                    self.match_data.iloc[anchor_idx]['prob_empate'],
                    self.match_data.iloc[anchor_idx]['prob_visitante']
                ]
                best_result = np.argmax(probs)
                
                # Forzar este resultado en todas las quinielas
                for q in range(self.num_quinielas):
                    model.Add(x[q, anchor_idx, best_result] == 1)
            
            # RESTRICCIÓN 6: Cobertura combinatoria para diversidad
            self._add_coverage_constraints(model, x)
            
            # RESTRICCIÓN 7: Distribución global aproximada
            self._add_global_distribution_constraints(model, x)
            
            # OBJETIVO: Maximizar valor esperado total
            objective_terms = []
            for q in range(self.num_quinielas):
                for m in range(self.num_matches):
                    probs = [
                        self.match_data.iloc[m]['prob_local'],
                        self.match_data.iloc[m]['prob_empate'],
                        self.match_data.iloc[m]['prob_visitante']
                    ]
                    for r in range(3):
                        # Escalar por 1000 para evitar decimales
                        objective_terms.append(x[q, m, r] * int(probs[r] * 1000))
            
            model.Maximize(sum(objective_terms))
            
            # Solver con configuración optimizada
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = self.max_ip_time
            solver.parameters.num_search_workers = 1
            solver.parameters.log_search_progress = False
            
            self.logger.info(f"🔍 Resolviendo CP-SAT (timeout: {self.max_ip_time}s)...")
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                solve_time = solver.WallTime()
                self.logger.info(f"✅ CP-SAT encontró solución en {solve_time:.1f}s")
                
                portfolio = self._extract_solution_from_cpsat(solver, x)
                
                self.instrumentor.end_timer(timer_id, success=True, metrics={
                    "solve_time": solve_time,
                    "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                    "objective_value": solver.ObjectiveValue()
                })
                
                return portfolio
            else:
                self.logger.warning(f"⚠️ CP-SAT no encontró solución (status: {status})")
                self.instrumentor.end_timer(timer_id, success=False)
                return None
                
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"❌ Error en CP-SAT mejorado: {e}")
            return None
    
    def _add_coverage_constraints(self, model, x):
        """
        Añade restricciones de cobertura combinatoria para máxima diversidad
        """
        # Asegurar que cada resultado aparezca balanceadamente en cada posición
        for m in range(self.num_matches):
            if m in self.anchor_matches:
                continue  # Skip anclas, ya están fijas
            
            for r in range(3):
                # Cada resultado debe aparecer entre 20-67% de las veces
                min_appearances = int(self.num_quinielas * 0.20)
                max_appearances = int(self.num_quinielas * 0.67)
                
                position_vars = [x[q, m, r] for q in range(self.num_quinielas)]
                model.Add(sum(position_vars) >= min_appearances)
                model.Add(sum(position_vars) <= max_appearances)
    
    def _add_global_distribution_constraints(self, model, x):
        """
        Añade restricciones de distribución global suave
        """
        # Calcular totales esperados para toda la matriz 30x14
        total_cells = self.num_quinielas * self.num_matches
        target_L = int(total_cells * 0.38)
        target_E = int(total_cells * 0.29)
        target_V = int(total_cells * 0.33)
        
        # Variables para totales globales
        total_L_vars = []
        total_E_vars = []
        total_V_vars = []
        
        for q in range(self.num_quinielas):
            for m in range(self.num_matches):
                total_L_vars.append(x[q, m, 0])  # L = 0
                total_E_vars.append(x[q, m, 1])  # E = 1
                total_V_vars.append(x[q, m, 2])  # V = 2
        
        # Restricciones suaves (con tolerancia)
        tolerance = int(total_cells * 0.05)  # 5% de tolerancia
        
        model.Add(sum(total_L_vars) >= target_L - tolerance)
        model.Add(sum(total_L_vars) <= target_L + tolerance)
        model.Add(sum(total_E_vars) >= target_E - tolerance)
        model.Add(sum(total_E_vars) <= target_E + tolerance)
        model.Add(sum(total_V_vars) >= target_V - tolerance)
        model.Add(sum(total_V_vars) <= target_V + tolerance)
    
    def _extract_solution_from_cpsat(self, solver, x) -> List[Dict[str, Any]]:
        """
        Extrae la solución del solver CP-SAT y la convierte al formato requerido
        """
        portfolio = []
        
        for q in range(self.num_quinielas):
            quiniela_results = []
            for m in range(self.num_matches):
                for r in range(3):
                    if solver.BooleanValue(x[q, m, r]):
                        quiniela_results.append(self.resultados_rev_map[r])
                        break
            
            # Determinar tipo y metadata según arquitectura
            if q < self.num_cores:
                quiniela_type = "Core"
                quiniela_id = f"Core-{q+1}"
                par_id = None
            else:
                quiniela_type = "Satelite"
                sat_index = q - self.num_cores
                par_id = sat_index // 2
                sub_id = "A" if sat_index % 2 == 0 else "B"
                quiniela_id = f"Sat-{par_id+1}{sub_id}"
            
            empates = quiniela_results.count("E")
            portfolio.append({
                "id": quiniela_id,
                "tipo": quiniela_type,
                "par_id": par_id,
                "resultados": quiniela_results,
                "empates": empates,
                "distribución": {
                    "L": quiniela_results.count("L"),
                    "E": empates,
                    "V": quiniela_results.count("V")
                }
            })
        
        return portfolio
    
    def _solve_adaptive_grasp(self) -> Optional[List[Dict[str, Any]]]:
        """
        GRASP adaptativo con múltiples reinicios y cooling schedule dinámico
        """
        timer_id = self.instrumentor.start_timer("adaptive_grasp")
        
        try:
            self.logger.info("🎯 Ejecutando GRASP adaptativo con múltiples reinicios...")
            
            best_portfolio = None
            best_score = -float('inf')
            
            for restart in range(self.adaptive_restarts):
                restart_timer = self.instrumentor.start_timer(f"grasp_restart_{restart}")
                
                self.logger.info(f"🔄 GRASP reinicio {restart + 1}/{self.adaptive_restarts}")
                
                # Fase construcción con randomización adaptativa
                alpha = 0.15 + 0.1 * restart  # Aumentar randomización en reinicios
                initial_portfolio = self._grasp_construction(alpha)
                
                if not initial_portfolio:
                    self.instrumentor.end_timer(restart_timer, success=False)
                    continue
                
                # Fase mejoramiento con annealing adaptativo
                improved_portfolio = self._adaptive_annealing(
                    initial_portfolio, 
                    temperature=0.5 + 0.3 * restart,
                    cooling_factor=0.95 - 0.02 * restart
                )
                
                if improved_portfolio:
                    score = self._calculate_portfolio_score_detailed(improved_portfolio)
                    
                    if score > best_score:
                        best_portfolio = improved_portfolio
                        best_score = score
                        self.logger.info(f"🎉 Nuevo mejor score en reinicio {restart + 1}: {score:.6f}")
                
                self.instrumentor.end_timer(restart_timer, success=True, metrics={
                    "restart_number": restart,
                    "alpha_used": alpha,
                    "portfolio_generated": improved_portfolio is not None
                })
            
            self.instrumentor.end_timer(timer_id, success=best_portfolio is not None, metrics={
                "best_score": best_score,
                "successful_restarts": sum(1 for _ in range(self.adaptive_restarts) if best_portfolio)
            })
            
            return best_portfolio
            
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"❌ Error en GRASP adaptativo: {e}")
            return None
    
    def _grasp_construction(self, alpha: float) -> Optional[List[Dict[str, Any]]]:
        """
        Construcción GRASP con lista de candidatos restringida
        """
        portfolio = []
        
        try:
            # Generar 4 quinielas Core primero
            for core_idx in range(self.num_cores):
                core_quiniela = self._build_core_quiniela(core_idx, alpha)
                if core_quiniela:
                    portfolio.append(core_quiniela)
            
            # Generar 26 satélites en 13 pares
            num_pairs = (self.num_quinielas - self.num_cores) // 2
            
            for pair_idx in range(num_pairs):
                pair_a, pair_b = self._build_satellite_pair(pair_idx, alpha)
                if pair_a and pair_b:
                    portfolio.extend([pair_a, pair_b])
            
            return portfolio if len(portfolio) == self.num_quinielas else None
            
        except Exception as e:
            self.logger.error(f"Error en construcción GRASP: {e}")
            return None
    
    def _build_core_quiniela(self, core_idx: int, alpha: float) -> Optional[Dict[str, Any]]:
        """
        Construye una quiniela Core usando heurística GRASP
        """
        resultados = [""] * self.num_matches
        
        # Fijar anclas
        for anchor_idx in self.anchor_matches:
            probs = [
                self.match_data.iloc[anchor_idx]['prob_local'],
                self.match_data.iloc[anchor_idx]['prob_empate'],
                self.match_data.iloc[anchor_idx]['prob_visitante']
            ]
            best_result = self.resultados_rev_map[np.argmax(probs)]
            resultados[anchor_idx] = best_result
        
        # Completar resto de partidos con GRASP
        for m in range(self.num_matches):
            if m in self.anchor_matches:
                continue
            
            # Calcular lista de candidatos con probabilidades
            candidates = []
            probs = [
                self.match_data.iloc[m]['prob_local'],
                self.match_data.iloc[m]['prob_empate'],
                self.match_data.iloc[m]['prob_visitante']
            ]
            
            for r_idx, prob in enumerate(probs):
                candidates.append((self.resultados_rev_map[r_idx], prob))
            
            # Ordenar por probabilidad y aplicar restricción alpha
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_prob = candidates[0][1]
            threshold = best_prob - alpha * (best_prob - candidates[-1][1])
            
            # Filtrar candidatos por threshold
            restricted_candidates = [c for c in candidates if c[1] >= threshold]
            
            # Selección aleatoria ponderada
            weights = [c[1] for c in restricted_candidates]
            selected = np.random.choice(
                [c[0] for c in restricted_candidates],
                p=np.array(weights) / sum(weights)
            )
            
            resultados[m] = selected
        
        # Ajustar empates si es necesario
        resultados = self._adjust_empates_grasp(resultados)
        
        empates = resultados.count("E")
        return {
            "id": f"Core-{core_idx + 1}",
            "tipo": "Core",
            "par_id": None,
            "resultados": resultados,
            "empates": empates,
            "distribución": {
                "L": resultados.count("L"),
                "E": empates,
                "V": resultados.count("V")
            }
        }
    
    def _build_satellite_pair(self, pair_idx: int, alpha: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Construye un par de satélites anticorrelados
        """
        # Construir satélite A similar a Core pero con variación
        satelite_a = self._build_core_quiniela(0, alpha * 1.5)  # Más randomización
        
        if not satelite_a:
            return None, None
        
        # Construir satélite B invirtiendo divisores
        resultados_b = satelite_a["resultados"].copy()
        
        for divisor_idx in self.divisor_matches:
            current_result = resultados_b[divisor_idx]
            # Invertir a resultado menos probable para crear anticorrelación
            probs = [
                self.match_data.iloc[divisor_idx]['prob_local'],
                self.match_data.iloc[divisor_idx]['prob_empate'],
                self.match_data.iloc[divisor_idx]['prob_visitante']
            ]
            
            # Escoger segundo más probable (no el menos probable para mantener algo de sentido)
            sorted_indices = np.argsort(probs)[::-1]
            alternative_result = self.resultados_rev_map[sorted_indices[1]]
            resultados_b[divisor_idx] = alternative_result
        
        # Ajustar empates en B
        resultados_b = self._adjust_empates_grasp(resultados_b)
        
        # Crear objetos satélite
        empates_a = satelite_a["resultados"].count("E")
        empates_b = resultados_b.count("E")
        
        satelite_a.update({
            "id": f"Sat-{pair_idx + 1}A",
            "tipo": "Satelite",
            "par_id": pair_idx
        })
        
        satelite_b = {
            "id": f"Sat-{pair_idx + 1}B",
            "tipo": "Satelite",
            "par_id": pair_idx,
            "resultados": resultados_b,
            "empates": empates_b,
            "distribución": {
                "L": resultados_b.count("L"),
                "E": empates_b,
                "V": resultados_b.count("V")
            }
        }
        
        return satelite_a, satelite_b
    
    def _adjust_empates_grasp(self, resultados: List[str]) -> List[str]:
        """
        Ajusta empates para cumplir restricción 4-6 empates
        """
        empates_actuales = resultados.count("E")
        
        if empates_actuales < self.rules['empates_min']:
            # Convertir algunos L/V a E, priorizando partidos no-ancla
            needed = self.rules['empates_min'] - empates_actuales
            candidates = [i for i in range(self.num_matches) 
                         if i not in self.anchor_matches and resultados[i] != "E"]
            
            for i in range(min(needed, len(candidates))):
                resultados[candidates[i]] = "E"
                
        elif empates_actuales > self.rules['empates_max']:
            # Convertir algunos E a L/V
            excess = empates_actuales - self.rules['empates_max']
            candidates = [i for i in range(self.num_matches) 
                         if i not in self.anchor_matches and resultados[i] == "E"]
            
            for i in range(min(excess, len(candidates))):
                # Elegir L o V basado en probabilidades
                probs = [
                    self.match_data.iloc[candidates[i]]['prob_local'],
                    self.match_data.iloc[candidates[i]]['prob_visitante']
                ]
                resultados[candidates[i]] = "L" if probs[0] > probs[1] else "V"
        
        return resultados
    
    def _adaptive_annealing(self, initial_portfolio: List[Dict[str, Any]], 
                           temperature: float, cooling_factor: float) -> Optional[List[Dict[str, Any]]]:
        """
        Simulated Annealing adaptativo con cooling schedule dinámico
        """
        current_portfolio = copy.deepcopy(initial_portfolio)
        best_portfolio = copy.deepcopy(initial_portfolio)
        
        current_score = self._calculate_portfolio_score_detailed(current_portfolio)
        best_score = current_score
        
        temp = temperature
        iterations_without_improvement = 0
        max_iterations_without_improvement = 1000
        
        for iteration in range(self.max_annealing_iterations):
            # Generar vecino
            neighbor = self._generate_neighbor_annealing(current_portfolio)
            
            if not neighbor:
                continue
            
            neighbor_score = self._calculate_portfolio_score_detailed(neighbor)
            delta = neighbor_score - current_score
            
            # Decisión de aceptación
            accept = delta > 0 or (temp > 0.001 and random.random() < math.exp(delta / temp))
            
            if accept:
                current_portfolio = neighbor
                current_score = neighbor_score
                iterations_without_improvement = 0
                
                if neighbor_score > best_score:
                    best_portfolio = copy.deepcopy(neighbor)
                    best_score = neighbor_score
            else:
                iterations_without_improvement += 1
            
            # Enfriamiento adaptativo
            if iterations_without_improvement > 100:
                cooling_factor = min(0.99, cooling_factor + 0.01)  # Enfriar más lento
            
            temp *= cooling_factor
            
            # Log periódico
            if iteration % 1000 == 0:
                self.instrumentor.log_optimization_iteration(
                    iteration, best_score, temp, accept, 
                    self._calculate_portfolio_hash(best_portfolio)
                )
            
            # Parada temprana
            if iterations_without_improvement > max_iterations_without_improvement:
                self.logger.debug(f"Annealing parada temprana en iteración {iteration}")
                break
        
        return best_portfolio
    
    def _generate_neighbor_annealing(self, portfolio: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Genera vecino para Simulated Annealing con movimientos válidos
        """
        neighbor = copy.deepcopy(portfolio)
        
        # Seleccionar quiniela aleatoria (preferir satélites)
        satellite_indices = [i for i, q in enumerate(portfolio) if q["tipo"] == "Satelite"]
        if satellite_indices and random.random() < 0.8:
            quiniela_idx = random.choice(satellite_indices)
        else:
            quiniela_idx = random.randrange(len(portfolio))
        
        # Seleccionar partido modificable (no ancla)
        modifiable_matches = [i for i in range(self.num_matches) if i not in self.anchor_matches]
        if not modifiable_matches:
            return None
        
        match_idx = random.choice(modifiable_matches)
        
        # Generar nuevo resultado
        current_result = neighbor[quiniela_idx]["resultados"][match_idx]
        possible_results = ["L", "E", "V"]
        possible_results.remove(current_result)
        new_result = random.choice(possible_results)
        
        # Aplicar cambio
        neighbor[quiniela_idx]["resultados"][match_idx] = new_result
        
        # Verificar que sigue siendo válida la restricción de empates
        empates = neighbor[quiniela_idx]["resultados"].count("E")
        if not (self.rules['empates_min'] <= empates <= self.rules['empates_max']):
            return None
        
        # Actualizar metadata
        neighbor[quiniela_idx]["empates"] = empates
        neighbor[quiniela_idx]["distribución"] = {
            "L": neighbor[quiniela_idx]["resultados"].count("L"),
            "E": empates,
            "V": neighbor[quiniela_idx]["resultados"].count("V")
        }
        
        return neighbor
    
    def _solve_robust_heuristic(self) -> Optional[List[Dict[str, Any]]]:
        """
        Heurística robusta basada en probabilidades y reglas determinísticas
        """
        timer_id = self.instrumentor.start_timer("robust_heuristic")
        
        try:
            self.logger.info("🛡️ Ejecutando heurística robusta determinística...")
            
            portfolio = []
            
            # Generar 4 Cores con lógica determinística
            for core_idx in range(self.num_cores):
                core = self._build_deterministic_core(core_idx)
                if core:
                    portfolio.append(core)
            
            # Generar 26 satélites en pares
            for pair_idx in range(13):
                sat_a, sat_b = self._build_deterministic_satellite_pair(pair_idx)
                if sat_a and sat_b:
                    portfolio.extend([sat_a, sat_b])
            
            self.instrumentor.end_timer(timer_id, success=len(portfolio) == self.num_quinielas)
            
            return portfolio if len(portfolio) == self.num_quinielas else None
            
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"❌ Error en heurística robusta: {e}")
            return None
    
    def _build_deterministic_core(self, core_idx: int) -> Dict[str, Any]:
        """
        Construye Core de manera determinística basada en probabilidades
        """
        resultados = []
        
        for m in range(self.num_matches):
            probs = [
                self.match_data.iloc[m]['prob_local'],
                self.match_data.iloc[m]['prob_empate'],
                self.match_data.iloc[m]['prob_visitante']
            ]
            
            # Para Cores, siempre tomar el más probable
            best_result_idx = np.argmax(probs)
            resultados.append(self.resultados_rev_map[best_result_idx])
        
        # Ajustar empates con patrón determinístico
        resultados = self._force_empates_deterministic(resultados, core_idx)
        
        empates = resultados.count("E")
        return {
            "id": f"Core-{core_idx + 1}",
            "tipo": "Core",
            "par_id": None,
            "resultados": resultados,
            "empates": empates,
            "distribución": {
                "L": resultados.count("L"),
                "E": empates,
                "V": resultados.count("V")
            }
        }
    
    def _build_deterministic_satellite_pair(self, pair_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Construye par de satélites de manera determinística
        """
        # Satelite A: similar a Core pero con rotación
        resultados_a = []
        for m in range(self.num_matches):
            probs = [
                self.match_data.iloc[m]['prob_local'],
                self.match_data.iloc[m]['prob_empate'],
                self.match_data.iloc[m]['prob_visitante']
            ]
            
            # Aplicar rotación basada en pair_idx
            if m in self.divisor_matches and (m + pair_idx) % 3 == 0:
                # Tomar segundo más probable
                sorted_indices = np.argsort(probs)[::-1]
                result_idx = sorted_indices[1]
            else:
                # Tomar más probable
                result_idx = np.argmax(probs)
            
            resultados_a.append(self.resultados_rev_map[result_idx])
        
        # Satelite B: invertir algunos divisores para crear anticorrelación
        resultados_b = resultados_a.copy()
        for divisor_idx in self.divisor_matches[::2]:  # Cada segundo divisor
            probs = [
                self.match_data.iloc[divisor_idx]['prob_local'],
                self.match_data.iloc[divisor_idx]['prob_empate'],
                self.match_data.iloc[divisor_idx]['prob_visitante']
            ]
            
            # Tomar tercero más probable para máxima anticorrelación
            sorted_indices = np.argsort(probs)[::-1]
            alternative_idx = sorted_indices[2] if len(sorted_indices) > 2 else sorted_indices[1]
            resultados_b[divisor_idx] = self.resultados_rev_map[alternative_idx]
        
        # Ajustar empates en ambos
        resultados_a = self._force_empates_deterministic(resultados_a, pair_idx)
        resultados_b = self._force_empates_deterministic(resultados_b, pair_idx + 100)
        
        # Crear objetos
        empates_a = resultados_a.count("E")
        empates_b = resultados_b.count("E")
        
        satelite_a = {
            "id": f"Sat-{pair_idx + 1}A",
            "tipo": "Satelite",
            "par_id": pair_idx,
            "resultados": resultados_a,
            "empates": empates_a,
            "distribución": {
                "L": resultados_a.count("L"),
                "E": empates_a,
                "V": resultados_a.count("V")
            }
        }
        
        satelite_b = {
            "id": f"Sat-{pair_idx + 1}B",
            "tipo": "Satelite",
            "par_id": pair_idx,
            "resultados": resultados_b,
            "empates": empates_b,
            "distribución": {
                "L": resultados_b.count("L"),
                "E": empates_b,
                "V": resultados_b.count("V")
            }
        }
        
        return satelite_a, satelite_b
    
    def _force_empates_deterministic(self, resultados: List[str], seed: int) -> List[str]:
        """
        Fuerza empates de manera determinística usando seed
        """
        empates_actuales = resultados.count("E")
        target_empates = 4 + (seed % 3)  # Entre 4-6 basado en seed
        
        if empates_actuales == target_empates:
            return resultados
        
        # Identificar candidatos modificables
        modifiable = [i for i in range(self.num_matches) if i not in self.anchor_matches]
        
        if empates_actuales < target_empates:
            # Convertir L/V a E
            needed = target_empates - empates_actuales
            candidates = [i for i in modifiable if resultados[i] != "E"]
            
            # Usar seed para selección determinística
            np.random.seed(seed)
            selected = np.random.choice(candidates, min(needed, len(candidates)), replace=False)
            
            for i in selected:
                resultados[i] = "E"
        
        elif empates_actuales > target_empates:
            # Convertir E a L/V
            excess = empates_actuales - target_empates
            candidates = [i for i in modifiable if resultados[i] == "E"]
            
            np.random.seed(seed)
            selected = np.random.choice(candidates, min(excess, len(candidates)), replace=False)
            
            for i in selected:
                # Elegir L o V basado en probabilidades + seed
                probs = [
                    self.match_data.iloc[i]['prob_local'],
                    self.match_data.iloc[i]['prob_visitante']
                ]
                resultados[i] = "L" if (probs[0] > probs[1]) ^ (seed % 2 == 0) else "V"
        
        return resultados
    
    def _solve_emergency_fallback(self) -> List[Dict[str, Any]]:
        """
        Fallback de emergencia que SIEMPRE produce un portafolio válido
        """
        timer_id = self.instrumentor.start_timer("emergency_fallback")
        
        self.logger.warning("🚨 Activando fallback de emergencia - generación ultra-simple")
        
        try:
            portfolio = []
            
            # Patrón base ultra-simple pero válido
            base_pattern = ["L", "L", "E", "V", "L", "E", "E", "V", "L", "E", "V", "V", "L", "E"]
            
            for q in range(self.num_quinielas):
                # Rotar patrón base para crear variedad
                rotated_pattern = base_pattern[q % len(base_pattern):] + base_pattern[:q % len(base_pattern)]
                
                # Asegurar que tiene exactamente 14 elementos
                if len(rotated_pattern) != 14:
                    rotated_pattern = (rotated_pattern * 14)[:14]
                
                # Forzar anclas si existen
                for anchor_idx in self.anchor_matches:
                    if anchor_idx < len(rotated_pattern):
                        probs = [
                            self.match_data.iloc[anchor_idx]['prob_local'],
                            self.match_data.iloc[anchor_idx]['prob_empate'],
                            self.match_data.iloc[anchor_idx]['prob_visitante']
                        ]
                        rotated_pattern[anchor_idx] = self.resultados_rev_map[np.argmax(probs)]
                
                # Asegurar 4-6 empates
                empates = rotated_pattern.count("E")
                while empates < 4:
                    # Convertir algún L/V a E
                    for i in range(14):
                        if i not in self.anchor_matches and rotated_pattern[i] in ["L", "V"]:
                            rotated_pattern[i] = "E"
                            empates += 1
                            break
                
                while empates > 6:
                    # Convertir algún E a L
                    for i in range(14):
                        if i not in self.anchor_matches and rotated_pattern[i] == "E":
                            rotated_pattern[i] = "L"
                            empates -= 1
                            break
                
                # Determinar tipo
                if q < 4:
                    quiniela_type = "Core"
                    quiniela_id = f"Core-{q + 1}"
                    par_id = None
                else:
                    quiniela_type = "Satelite"
                    sat_index = q - 4
                    par_id = sat_index // 2
                    sub_id = "A" if sat_index % 2 == 0 else "B"
                    quiniela_id = f"Sat-{par_id + 1}{sub_id}"
                
                empates_final = rotated_pattern.count("E")
                portfolio.append({
                    "id": quiniela_id,
                    "tipo": quiniela_type,
                    "par_id": par_id,
                    "resultados": rotated_pattern.copy(),
                    "empates": empates_final,
                    "distribución": {
                        "L": rotated_pattern.count("L"),
                        "E": empates_final,
                        "V": rotated_pattern.count("V")
                    }
                })
            
            self.instrumentor.end_timer(timer_id, success=True, metrics={
                "fallback_portfolio_size": len(portfolio)
            })
            
            self.logger.info(f"✅ Fallback de emergencia generó {len(portfolio)} quinielas")
            return portfolio
            
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"❌ Incluso el fallback de emergencia falló: {e}")
            return []
    
    def _calculate_portfolio_score_detailed(self, portfolio: List[Dict[str, Any]]) -> float:
        """
        Calcula score detallado del portafolio considerando múltiples factores
        """
        try:
            # Componentes del score
            prob_score = 0.0  # Probabilidad promedio de aciertos
            diversity_score = 0.0  # Diversidad entre quinielas
            balance_score = 0.0  # Balance de distribución
            
            # 1. Score de probabilidades
            for quiniela in portfolio:
                quiniela_prob = 1.0
                for i, resultado in enumerate(quiniela["resultados"]):
                    prob_col = self.prob_columns[resultado]
                    prob = self.match_data.iloc[i][prob_col]
                    quiniela_prob *= prob
                
                prob_score += quiniela_prob
            
            prob_score /= len(portfolio)  # Promedio
            
            # 2. Score de diversidad (anticorrelación entre pares)
            if len(portfolio) > 1:
                correlations = []
                for i in range(len(portfolio)):
                    for j in range(i + 1, len(portfolio)):
                        corr = self._calculate_jaccard_similarity(
                            portfolio[i]["resultados"],
                            portfolio[j]["resultados"]
                        )
                        correlations.append(1.0 - corr)  # Anticorrelación
                
                diversity_score = np.mean(correlations) if correlations else 0
            
            # 3. Score de balance de distribución
            total_L = sum(q["distribución"]["L"] for q in portfolio)
            total_E = sum(q["distribución"]["E"] for q in portfolio)
            total_V = sum(q["distribución"]["V"] for q in portfolio)
            total = total_L + total_E + total_V
            
            if total > 0:
                actual_dist = [total_L / total, total_E / total, total_V / total]
                target_dist = [0.38, 0.29, 0.33]
                
                # Calcular distancia euclidiana invertida
                distance = np.sqrt(sum((a - t) ** 2 for a, t in zip(actual_dist, target_dist)))
                balance_score = max(0, 1.0 - distance)
            
            # Score final combinado
            final_score = (0.5 * prob_score + 0.3 * diversity_score + 0.2 * balance_score)
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error calculando score de portafolio: {e}")
            return 0.0
    
    def _calculate_jaccard_similarity(self, quiniela_a: List[str], quiniela_b: List[str]) -> float:
        """Calcula similitud de Jaccard entre dos quinielas"""
        if len(quiniela_a) != len(quiniela_b):
            return 0.0
        
        coincidences = sum(1 for a, b in zip(quiniela_a, quiniela_b) if a == b)
        return coincidences / len(quiniela_a)
    
    def _calculate_portfolio_hash(self, portfolio: List[Dict[str, Any]]) -> str:
        """Calcula hash único del portafolio para trazabilidad"""
        portfolio_str = ""
        for q in sorted(portfolio, key=lambda x: x["id"]):
            portfolio_str += "".join(q["resultados"])
        
        return hashlib.md5(portfolio_str.encode()).hexdigest()[:8]
    
    def _validate_portfolio_fast(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validación rápida del portafolio generado
        """
        issues = []
        score = 1.0
        
        try:
            # Validar tamaño
            if len(portfolio) != self.num_quinielas:
                issues.append(f"Portfolio tiene {len(portfolio)} quinielas, esperadas {self.num_quinielas}")
                score *= 0.5
            
            # Validar empates por quiniela
            for quiniela in portfolio:
                empates = quiniela.get("empates", 0)
                if not (self.rules['empates_min'] <= empates <= self.rules['empates_max']):
                    issues.append(f"Quiniela {quiniela['id']}: {empates} empates fuera de rango [{self.rules['empates_min']}-{self.rules['empates_max']}]")
                    score *= 0.9
            
            # Validar concentración
            for quiniela in portfolio:
                distribución = quiniela.get("distribución", {})
                max_conc = max(distribución.values()) / 14 if distribución else 0
                
                if max_conc > self.rules['concentracion_max']:
                    issues.append(f"Quiniela {quiniela['id']}: concentración {max_conc:.1%} > {self.rules['concentracion_max']:.1%}")
                    score *= 0.8
            
            # Validar arquitectura
            cores = [q for q in portfolio if q["tipo"] == "Core"]
            satelites = [q for q in portfolio if q["tipo"] == "Satelite"]
            
            if len(cores) != 4:
                issues.append(f"Esperadas 4 quinielas Core, encontradas {len(cores)}")
                score *= 0.7
            
            if len(satelites) != 26:
                issues.append(f"Esperadas 26 satélites, encontradas {len(satelites)}")
                score *= 0.7
            
            return {
                "valid": len(issues) == 0,
                "score": score,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "valid": False,
                "score": 0.0,
                "issues": [f"Error en validación: {e}"]
            }
    
    def _identify_anchor_matches(self) -> List[int]:
        """Identifica índices de partidos que califican como Ancla"""
        anchors = []
        for i, row in self.match_data.iterrows():
            if row.get('clasificacion') == 'Ancla':
                anchors.append(i)
            else:
                # Verificar si cumple criterios de Ancla por probabilidad
                max_prob = max(row['prob_local'], row['prob_empate'], row['prob_visitante'])
                if max_prob > 0.60:
                    anchors.append(i)
        
        return anchors
    
    def _identify_divisor_matches(self) -> List[int]:
        """Identifica índices de partidos que califican como Divisor"""
        divisors = []
        for i, row in self.match_data.iterrows():
            if row.get('clasificacion') == 'Divisor':
                divisors.append(i)
            else:
                # Verificar si cumple criterios de Divisor por probabilidad
                max_prob = max(row['prob_local'], row['prob_empate'], row['prob_visitante'])
                if 0.40 < max_prob < 0.60:
                    divisors.append(i)
        
        return divisors
    
    def _validate_input_data(self, match_data: pd.DataFrame):
        """Valida que los datos de entrada sean correctos"""
        required_columns = ['prob_local', 'prob_empate', 'prob_visitante']
        missing_columns = [col for col in required_columns if col not in match_data.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        if len(match_data) != 14:
            raise ValueError(f"Se requieren exactamente 14 partidos, se encontraron {len(match_data)}")
        
        self.logger.debug("✅ Validación de datos de entrada exitosa")


# Clase de compatibilidad
class HybridOptimizer(EnhancedHybridOptimizer):
    """Wrapper para mantener compatibilidad con código existente"""
    
    def __init__(self, match_data: pd.DataFrame):
        super().__init__(match_data)
        self.logger.warning("⚠️ Usando HybridOptimizer legacy - migrar a EnhancedHybridOptimizer")