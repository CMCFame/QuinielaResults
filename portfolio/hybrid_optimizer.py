# progol_optimizer/portfolio/hybrid_optimizer.py - VERSIÓN RÁPIDA Y ROBUSTA
"""
Implementa la estrategia de generación de portafolios Híbrida (IP + GRASP-Annealing).
VERSIÓN OPTIMIZADA: Timeouts agresivos, modelo simplificado, fallbacks automáticos
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import math
import random
import copy
import time
from ortools.sat.python import cp_model

from config.constants import PROGOL_CONFIG

# Configuración del logger para este módulo
logging.basicConfig(level=PROGOL_CONFIG.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class HybridOptimizer:
    """
    Gestiona la generación de portafolios de quinielas mediante un enfoque híbrido RÁPIDO.
    """

    def __init__(self, match_data: pd.DataFrame):
        """
        Inicializa el optimizador híbrido.
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
        
        # Mapeo correcto de columnas
        self.prob_columns = {
            "L": "prob_local",
            "E": "prob_empate", 
            "V": "prob_visitante"
        }
        
        # Timeouts más agresivos
        self.max_ip_time = 30  # Máximo 30 segundos para IP
        self.max_annealing_iterations = 5000  # Menos iteraciones para ser más rápido
        
        logger.info("Inicializando HybridOptimizer RÁPIDO con %d partidos y para %d quinielas.",
                    self.num_matches, self.num_quinielas)

    def generate_portfolio(self) -> List[Dict[str, Any]]:
        """
        Punto de entrada principal para generar el portafolio completo.
        VERSIÓN RÁPIDA con timeouts estrictos
        """
        logger.info("Iniciando generación de portafolio RÁPIDA con estrategia Híbrida.")
        start_time = time.time()
        
        try:
            # Validar datos de entrada
            self._validate_input_data()
            
            # --- Fase 1: Generación RÁPIDA con IP simplificado ---
            logger.info("🚀 Fase 1: Generación rápida con IP (timeout: 30s)")
            initial_solution = self._solve_ip_model_fast()
            
            if not initial_solution:
                logger.warning("⚠️ IP rápido falló, usando generación heurística")
                initial_solution = self._generate_heuristic_fallback()
            
            if not initial_solution:
                logger.error("❌ No se pudo generar solución inicial")
                return []

            # --- Fase 2: Mejora RÁPIDA con Annealing reducido ---
            logger.info("🔥 Fase 2: Mejora rápida con Annealing (5000 iter)")
            optimized_solution = self._run_fast_annealing(initial_solution)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Generación RÁPIDA completada en {elapsed:.1f}s")
            return optimized_solution
            
        except Exception as e:
            logger.error(f"❌ Error en generate_portfolio: {e}")
            # Último fallback: generar con método simple
            return self._generate_simple_fallback()

    def _validate_input_data(self):
        """Valida que los datos de entrada tengan el formato correcto."""
        required_columns = ['prob_local', 'prob_empate', 'prob_visitante', 'clasificacion']
        missing_columns = [col for col in required_columns if col not in self.match_data.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        if len(self.match_data) != 14:
            raise ValueError(f"Se requieren exactamente 14 partidos, se encontraron {len(self.match_data)}")
        
        logger.info("✅ Validación de datos de entrada exitosa")

    def _solve_ip_model_fast(self) -> Optional[List[Dict[str, Any]]]:
        """
        Resuelve un modelo IP SIMPLIFICADO y RÁPIDO
        """
        logger.info("🏗️ Construyendo modelo IP simplificado...")
        
        try:
            model = cp_model.CpModel()
            
            # Variables: solo para 30 quinielas x 14 partidos x 3 resultados
            x = {}
            for j in range(self.num_quinielas):
                for i in range(self.num_matches):
                    for r_idx in range(3):  # L, E, V
                        x[j, i, r_idx] = model.NewBoolVar(f'x_{j}_{i}_{r_idx}')

            # Restricción 1: Exactamente un resultado por partido por quiniela
            for j in range(self.num_quinielas):
                for i in range(self.num_matches):
                    model.AddExactlyOne([x[j, i, r_idx] for r_idx in range(3)])

            # Restricción 2: Empates por quiniela (SIMPLIFICADA)
            for j in range(self.num_quinielas):
                empates_vars = [x[j, i, 1] for i in range(self.num_matches)]  # 1 = E
                model.Add(sum(empates_vars) >= self.rules['empates_min'])
                model.Add(sum(empates_vars) <= self.rules['empates_max'])

            # Restricción 3: Anclas solo para cores (SIMPLIFICADA)
            anchor_matches = self.match_data[self.match_data['clasificacion'] == 'Ancla']
            for idx, partido in anchor_matches.iterrows():
                probs = [
                    partido['prob_local'],
                    partido['prob_empate'], 
                    partido['prob_visitante']
                ]
                best_result_idx = np.argmax(probs)
                
                # Solo aplicar a los primeros 4 (cores)
                for j in range(min(4, self.num_quinielas)):
                    model.Add(x[j, idx, best_result_idx] == 1)

            # Objetivo: Maximizar probabilidades (SIMPLIFICADO)
            objective_terms = []
            for j in range(self.num_quinielas):
                for i in range(self.num_matches):
                    probs = [
                        self.match_data.iloc[i]['prob_local'],
                        self.match_data.iloc[i]['prob_empate'],
                        self.match_data.iloc[i]['prob_visitante']
                    ]
                    for r_idx in range(3):
                        # Escalar por 1000 en lugar de 10000 para reducir complejidad
                        objective_terms.append(x[j, i, r_idx] * int(probs[r_idx] * 1000))
            
            model.Maximize(sum(objective_terms))

            # Solver con timeout AGRESIVO
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = self.max_ip_time  # 30 segundos máximo
            solver.parameters.num_search_workers = 1  # Un solo worker para ser más rápido
            
            logger.info(f"🔍 Resolviendo IP (timeout: {self.max_ip_time}s)...")
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                logger.info(f"✅ Solución IP encontrada en {solver.WallTime():.1f}s")
                return self._extract_solution_from_solver(solver, x)
            else:
                logger.warning(f"⚠️ IP no encontró solución en {self.max_ip_time}s (status: {status})")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en IP rápido: {e}")
            return None

    def _extract_solution_from_solver(self, solver, x) -> List[Dict[str, Any]]:
        """Extrae la solución del solver y la convierte al formato requerido"""
        portfolio = []
        
        for j in range(self.num_quinielas):
            quiniela_results = []
            for i in range(self.num_matches):
                for r_idx in range(3):
                    if solver.BooleanValue(x[j, i, r_idx]):
                        quiniela_results.append(self.resultados_rev_map[r_idx])
                        break
            
            # Determinar tipo y metadata
            if j < self.num_cores:
                quiniela_type = "Core"
                quiniela_id = f"Core-{j+1}"
                par_id = None
            else:
                quiniela_type = "Satelite"
                sat_index = j - self.num_cores
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

    def _run_fast_annealing(self, initial_solution: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ejecuta Annealing RÁPIDO con menos iteraciones
        """
        logger.info("🔥 Iniciando Annealing rápido...")
        
        # Parámetros más agresivos para velocidad
        temp = 0.5  # Temperatura inicial más baja
        cooling_factor = 0.98  # Enfriamiento más rápido
        iterations = self.max_annealing_iterations  # Menos iteraciones

        current_portfolio = copy.deepcopy(initial_solution)
        best_portfolio = copy.deepcopy(initial_solution)

        current_score = self._calculate_portfolio_score_fast(current_portfolio)
        best_score = current_score
        
        for i in range(iterations):
            # Generar vecino más simple
            neighbor_portfolio = self._generate_simple_neighbor(current_portfolio)
            
            if not neighbor_portfolio:
                continue

            neighbor_score = self._calculate_portfolio_score_fast(neighbor_portfolio)
            delta = neighbor_score - current_score
            
            # Decisión de aceptación
            if delta > 0 or (temp > 0.001 and math.exp(delta / temp) > random.random()):
                current_portfolio = copy.deepcopy(neighbor_portfolio)
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_portfolio = copy.deepcopy(current_portfolio)
                    best_score = current_score

            # Enfriamiento más agresivo
            temp *= cooling_factor
            
            # Log cada 1000 iteraciones
            if i % 1000 == 0:
                logger.debug(f"Annealing iter {i}/{iterations} - Score: {best_score:.4f}")

        logger.info(f"✅ Annealing completado. Score final: {best_score:.6f}")
        return best_portfolio

    def _calculate_portfolio_score_fast(self, portfolio: List[Dict[str, Any]]) -> float:
        """
        Cálculo RÁPIDO del score del portafolio (menos simulaciones)
        """
        try:
            producto = 1.0
            for quiniela in portfolio:
                prob_11_plus = self._calculate_prob_11_fast(quiniela['resultados'])
                producto *= (1.0 - prob_11_plus)
            return 1.0 - producto
        except:
            return 0.0

    def _calculate_prob_11_fast(self, quiniela_results: List[str]) -> float:
        """
        Cálculo RÁPIDO de Pr(≥11) con menos simulaciones
        """
        try:
            prob_acierto = np.array([
                self.match_data.iloc[i][self.prob_columns[res]] 
                for i, res in enumerate(quiniela_results)
            ])
            
            # Menos simulaciones para velocidad
            num_sims = 1000  # En lugar de 2000
            simulaciones = np.random.rand(num_sims, self.num_matches)
            aciertos = np.sum(simulaciones < prob_acierto, axis=1)
            return np.sum(aciertos >= 11) / num_sims
        except:
            return 0.0

    def _generate_simple_neighbor(self, portfolio: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Genera un vecino SIMPLE (un solo cambio)
        """
        try:
            neighbor = copy.deepcopy(portfolio)
            
            # Solo 3 intentos para ser más rápido
            for _ in range(3):
                q_idx = random.randrange(self.num_quinielas)
                m_idx = random.randrange(self.num_matches)

                # No modificar anclas
                if self.match_data.iloc[m_idx]['clasificacion'] == 'Ancla':
                    continue

                quiniela = neighbor[q_idx]
                resultado_actual = quiniela['resultados'][m_idx]
                
                # Cambio simple a resultado aleatorio
                opciones = ['L', 'E', 'V']
                opciones.remove(resultado_actual)
                nuevo_resultado = random.choice(opciones)
                
                quiniela['resultados'][m_idx] = nuevo_resultado
                
                # Verificar empates válidos
                empates = quiniela['resultados'].count('E')
                if self.rules['empates_min'] <= empates <= self.rules['empates_max']:
                    quiniela['empates'] = empates
                    quiniela['distribución'] = {
                        "L": quiniela['resultados'].count("L"),
                        "E": empates,
                        "V": quiniela['resultados'].count("V")
                    }
                    return neighbor

            return None
        except:
            return None

    def _generate_heuristic_fallback(self) -> List[Dict[str, Any]]:
        """
        Generación heurística rápida como fallback cuando IP falla
        """
        logger.info("🔧 Generando portafolio con método heurístico rápido...")
        
        try:
            portfolio = []
            
            # Generar cada quiniela de forma simple
            for j in range(self.num_quinielas):
                quiniela_results = []
                
                for i in range(self.num_matches):
                    clasificacion = self.match_data.iloc[i]['clasificacion']
                    
                    if clasificacion == 'Ancla':
                        # Usar resultado más probable
                        probs = [
                            self.match_data.iloc[i]['prob_local'],
                            self.match_data.iloc[i]['prob_empate'],
                            self.match_data.iloc[i]['prob_visitante']
                        ]
                        resultado = ['L', 'E', 'V'][np.argmax(probs)]
                    else:
                        # Resultado basado en probabilidades con algo de aleatoriedad
                        probs = [
                            self.match_data.iloc[i]['prob_local'],
                            self.match_data.iloc[i]['prob_empate'],
                            self.match_data.iloc[i]['prob_visitante']
                        ]
                        resultado = np.random.choice(['L', 'E', 'V'], p=probs)
                    
                    quiniela_results.append(resultado)
                
                # Ajustar empates si es necesario
                quiniela_results = self._adjust_empates_simple(quiniela_results)
                
                # Metadata
                if j < self.num_cores:
                    quiniela_type = "Core"
                    quiniela_id = f"Core-{j+1}"
                    par_id = None
                else:
                    quiniela_type = "Satelite"
                    sat_index = j - self.num_cores
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
            
            logger.info("✅ Portafolio heurístico generado exitosamente")
            return portfolio
            
        except Exception as e:
            logger.error(f"❌ Error en fallback heurístico: {e}")
            return []

    def _adjust_empates_simple(self, resultados: List[str]) -> List[str]:
        """Ajusta empates de forma simple"""
        empates_actuales = resultados.count('E')
        
        if empates_actuales < self.rules['empates_min']:
            # Convertir algunos L/V a E
            for i in range(len(resultados)):
                if resultados[i] in ['L', 'V'] and empates_actuales < self.rules['empates_min']:
                    resultados[i] = 'E'
                    empates_actuales += 1
        
        elif empates_actuales > self.rules['empates_max']:
            # Convertir algunos E a L/V
            for i in range(len(resultados)):
                if resultados[i] == 'E' and empates_actuales > self.rules['empates_max']:
                    resultados[i] = 'L' if random.random() > 0.5 else 'V'
                    empates_actuales -= 1
        
        return resultados

    def _generate_simple_fallback(self) -> List[Dict[str, Any]]:
        """
        Último fallback: generar portafolio simple básico
        """
        logger.warning("🆘 Usando último fallback: generación simple")
        
        try:
            portfolio = []
            
            for j in range(30):
                # Quiniela muy simple: rotar resultados
                resultados = []
                for i in range(14):
                    if i % 3 == 0:
                        resultados.append('L')
                    elif i % 3 == 1:
                        resultados.append('E')
                    else:
                        resultados.append('V')
                
                # Asegurar 4-6 empates
                empates_actuales = resultados.count('E')
                if empates_actuales < 4:
                    for i in range(14):
                        if resultados[i] != 'E' and empates_actuales < 4:
                            resultados[i] = 'E'
                            empates_actuales += 1
                
                # Metadata básica
                quiniela_type = "Core" if j < 4 else "Satelite"
                par_id = (j - 4) // 2 if j >= 4 else None
                quiniela_id = f"Core-{j+1}" if j < 4 else f"Sat-{par_id+1}{'A' if (j-4)%2==0 else 'B'}"
                
                empates = resultados.count("E")
                portfolio.append({
                    "id": quiniela_id,
                    "tipo": quiniela_type,
                    "par_id": par_id,
                    "resultados": resultados,
                    "empates": empates,
                    "distribución": {
                        "L": resultados.count("L"),
                        "E": empates,
                        "V": resultados.count("V")
                    }
                })
            
            return portfolio
            
        except:
            return []