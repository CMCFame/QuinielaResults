# progol_optimizer/portfolio/optimizer.py - CORREGIDO PARA CONCENTRACIÓN
"""
Optimizador GRASP-Annealing CORREGIDO
CORRECCIÓN PRINCIPAL: Arregla la función de ajuste final duplicada y mejora la corrección de concentración
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
        mejor_portafolio = self._ajuste_final_del_portafolio_corregido(mejor_portafolio, partidos)

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

    def _ajuste_final_del_portafolio_corregido(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        VERSIÓN CORREGIDA: Ajuste final que corrige concentración Y distribución por posición
        """
        self.logger.info("Iniciando ajuste final CORREGIDO del portafolio...")
        portafolio_ajustado = [q.copy() for q in portafolio]

        # --- PASO 1: CORREGIR CONCENTRACIÓN EN QUINIELAS INDIVIDUALES ---
        self.logger.info("Paso 1: Corrigiendo concentración individual...")
        for i, quiniela in enumerate(portafolio_ajustado):
            if quiniela["tipo"] == "Core":  # No tocar las Core
                continue
                
            resultados = quiniela["resultados"].copy()
            
            # Verificar concentración general
            if self._tiene_concentracion_excesiva(resultados):
                self.logger.debug(f"Corrigiendo concentración en {quiniela['id']}")
                resultados_corregidos = self._corregir_concentracion_quiniela(resultados, partidos)
                
                # Actualizar quiniela
                portafolio_ajustado[i]["resultados"] = resultados_corregidos
                portafolio_ajustado[i]["empates"] = resultados_corregidos.count("E")
                portafolio_ajustado[i]["distribución"] = {
                    "L": resultados_corregidos.count("L"),
                    "E": resultados_corregidos.count("E"),
                    "V": resultados_corregidos.count("V")
                }

        # --- PASO 2: CORREGIR DISTRIBUCIÓN POR POSICIÓN ---
        self.logger.info("Paso 2: Corrigiendo distribución por posición...")
        for intento in range(100):  # Máximo 100 intentos
            if self.validator._validar_distribucion_equilibrada(portafolio_ajustado):
                self.logger.info("✅ Distribución por posición corregida")
                break
                
            # Encontrar la peor violación
            peor_posicion, peor_signo = self._encontrar_peor_desequilibrio(portafolio_ajustado)
            
            if peor_posicion is None:
                break
                
            # Corregir la violación
            if self._corregir_desequilibrio_posicion(portafolio_ajustado, peor_posicion, peor_signo, partidos):
                self.logger.debug(f"Corregido desequilibrio en posición {peor_posicion+1}, signo {peor_signo}")
            else:
                # Si no se puede corregir, salir del bucle
                break
        else:
            self.logger.warning("⚠️ No se pudo corregir completamente la distribución por posición")

        return portafolio_ajustado

    def _tiene_concentracion_excesiva(self, resultados: List[str]) -> bool:
        """Verifica si una quiniela tiene concentración excesiva"""
        # Concentración general
        max_conc_general = max(resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc_general > self.config["CONCENTRACION_MAX_GENERAL"]:
            return True
            
        # Concentración inicial
        primeros_3 = resultados[:3]
        max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > self.config["CONCENTRACION_MAX_INICIAL"]:
            return True
            
        return False

    def _corregir_concentracion_quiniela(self, resultados: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """Corrige la concentración de una quiniela individual"""
        resultados_corregidos = resultados.copy()
        
        # Corregir concentración general
        for signo in ["L", "E", "V"]:
            concentracion = resultados_corregidos.count(signo) / 14.0
            if concentracion > self.config["CONCENTRACION_MAX_GENERAL"]:
                # Cambiar los menos probables de este signo
                indices_signo = [i for i, r in enumerate(resultados_corregidos) if r == signo]
                # Ordenar por probabilidad (cambiar los menos probables)
                indices_signo.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                
                # Cambiar hasta que la concentración sea válida
                cambios_necesarios = int(resultados_corregidos.count(signo) - 14 * self.config["CONCENTRACION_MAX_GENERAL"])
                for i in range(min(cambios_necesarios, len(indices_signo))):
                    idx = indices_signo[i]
                    # Cambiar al resultado más probable que no sea el actual
                    opciones = ["L", "E", "V"]
                    opciones.remove(signo)
                    mejor_opcion = max(opciones, key=lambda s: partidos[idx][f"prob_{self._resultado_a_clave(s)}"])
                    resultados_corregidos[idx] = mejor_opcion

        # Corregir concentración inicial
        primeros_3 = resultados_corregidos[:3]
        for signo in ["L", "E", "V"]:
            concentracion_inicial = primeros_3.count(signo) / 3.0
            if concentracion_inicial > self.config["CONCENTRACION_MAX_INICIAL"]:
                # Cambiar en los primeros 3 partidos
                indices_signo = [i for i in range(3) if resultados_corregidos[i] == signo]
                if len(indices_signo) > int(3 * self.config["CONCENTRACION_MAX_INICIAL"]):
                    # Cambiar el menos probable
                    indices_signo.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                    idx_cambiar = indices_signo[0]
                    opciones = ["L", "E", "V"]
                    opciones.remove(signo)
                    mejor_opcion = max(opciones, key=lambda s: partidos[idx_cambiar][f"prob_{self._resultado_a_clave(s)}"])
                    resultados_corregidos[idx_cambiar] = mejor_opcion

        return resultados_corregidos

    def _encontrar_peor_desequilibrio(self, portafolio: List[Dict[str, Any]]) -> tuple:
        """Encuentra la peor violación de distribución por posición"""
        max_desequilibrio = 0
        peor_posicion = None
        peor_signo = None
        
        max_apariciones = len(portafolio) * 0.67  # 67% máximo
        
        for posicion in range(14):
            conteos = {"L": 0, "E": 0, "V": 0}
            for q in portafolio:
                conteos[q["resultados"][posicion]] += 1
            
            for signo, count in conteos.items():
                if count > max_apariciones and count > max_desequilibrio:
                    max_desequilibrio = count
                    peor_posicion = posicion
                    peor_signo = signo
        
        return peor_posicion, peor_signo

    def _corregir_desequilibrio_posicion(self, portafolio: List[Dict[str, Any]], posicion: int, signo_exceso: str, partidos: List[Dict[str, Any]]) -> bool:
        """Corrige el desequilibrio en una posición específica"""
        # Buscar quinielas Satélite que tengan el signo en exceso en esa posición
        candidatos = []
        for i, q in enumerate(portafolio):
            if q["tipo"] == "Satelite" and q["resultados"][posicion] == signo_exceso:
                # Evaluar qué tan "fácil" es cambiar este resultado
                prob_actual = partidos[posicion][f"prob_{self._resultado_a_clave(signo_exceso)}"]
                candidatos.append((i, prob_actual))
        
        # Ordenar por probabilidad (cambiar primero los menos probables)
        candidatos.sort(key=lambda x: x[1])
        
        # Intentar cambiar el menos probable
        for q_idx, _ in candidatos[:3]:  # Intentar con los 3 menos probables
            quiniela = portafolio[q_idx]
            resultados_nuevos = quiniela["resultados"].copy()
            
            # Intentar cambiar a cada una de las otras opciones
            opciones = [s for s in ["L", "E", "V"] if s != signo_exceso]
            for nuevo_resultado in opciones:
                resultados_test = resultados_nuevos.copy()
                resultados_test[posicion] = nuevo_resultado
                
                # Verificar que el movimiento sea válido
                if self._es_movimiento_valido(portafolio, q_idx, resultados_test):
                    # Aplicar el cambio
                    portafolio[q_idx]["resultados"] = resultados_test
                    portafolio[q_idx]["empates"] = resultados_test.count("E")
                    portafolio[q_idx]["distribución"] = {
                        "L": resultados_test.count("L"),
                        "E": resultados_test.count("E"),
                        "V": resultados_test.count("V")
                    }
                    return True
        
        return False

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
        
        prob_acierto = np.array([partidos[i][f"prob_{self._resultado_a_clave(res)}"] for i, res in enumerate(resultados)])
        
        simulaciones = np.random.rand(num_simulaciones, 14)
        aciertos = np.sum(simulaciones < prob_acierto, axis=1)
        aciertos_11_plus = np.sum(aciertos >= 11)
                
        return aciertos_11_plus / num_simulaciones

    def _crear_cache_key(self, portafolio: List[Dict[str, Any]]) -> int:
        return hash(tuple("".join(q["resultados"]) for q in portafolio))