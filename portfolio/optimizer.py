"""
Optimizador GRASP-Annealing - CORRECCIóN DEFINITIVA
Menos restrictivo durante optimización, más agresivo en ajuste final
"""

import logging
# ... (el resto del archivo continúa como estaba)

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
        
        # NUEVO: Intentar cargar asistente AI
        try:
            from models.ai_assistant import ProgolAIAssistant
            self.ai_assistant = ProgolAIAssistant()
        except:
            self.ai_assistant = None
            self.logger.debug("Asistente AI no disponible")

    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte resultado a clave de probabilidad - CORREGIDO"""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "local")

    def optimizar_portafolio_grasp_annealing(self, quinielas_iniciales: List[Dict[str, Any]],
                                           partidos: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
        self.logger.info("馃殌 Iniciando optimizaci贸n GRASP-Annealing DEFINITIVA...")
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
                    self.logger.debug(f"Iter {iteracion}: Nueva mejor soluci贸n -> Score {mejor_score:.6f}")
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
                self.logger.info(f"鈴癸笍 Parada temprana en iteraci贸n {iteracion} (sin mejora por {iteraciones_sin_mejora} iteraciones)")
                break
        
        self.logger.info("馃敡 Fase final: Ajuste agresivo de concentraci贸n y distribuci贸n...")
        mejor_portafolio = self._ajuste_final_definitivo(mejor_portafolio, partidos)

        score_final = self._calcular_objetivo_f_optimizado(mejor_portafolio, partidos)
        self.logger.info(f"鉁?Optimizaci贸n DEFINITIVA completada: F={score_final:.6f}")
        return mejor_portafolio

    def _generar_movimiento_valido_mejorado(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generaci贸n de movimientos MENOS restrictiva durante optimizaci贸n"""
        intentos = 0
        while intentos < 50:  # M谩s intentos
            intentos += 1
            
            nuevo_portafolio = [q.copy() for q in portafolio]
            quiniela_idx = random.randrange(len(nuevo_portafolio))
            
            # Priorizar sat茅lites (80% de probabilidad)
            if random.random() < 0.8:
                satelite_indices = [i for i, q in enumerate(nuevo_portafolio) if q['tipo'] == 'Satelite']
                if satelite_indices:
                    quiniela_idx = random.choice(satelite_indices)
            
            quiniela_original = nuevo_portafolio[quiniela_idx]
            nuevos_resultados = quiniela_original["resultados"].copy()
            
            partidos_modificables = [i for i, p in enumerate(partidos) if p.get("clasificacion") != "Ancla"]
            if not partidos_modificables: 
                continue
            
            # Cambios m谩s peque帽os para mayor 茅xito
            num_cambios = random.choice([1, 1, 2])  # M谩s probabilidad de 1 cambio
            indices_a_cambiar = random.sample(partidos_modificables, min(num_cambios, len(partidos_modificables)))

            for idx in indices_a_cambiar:
                resultado_actual = nuevos_resultados[idx]
                opciones = ["L", "E", "V"]
                opciones.remove(resultado_actual)
                nuevos_resultados[idx] = random.choice(opciones)
            
            # Validaci贸n MENOS estricta durante optimizaci贸n
            if self._es_movimiento_valido_permisivo(nuevo_portafolio, quiniela_idx, nuevos_resultados):
                quiniela_modificada = quiniela_original.copy()
                quiniela_modificada["resultados"] = nuevos_resultados
                quiniela_modificada["empates"] = nuevos_resultados.count("E")
                quiniela_modificada["distribuci贸n"] = {
                    "L": nuevos_resultados.count("L"), 
                    "E": nuevos_resultados.count("E"), 
                    "V": nuevos_resultados.count("V")
                }
                nuevo_portafolio[quiniela_idx] = quiniela_modificada
                return nuevo_portafolio
                
        return None

    def _es_movimiento_valido_permisivo(self, portafolio: List[Dict[str, Any]], quiniela_idx: int, nuevos_resultados: List[str]) -> bool:
        """Validaci贸n PERMISIVA durante optimizaci贸n (el ajuste final corregir谩 problemas)"""
        
        # 1. Validar empates (obligatorio)
        empates = nuevos_resultados.count("E")
        if not (self.config["EMPATES_MIN"] <= empates <= self.config["EMPATES_MAX"]):
            return False

        # 2. Concentraci贸n M脕S PERMISIVA durante optimizaci贸n (85% vs 70%)
        max_conc = max(nuevos_resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc > 0.85:  # 85% en lugar de 70%
            return False
            
        # 3. Concentraci贸n inicial M脕S PERMISIVA (85% vs 60%)
        max_conc_inicial = max(nuevos_resultados[:3].count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > 0.85:  # 85% en lugar de 60%
            return False

        # 4. Validar correlaci贸n Jaccard para sat茅lites
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
        AJUSTE FINAL DEFINITIVO: Corrige TODO lo necesario para pasar validaci贸n
        """
        self.logger.info("馃敡 Iniciando ajuste final DEFINITIVO...")
        portafolio_ajustado = [q.copy() for q in portafolio]

        # PASO 1: Correcci贸n AGRESIVA de concentraci贸n individual
        self.logger.info("馃幆 Paso 1: Correcci贸n AGRESIVA de concentraci贸n individual...")
        for iter_conc in range(3):  # M煤ltiples pasadas
            cambios_realizados = 0
            for i, quiniela in enumerate(portafolio_ajustado):
                if quiniela["tipo"] == "Core":
                    continue
                    
                resultados_originales = quiniela["resultados"].copy()
                resultados_corregidos = self._forzar_concentracion_valida(resultados_originales, partidos)
                
                if resultados_originales != resultados_corregidos:
                    portafolio_ajustado[i]["resultados"] = resultados_corregidos
                    portafolio_ajustado[i]["empates"] = resultados_corregidos.count("E")
                    portafolio_ajustado[i]["distribuci贸n"] = {
                        "L": resultados_corregidos.count("L"),
                        "E": resultados_corregidos.count("E"),
                        "V": resultados_corregidos.count("V")
                    }
                    cambios_realizados += 1
            
            self.logger.debug(f"Iteraci贸n concentraci贸n {iter_conc+1}: {cambios_realizados} cambios")
            if cambios_realizados == 0:
                break

        # PASO 2: Correcci贸n AGRESIVA de distribuci贸n por posici贸n
        self.logger.info("馃幆 Paso 2: Correcci贸n AGRESIVA de distribuci贸n por posici贸n...")
        for iter_dist in range(5):  # M煤ltiples pasadas
            cambios_realizados = 0
            
            # Analizar cada posici贸n
            for posicion in range(14):
                cambios_realizados += self._balancear_posicion_agresivo(portafolio_ajustado, posicion, partidos)
            
            self.logger.debug(f"Iteraci贸n distribuci贸n {iter_dist+1}: {cambios_realizados} cambios")
            if cambios_realizados == 0:
                break

        # PASO 3: Verificaci贸n final
        concentracion_ok = self.validator._validar_concentracion_70_60(portafolio_ajustado)
        distribucion_ok = self.validator._validar_distribucion_equilibrada(portafolio_ajustado)
        
        self.logger.info(f"馃搳 Resultado final: Concentraci贸n={concentracion_ok}, Distribuci贸n={distribucion_ok}")
        
        return portafolio_ajustado

    def _forzar_concentracion_valida(self, resultados: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """
        FUERZA que una quiniela tenga concentraci贸n v谩lida 鈮?0% general y 鈮?0% primeros 3
        """
        resultados_corregidos = resultados.copy()
        anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Correcci贸n 1: Concentraci贸n general 鈮?0% (m谩ximo 9 de 14)
        max_permitido_general = int(14 * 0.70)  # 9
        for signo in ["L", "E", "V"]:
            count_signo = resultados_corregidos.count(signo)
            if count_signo > max_permitido_general:
                # Encontrar 铆ndices de este signo que se pueden modificar
                indices_modificables = [i for i in modificables if resultados_corregidos[i] == signo]
                # Ordenar por probabilidad (cambiar los menos probables primero)
                indices_modificables.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                
                # Cambiar hasta que sea v谩lido
                exceso = count_signo - max_permitido_general
                for j in range(min(exceso, len(indices_modificables))):
                    idx = indices_modificables[j]
                    # Cambiar al resultado que menos tengamos
                    otros_signos = [s for s in ["L", "E", "V"] if s != signo]
                    mejor_cambio = min(otros_signos, key=lambda s: resultados_corregidos.count(s))
                    resultados_corregidos[idx] = mejor_cambio

        # Correcci贸n 2: Concentraci贸n inicial 鈮?0% (m谩ximo 1 de 3)
        max_permitido_inicial = int(3 * 0.60)  # 1
        for signo in ["L", "E", "V"]:
            count_inicial = resultados_corregidos[:3].count(signo)
            if count_inicial > max_permitido_inicial:
                # Encontrar 铆ndices en primeros 3 partidos
                indices_inicial = [i for i in range(3) if resultados_corregidos[i] == signo and i not in anclas_indices]
                # Ordenar por probabilidad
                indices_inicial.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                
                # Cambiar hasta que sea v谩lido
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
            Balancea AGRESIVAMENTE una posición específica para cumplir con max y min de apariciones.
            """
            if partidos[posicion].get("clasificacion") == "Ancla":
                return 0  # No tocar anclas

            total_quinielas = len(portafolio)
            max_apariciones = int(total_quinielas * 0.67)
            min_apariciones = int(total_quinielas * 0.10)
            
            conteos = {"L": 0, "E": 0, "V": 0}
            indices_por_signo = {"L": [], "E": [], "V": []}

            # Solo modificar satélites para mantener los Core estables
            for i, q in enumerate(portafolio):
                if q.get("tipo") == "Satelite":
                    resultado = q["resultados"][posicion]
                    conteos[resultado] += 1
                    indices_por_signo[resultado].append(i)

            cambios_realizados = 0

            # Corregir exceso (signos que aparecen DEMASIADO)
            for signo_exceso in ["L", "E", "V"]:
                if conteos[signo_exceso] > max_apariciones:
                    exceso = conteos[signo_exceso] - max_apariciones
                    signos_destino = [s for s, c in conteos.items() if c < max_apariciones]
                    if not signos_destino: continue

                    candidatos_a_cambiar = indices_por_signo[signo_exceso]
                    candidatos_a_cambiar.sort(key=lambda i: partidos[posicion][f"prob_{self._resultado_a_clave(signo_exceso)}"])

                    for i in range(min(exceso, len(candidatos_a_cambiar))):
                        q_idx = candidatos_a_cambiar[i]
                        destino = min(signos_destino, key=lambda s: conteos[s])
                        
                        portafolio[q_idx]["resultados"][posicion] = destino
                        cambios_realizados += 1
                        conteos[signo_exceso] -= 1
                        conteos[destino] += 1
            
            # Corregir defecto (signos que aparecen MUY POCO)
            for signo_defecto in ["L", "E", "V"]:
                if conteos[signo_defecto] < min_apariciones:
                    necesarios = min_apariciones - conteos[signo_defecto]
                    signos_fuente = [s for s, c in conteos.items() if c > min_apariciones + necesarios]
                    if not signos_fuente: continue
                    
                    fuente = max(signos_fuente, key=lambda s: conteos[s])
                    
                    candidatos_a_cambiar = indices_por_signo[fuente]
                    if not candidatos_a_cambiar: continue
                    
                    candidatos_a_cambiar.sort(key=lambda i: partidos[posicion][f"prob_{self._resultado_a_clave(fuente)}"], reverse=True)

                    for i in range(min(necesarios, len(candidatos_a_cambiar))):
                        q_idx = candidatos_a_cambiar[i]
                        portafolio[q_idx]["resultados"][posicion] = signo_defecto
                        cambios_realizados += 1
                        conteos[fuente] -= 1
                        conteos[signo_defecto] += 1

            # Re-calcular la distribución en las quinielas modificadas si hubo cambios
            if cambios_realizados > 0:
                quinielas_modificadas_indices = {idx for sublist in indices_por_signo.values() for idx in sublist}
                for i in quinielas_modificadas_indices:
                    quiniela = portafolio[i]
                    resultados = quiniela["resultados"]
                    quiniela["empates"] = resultados.count("E")
                    quiniela["distribución"] = {"L": resultados.count("L"), "E": resultados.count("E"), "V": resultados.count("V")}
            
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
        """CORREGIDO: C谩lculo r谩pido de probabilidad 鈮?1 aciertos"""
        num_simulaciones = 1000
        
        # CORRECCI脫N: Mapeo correcto de nombres
        prob_acierto = np.array([
            partidos[i][f"prob_{self._resultado_a_clave(res)}"] 
            for i, res in enumerate(resultados)
        ])
        
        simulaciones = np.random.rand(num_simulaciones, 14)
        aciertos = np.sum(simulaciones < prob_acierto, axis=1)
        aciertos_11_plus = np.sum(aciertos >= 11)
        return aciertos_11_plus / num_simulaciones

    def _crear_cache_key(self, portafolio: List[Dict[str, Any]]) -> int:
        return hash(tuple("".join(q["resultados"]) for q in portafolio))