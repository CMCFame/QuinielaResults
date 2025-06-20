# progol_optimizer/portfolio/core_generator.py - VARIACIÓN MÍNIMA CONSERVADORA
"""
Generador de Quinielas Core - VARIACIÓN MÍNIMA CONSERVADORA
Crea diferencias pequeñas sin romper reglas básicas
"""

import logging
import random
import copy
from typing import List, Dict, Any

class CoreGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.rangos = self.config["RANGOS_HISTORICOS"]
        self.logger.debug("CoreGenerator MÍNIMO CONSERVADOR inicializado")

    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genera 4 quinielas Core con variación MÍNIMA pero conservadora"""
        self.logger.info("🎯 Generando 4 quinielas Core con VARIACIÓN MÍNIMA CONSERVADORA...")
        
        # Identificar partidos
        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        self.logger.info(f"📌 Anclas: {len(anclas_indices)} en posiciones {[i+1 for i in anclas_indices]}")
        self.logger.info(f"🔧 Modificables: {len(modificables)} en posiciones {[i+1 for i in modificables]}")
        
        # PASO 1: Generar quiniela base óptima (como antes)
        quiniela_base = self._generar_quiniela_base_optima(partidos_clasificados, anclas_indices)
        
        # PASO 2: Crear 4 Core con variaciones SÚPER MÍNIMAS
        quinielas_core = []
        
        for core_num in range(4):
            core_id = f"Core-{core_num + 1}"
            
            if core_num == 0:
                # Core-1: usar base sin cambios
                quiniela = quiniela_base.copy()
            else:
                # Core-2,3,4: cambiar SOLO 1 posición cada una
                quiniela = self._crear_variacion_minimal(quiniela_base, core_num, modificables, partidos_clasificados)
            
            # Crear info
            quiniela_info = self._crear_info_quiniela(core_id, quiniela)
            quinielas_core.append(quiniela_info)
            
            self.logger.info(f"✅ {core_id}: {','.join(quiniela)} (empates: {quiniela.count('E')})")
        
        # Verificar diferencias mínimas
        self._verificar_diferencias_minimas(quinielas_core)
        
        return quinielas_core

    def _generar_quiniela_base_optima(self, partidos: List[Dict[str, Any]], anclas_indices: List[int]) -> List[str]:
        """Genera la quiniela base ÓPTIMA que cumple todas las reglas básicas"""
        
        resultados = [""] * 14
        
        # PASO 1: Fijar Anclas con resultados óptimos
        for idx in anclas_indices:
            resultados[idx] = self._get_resultado_max_prob(partidos[idx])
        
        # PASO 2: Llenar partidos modificables conservadoramente
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        for idx in modificables:
            resultados[idx] = self._get_resultado_max_prob(partidos[idx])
        
        # PASO 3: Ajustar empates (crítico)
        resultados = self._ajustar_empates_conservador(resultados, partidos, anclas_indices)
        
        # PASO 4: Verificar distribución y ajustar si es necesario
        resultados = self._ajustar_distribucion_conservadora(resultados, partidos, anclas_indices)
        
        return resultados

    def _crear_variacion_minimal(self, quiniela_base: List[str], core_num: int, 
                                 modificables: List[int], partidos: List[Dict[str, Any]]) -> List[str]:
        """Crea variación MÍNIMA: cambiar solo 1 posición por Core"""
        
        quiniela_variada = quiniela_base.copy()
        
        if not modificables:
            return quiniela_variada
        
        # Seleccionar UNA posición específica para cada Core
        if core_num == 1:  # Core-2
            # Cambiar primera posición modificable disponible
            pos_a_cambiar = modificables[0] if len(modificables) > 0 else None
            cambio_a = "V"  # Intentar cambiar a visitante
            
        elif core_num == 2:  # Core-3
            # Cambiar segunda posición modificable disponible
            pos_a_cambiar = modificables[1] if len(modificables) > 1 else modificables[0]
            cambio_a = "E"  # Intentar cambiar a empate
            
        elif core_num == 3:  # Core-4
            # Cambiar tercera posición modificable disponible
            pos_a_cambiar = modificables[2] if len(modificables) > 2 else modificables[-1]
            cambio_a = None  # Usar segunda opción más probable
        
        # Aplicar el cambio si es viable
        if pos_a_cambiar is not None:
            resultado_actual = quiniela_variada[pos_a_cambiar]
            
            if core_num == 3:  # Core-4: usar segunda opción
                probs_ordenadas = self._get_probabilidades_ordenadas(partidos[pos_a_cambiar])
                if len(probs_ordenadas) > 1 and probs_ordenadas[1][1] > 0.20:
                    nuevo_resultado = probs_ordenadas[1][0]
                else:
                    nuevo_resultado = resultado_actual  # No cambiar si no es viable
            else:  # Core-2, Core-3: cambiar a resultado específico
                prob_nuevo = partidos[pos_a_cambiar][f"prob_{self._resultado_a_clave(cambio_a)}"]
                if prob_nuevo > 0.20 and cambio_a != resultado_actual:
                    nuevo_resultado = cambio_a
                else:
                    nuevo_resultado = resultado_actual  # No cambiar si no es viable
            
            # Aplicar cambio solo si es diferente y no rompe empates
            if nuevo_resultado != resultado_actual:
                quiniela_test = quiniela_variada.copy()
                quiniela_test[pos_a_cambiar] = nuevo_resultado
                
                # Verificar que sigue teniendo empates válidos
                empates_test = quiniela_test.count("E")
                if self.empates_min <= empates_test <= self.empates_max:
                    quiniela_variada[pos_a_cambiar] = nuevo_resultado
                    self.logger.debug(f"Core-{core_num + 1}: P{pos_a_cambiar + 1} cambiado de {resultado_actual} a {nuevo_resultado}")
                else:
                    self.logger.debug(f"Core-{core_num + 1}: cambio rechazado por empates ({empates_test})")
        
        return quiniela_variada

    def _ajustar_empates_conservador(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                     anclas_indices: List[int]) -> List[str]:
        """Ajuste conservador de empates que preserva distribución"""
        
        empates_actuales = resultados.count("E")
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Si faltan empates, cambiar los mejores candidatos
        while empates_actuales < self.empates_min:
            mejor_candidato = None
            mejor_score = 0
            
            for idx in modificables:
                if resultados[idx] != "E":
                    prob_empate = partidos[idx]["prob_empate"]
                    # Score: probabilidad de empate - penalizar si ya hay muchos de este tipo
                    count_actual = resultados.count(resultados[idx])
                    score = prob_empate - (count_actual * 0.05)  # Penalización leve
                    
                    if score > mejor_score:
                        mejor_score = score
                        mejor_candidato = idx
            
            if mejor_candidato is not None:
                resultados[mejor_candidato] = "E"
                empates_actuales += 1
            else:
                break
        
        # Si sobran empates, cambiar los peores candidatos
        while empates_actuales > self.empates_max:
            peor_candidato = None
            peor_score = 1.0
            
            for idx in modificables:
                if resultados[idx] == "E":
                    prob_empate = partidos[idx]["prob_empate"]
                    
                    if prob_empate < peor_score:
                        peor_score = prob_empate
                        peor_candidato = idx
            
            if peor_candidato is not None:
                # Cambiar al resultado más probable (no empate)
                probs_ordenadas = self._get_probabilidades_ordenadas(partidos[peor_candidato])
                for resultado, prob in probs_ordenadas:
                    if resultado != "E":
                        resultados[peor_candidato] = resultado
                        break
                empates_actuales -= 1
            else:
                break
        
        return resultados

    def _ajustar_distribucion_conservadora(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                           anclas_indices: List[int]) -> List[str]:
        """Ajuste conservador de distribución L/E/V"""
        
        # Calcular distribución actual
        total_L = resultados.count("L")
        total_E = resultados.count("E")
        total_V = resultados.count("V")
        
        # Distribución target (punto medio de rangos históricos)
        target_L = int(14 * 0.38)  # ~5
        target_E = int(14 * 0.29)  # ~4
        target_V = int(14 * 0.33)  # ~5
        
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Solo hacer ajustes MUY conservadores si hay desequilibrios grandes
        if abs(total_L - target_L) > 2 or abs(total_V - target_V) > 2:
            
            # Si hay exceso de L, cambiar algunos L a V
            if total_L > target_L + 1 and total_V < target_V:
                candidatos_L = [(i, partidos[i]["prob_local"]) for i in modificables if resultados[i] == "L"]
                candidatos_L.sort(key=lambda x: x[1])  # Ordenar por prob ascendente
                
                for i, (idx, _) in enumerate(candidatos_L):
                    if i >= abs(total_L - target_L):
                        break
                    
                    prob_V = partidos[idx]["prob_visitante"]
                    if prob_V > 0.25:  # Solo si V es razonablemente probable
                        resultados[idx] = "V"
                        total_L -= 1
                        total_V += 1
        
        return resultados

    def _verificar_diferencias_minimas(self, quinielas_core: List[Dict[str, Any]]):
        """Verifica que hay al menos diferencias mínimas"""
        
        self.logger.info("=== VERIFICACIÓN DE DIFERENCIAS MÍNIMAS ===")
        
        diferencias_totales = 0
        
        for i in range(len(quinielas_core)):
            for j in range(i + 1, len(quinielas_core)):
                q1 = quinielas_core[i]
                q2 = quinielas_core[j]
                
                coincidencias = sum(1 for a, b in zip(q1["resultados"], q2["resultados"]) if a == b)
                diferencias = 14 - coincidencias
                diferencias_totales += diferencias
                
                if diferencias == 0:
                    self.logger.warning(f"⚠️ {q1['id']} y {q2['id']} son IDÉNTICAS!")
                elif diferencias == 1:
                    self.logger.info(f"✅ {q1['id']} vs {q2['id']}: {diferencias} diferencia (mínima)")
                else:
                    self.logger.info(f"✅ {q1['id']} vs {q2['id']}: {diferencias} diferencias")
        
        promedio_diferencias = diferencias_totales / 6  # 6 pares posibles
        self.logger.info(f"📊 Promedio de diferencias: {promedio_diferencias:.1f}")

    # Funciones auxiliares (sin cambios)
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        return max(probs, key=probs.get)

    def _get_probabilidades_ordenadas(self, partido: Dict[str, Any]) -> List[tuple]:
        """Obtiene probabilidades ordenadas de mayor a menor"""
        probs = [
            ("L", partido["prob_local"]),
            ("E", partido["prob_empate"]),
            ("V", partido["prob_visitante"])
        ]
        return sorted(probs, key=lambda x: x[1], reverse=True)

    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte resultado a clave de probabilidad"""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "local")

    def _crear_info_quiniela(self, q_id: str, resultados: List[str]) -> Dict[str, Any]:
        """Crea estructura completa de quiniela"""
        return {
            "id": q_id,
            "tipo": "Core",
            "resultados": resultados,
            "empates": resultados.count("E"),
            "distribución": {
                "L": resultados.count("L"),
                "E": resultados.count("E"),
                "V": resultados.count("V")
            }
        }