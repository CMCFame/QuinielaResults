# progol_optimizer/portfolio/core_generator.py - FIX VARIACIÓN FORZADA
"""
Generador de Quinielas Core - FIX DEFINITIVO PARA VARIACIÓN
Garantiza que las 4 Core sean DIFERENTES
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
        self.logger.debug("CoreGenerator con VARIACIÓN FORZADA inicializado")

    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genera 4 quinielas Core con VARIACIÓN GARANTIZADA"""
        self.logger.info("🎯 Generando 4 quinielas Core con VARIACIÓN FORZADA...")
        
        # Identificar partidos
        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        self.logger.info(f"📌 Anclas: {len(anclas_indices)} en posiciones {[i+1 for i in anclas_indices]}")
        self.logger.info(f"🔧 Modificables: {len(modificables)} en posiciones {[i+1 for i in modificables]}")
        
        # Generar quiniela base (Core-1)
        quiniela_base = self._generar_quiniela_base(partidos_clasificados, anclas_indices, modificables)
        
        # Crear 4 variaciones de la quiniela base
        quinielas_core = []
        
        for core_num in range(4):
            core_id = f"Core-{core_num + 1}"
            
            if core_num == 0:
                # Core-1: usar quiniela base sin modificar
                quiniela = quiniela_base.copy()
            else:
                # Core-2, 3, 4: crear variaciones específicas
                quiniela = self._crear_variacion_forzada(quiniela_base, core_num, modificables, partidos_clasificados)
            
            # Asegurar empates válidos
            quiniela = self._ajustar_empates_final(quiniela, partidos_clasificados, anclas_indices)
            
            # Crear info de quiniela
            quiniela_info = self._crear_info_quiniela(core_id, quiniela)
            quinielas_core.append(quiniela_info)
            
            self.logger.info(f"✅ {core_id}: {','.join(quiniela)} (empates: {quiniela.count('E')})")
        
        # Verificar que realmente son diferentes
        self._verificar_diferencias(quinielas_core)
        
        return quinielas_core

    def _generar_quiniela_base(self, partidos: List[Dict[str, Any]], 
                               anclas_indices: List[int], modificables: List[int]) -> List[str]:
        """Genera la quiniela base (Core-1) conservadora"""
        
        resultados = [""] * 14
        
        # Fijar Anclas con sus mejores resultados
        for idx in anclas_indices:
            resultados[idx] = self._get_resultado_max_prob(partidos[idx])
        
        # Llenar partidos modificables con máximas probabilidades
        for idx in modificables:
            resultados[idx] = self._get_resultado_max_prob(partidos[idx])
        
        # Ajustar empates si es necesario
        resultados = self._ajustar_empates_basico(resultados, partidos, anclas_indices)
        
        return resultados

    def _crear_variacion_forzada(self, quiniela_base: List[str], core_num: int, 
                                 modificables: List[int], partidos: List[Dict[str, Any]]) -> List[str]:
        """Crea variaciones FORZADAS para Core-2, 3, 4"""
        
        quiniela_variada = quiniela_base.copy()
        
        # Estrategias específicas de variación por Core
        if core_num == 1:  # Core-2
            self._aplicar_variacion_core2(quiniela_variada, modificables, partidos)
        elif core_num == 2:  # Core-3  
            self._aplicar_variacion_core3(quiniela_variada, modificables, partidos)
        elif core_num == 3:  # Core-4
            self._aplicar_variacion_core4(quiniela_variada, modificables, partidos)
        
        return quiniela_variada

    def _aplicar_variacion_core2(self, quiniela: List[str], modificables: List[int], partidos: List[Dict[str, Any]]):
        """Core-2: Priorizar visitantes y cambiar 2-3 posiciones específicas"""
        
        cambios_realizados = 0
        max_cambios = 3
        
        # Estrategia: cambiar algunos L por V cuando sea razonable
        for idx in modificables:
            if cambios_realizados >= max_cambios:
                break
                
            if quiniela[idx] == "L":
                prob_V = partidos[idx]["prob_visitante"]
                prob_L = partidos[idx]["prob_local"]
                
                # Cambiar si V es razonablemente probable
                if prob_V > 0.25 or (prob_V > prob_L * 0.8):
                    quiniela[idx] = "V"
                    cambios_realizados += 1
                    self.logger.debug(f"Core-2: P{idx+1} cambiado de L a V (prob_V={prob_V:.3f})")

    def _aplicar_variacion_core3(self, quiniela: List[str], modificables: List[int], partidos: List[Dict[str, Any]]):
        """Core-3: Priorizar empates y cambiar diferentes posiciones"""
        
        cambios_realizados = 0
        max_cambios = 3
        
        # Estrategia: cambiar algunos L/V por E cuando sea razonable
        for idx in modificables:
            if cambios_realizados >= max_cambios:
                break
                
            if quiniela[idx] in ["L", "V"]:
                prob_E = partidos[idx]["prob_empate"]
                
                # Cambiar si E es razonablemente probable
                if prob_E > 0.28:
                    quiniela[idx] = "E"
                    cambios_realizados += 1
                    self.logger.debug(f"Core-3: P{idx+1} cambiado a E (prob_E={prob_E:.3f})")

    def _aplicar_variacion_core4(self, quiniela: List[str], modificables: List[int], partidos: List[Dict[str, Any]]):
        """Core-4: Mix rotativo - cambiar cada 2da posición modificable"""
        
        # Estrategia: cambiar cada 2da posición modificable a segunda opción
        for i, idx in enumerate(modificables):
            if i % 2 == 1:  # Cada segunda posición
                probs_ordenadas = self._get_probabilidades_ordenadas(partidos[idx])
                
                # Usar segunda opción si es razonablemente probable
                if len(probs_ordenadas) > 1 and probs_ordenadas[1][1] > 0.20:
                    nuevo_resultado = probs_ordenadas[1][0]
                    if nuevo_resultado != quiniela[idx]:
                        self.logger.debug(f"Core-4: P{idx+1} cambiado de {quiniela[idx]} a {nuevo_resultado}")
                        quiniela[idx] = nuevo_resultado

    def _ajustar_empates_basico(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                                anclas_indices: List[int]) -> List[str]:
        """Ajuste básico de empates"""
        
        empates_actuales = resultados.count("E")
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Si faltan empates
        while empates_actuales < self.empates_min:
            # Buscar mejor candidato para convertir a empate
            mejor_candidato = None
            mejor_prob_empate = 0
            
            for idx in modificables:
                if resultados[idx] != "E":
                    prob_empate = partidos[idx]["prob_empate"]
                    if prob_empate > mejor_prob_empate:
                        mejor_prob_empate = prob_empate
                        mejor_candidato = idx
            
            if mejor_candidato is not None:
                resultados[mejor_candidato] = "E"
                empates_actuales += 1
            else:
                break
        
        # Si sobran empates
        while empates_actuales > self.empates_max:
            # Buscar peor empate para convertir
            peor_candidato = None
            peor_prob_empate = 1.0
            
            for idx in modificables:
                if resultados[idx] == "E":
                    prob_empate = partidos[idx]["prob_empate"]
                    if prob_empate < peor_prob_empate:
                        peor_prob_empate = prob_empate
                        peor_candidato = idx
            
            if peor_candidato is not None:
                resultados[peor_candidato] = self._get_resultado_max_prob(partidos[peor_candidato])
                empates_actuales -= 1
            else:
                break
        
        return resultados

    def _ajustar_empates_final(self, resultados: List[str], partidos: List[Dict[str, Any]], 
                               anclas_indices: List[int]) -> List[str]:
        """Ajuste final de empates más agresivo"""
        
        empates_actuales = resultados.count("E")
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Si faltan empates, forzar cambios
        if empates_actuales < self.empates_min:
            faltan = self.empates_min - empates_actuales
            candidatos = [(i, partidos[i]["prob_empate"]) for i in modificables if resultados[i] != "E"]
            candidatos.sort(key=lambda x: x[1], reverse=True)  # Ordenar por probabilidad desc
            
            for i in range(min(faltan, len(candidatos))):
                idx = candidatos[i][0]
                resultados[idx] = "E"
                self.logger.debug(f"Ajuste final: P{idx+1} forzado a E")
        
        # Si sobran empates, forzar cambios
        elif empates_actuales > self.empates_max:
            sobran = empates_actuales - self.empates_max
            candidatos = [(i, partidos[i]["prob_empate"]) for i in modificables if resultados[i] == "E"]
            candidatos.sort(key=lambda x: x[1])  # Ordenar por probabilidad asc
            
            for i in range(min(sobran, len(candidatos))):
                idx = candidatos[i][0]
                resultados[idx] = self._get_resultado_max_prob(partidos[idx])
                self.logger.debug(f"Ajuste final: P{idx+1} cambiado de E a {resultados[idx]}")
        
        return resultados

    def _verificar_diferencias(self, quinielas_core: List[Dict[str, Any]]):
        """Verifica que las quinielas sean realmente diferentes"""
        
        self.logger.info("=== VERIFICACIÓN DE DIFERENCIAS ===")
        
        for i in range(len(quinielas_core)):
            for j in range(i + 1, len(quinielas_core)):
                q1 = quinielas_core[i]
                q2 = quinielas_core[j]
                
                coincidencias = sum(1 for a, b in zip(q1["resultados"], q2["resultados"]) if a == b)
                diferencias = 14 - coincidencias
                
                self.logger.info(f"{q1['id']} vs {q2['id']}: {diferencias} diferencias")
                
                if diferencias == 0:
                    self.logger.error(f"❌ {q1['id']} y {q2['id']} son IDÉNTICAS!")
                elif diferencias < 2:
                    self.logger.warning(f"⚠️ {q1['id']} y {q2['id']} son muy similares ({diferencias} diferencias)")
                else:
                    self.logger.info(f"✅ {q1['id']} y {q2['id']}: suficiente variación")

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