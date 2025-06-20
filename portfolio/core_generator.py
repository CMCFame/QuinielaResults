# progol_optimizer/portfolio/core_generator.py - CORREGIDO
"""
Generador de Quinielas Core CORREGIDO - Variación en partidos no-Ancla
CORRECCIÓN: Las Core deben ser iguales en Anclas pero diferentes en otros partidos
"""

import logging
import random
from typing import List, Dict, Any

class CoreGenerator:
    """
    Genera exactamente 4 quinielas Core con variación correcta según metodología
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.concentracion_max_general = self.config["CONCENTRACION_MAX_GENERAL"]
        self.concentracion_max_inicial = self.config["CONCENTRACION_MAX_INICIAL"]
        self.logger.debug(f"CoreGenerator CORREGIDO inicializado")
    
    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("Generando 4 quinielas Core CORREGIDAS con variación...")
        if len(partidos_clasificados) != 14:
            raise ValueError(f"Se requieren exactamente 14 partidos, recibidos: {len(partidos_clasificados)}")
        
        # Identificar tipos de partidos
        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        divisores_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Divisor"]
        otros_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") not in ["Ancla", "Divisor"]]
        
        self.logger.info(f"Partidos: {len(anclas_indices)} Anclas, {len(divisores_indices)} Divisores, {len(otros_indices)} Otros")
        
        # Generar resultados base para Anclas (iguales en todas las Core)
        resultados_anclas = {}
        for i in anclas_indices:
            partido = partidos_clasificados[i]
            resultado_ancla = self._get_resultado_max_prob(partido)
            resultados_anclas[i] = resultado_ancla
            self.logger.debug(f"Ancla P{i+1}: {partido['home']} vs {partido['away']} → {resultado_ancla}")
        
        core_quinielas = []
        for core_id in range(4):
            intentos = 0
            while intentos < 10:
                quiniela_resultados = self._generar_core_individual_con_variacion(
                    partidos_clasificados, core_id, anclas_indices, divisores_indices, otros_indices, resultados_anclas
                )
                
                # Validar que cumple reglas básicas
                if self._es_quiniela_valida(quiniela_resultados):
                    break
                
                self.logger.debug(f"Core-{core_id+1} falló validación, reintentando...")
                intentos += 1
            else:
                self.logger.warning(f"Core-{core_id+1} no pudo validarse tras 10 intentos, usando último resultado")

            core_info = {
                "id": f"Core-{core_id+1}",
                "tipo": "Core",
                "resultados": quiniela_resultados,
                "empates": quiniela_resultados.count("E"),
                "distribución": {
                    "L": quiniela_resultados.count("L"),
                    "E": quiniela_resultados.count("E"),
                    "V": quiniela_resultados.count("V")
                }
            }
            core_quinielas.append(core_info)
            
            # Log para debug
            self.logger.debug(f"Core-{core_id+1}: {''.join(quiniela_resultados)} (E:{core_info['empates']})")
        
        # Validación final
        self._validar_diferencias_entre_cores(core_quinielas, anclas_indices)
        
        self.logger.info(f"✅ Generadas {len(core_quinielas)} quinielas Core con variación correcta")
        return core_quinielas
    
    def _generar_core_individual_con_variacion(self, partidos: List[Dict[str, Any]], core_id: int,
                                             anclas_indices: List[int], divisores_indices: List[int], 
                                             otros_indices: List[int], resultados_anclas: Dict[int, str]) -> List[str]:
        """
        Genera una quiniela Core individual con variación correcta
        """
        quiniela = [""] * 14
        
        # PASO 1: Fijar Anclas (iguales en todas las Core)
        for i in anclas_indices:
            quiniela[i] = resultados_anclas[i]
        
        # PASO 2: Variar Divisores según el core_id
        for j, i in enumerate(divisores_indices):
            partido = partidos[i]
            
            # Estrategia de variación basada en core_id
            if core_id == 0:  # Core-1: Conservador (máxima probabilidad)
                quiniela[i] = self._get_resultado_max_prob(partido)
            elif core_id == 1:  # Core-2: Alternativo 1 (segunda opción)
                quiniela[i] = self._get_resultado_segunda_opcion(partido)
            elif core_id == 2:  # Core-3: Alternativo 2 (empates cuando sea razonable)
                if partido["prob_empate"] > 0.25:  # Si empate es razonable
                    quiniela[i] = "E"
                else:
                    quiniela[i] = self._get_resultado_max_prob(partido)
            else:  # Core-4: Mix basado en posición
                if j % 2 == 0:
                    quiniela[i] = self._get_resultado_max_prob(partido)
                else:
                    quiniela[i] = self._get_resultado_segunda_opcion(partido)
        
        # PASO 3: Completar otros partidos con ligera variación
        for i in otros_indices:
            partido = partidos[i]
            
            # Variación sutil basada en core_id
            probabilidad_cambio = 0.3 if core_id > 0 else 0.1  # Core-1 más conservador
            
            if random.random() < probabilidad_cambio:
                quiniela[i] = self._get_resultado_segunda_opcion(partido)
            else:
                quiniela[i] = self._get_resultado_max_prob(partido)
        
        # PASO 4: Ajustar empates manteniendo variación
        quiniela = self._ajustar_empates_con_variacion(quiniela, partidos, anclas_indices, core_id)
        
        return quiniela
    
    def _ajustar_empates_con_variacion(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                     anclas_indices: List[int], core_id: int) -> List[str]:
        """
        Ajusta empates manteniendo la variación entre cores
        """
        q_ajustada = quiniela.copy()
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        empates_actuales = q_ajustada.count("E")
        
        if empates_actuales < self.empates_min:
            # Necesitamos más empates
            necesarios = self.empates_min - empates_actuales
            
            # Candidatos ordenados por probabilidad de empate (mayor probabilidad primero)
            candidatos = [(i, partidos[i]["prob_empate"]) for i in modificables if q_ajustada[i] != 'E']
            candidatos.sort(key=lambda x: x[1], reverse=True)
            
            # Agregar variación: Core diferentes priorizan diferentes candidatos
            if core_id > 0 and len(candidatos) > necesarios:
                # Rotar la lista de candidatos según core_id para crear variación
                offset = min(core_id, len(candidatos) - necesarios)
                candidatos = candidatos[offset:] + candidatos[:offset]
            
            for i in range(min(necesarios, len(candidatos))):
                idx, _ = candidatos[i]
                q_ajustada[idx] = 'E'
                
        elif empates_actuales > self.empates_max:
            # Demasiados empates
            exceso = empates_actuales - self.empates_max
            
            # Candidatos ordenados por probabilidad de empate (menor probabilidad primero)
            candidatos = [(i, partidos[i]["prob_empate"]) for i in modificables if q_ajustada[i] == 'E']
            candidatos.sort(key=lambda x: x[1])
            
            for i in range(min(exceso, len(candidatos))):
                idx, _ = candidatos[i]
                partido = partidos[idx]
                # Elegir entre L y V basado en probabilidad
                q_ajustada[idx] = 'L' if partido['prob_local'] > partido['prob_visitante'] else 'V'
        
        return q_ajustada
    
    def _es_quiniela_valida(self, resultados: List[str]) -> bool:
        """
        Validación básica de quiniela individual
        """
        empates = resultados.count("E")
        if not (self.empates_min <= empates <= self.empates_max):
            return False
        
        # Concentración general
        max_conc_general = max(resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc_general > self.concentracion_max_general:
            return False
        
        # Concentración inicial
        primeros_3 = resultados[:3]
        max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > self.concentracion_max_inicial:
            return False
        
        return True
    
    def _validar_diferencias_entre_cores(self, core_quinielas: List[Dict[str, Any]], anclas_indices: List[int]):
        """
        Valida que las Core tengan las diferencias correctas
        """
        if len(core_quinielas) < 2:
            return
        
        # Verificar que Anclas son iguales
        primera_core = core_quinielas[0]["resultados"]
        for i, core in enumerate(core_quinielas[1:], 1):
            resultados_actual = core["resultados"]
            
            # Verificar anclas iguales
            for ancla_idx in anclas_indices:
                if primera_core[ancla_idx] != resultados_actual[ancla_idx]:
                    self.logger.error(f"ANCLA DIFERENTE: Core-1[{ancla_idx}]={primera_core[ancla_idx]} != Core-{i+1}[{ancla_idx}]={resultados_actual[ancla_idx]}")
        
        # Verificar que NO son completamente iguales
        for i, core in enumerate(core_quinielas[1:], 1):
            diferencias = sum(1 for j in range(14) if primera_core[j] != core["resultados"][j])
            self.logger.info(f"Core-1 vs Core-{i+1}: {diferencias} diferencias")
            
            if diferencias == 0:
                self.logger.warning(f"⚠️ Core-1 y Core-{i+1} son IDÉNTICAS - esto no debería pasar")
            elif diferencias < 2:
                self.logger.warning(f"⚠️ Core-1 y Core-{i+1} tienen solo {diferencias} diferencias - muy pocas")
    
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        return max(probs, key=probs.get)
    
    def _get_resultado_segunda_opcion(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de segunda mayor probabilidad"""
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        sorted_results = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[1][0]  # Segunda opción