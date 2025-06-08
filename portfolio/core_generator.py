# progol_optimizer/portfolio/core_generator.py
"""
Generador de Quinielas Core - Implementación EXACTA de la página 4
4 quinielas Core que fijan resultados en partidos ANCLA (>60% probabilidad)
"""

import logging
import random
from typing import List, Dict, Any

class CoreGenerator:
    """
    Genera exactamente 4 quinielas Core según especificaciones del documento
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from progol_optimizer.config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        
        self.logger.debug(f"CoreGenerator inicializado: empates [{self.empates_min}-{self.empates_max}]")
    
    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera EXACTAMENTE 4 quinielas Core según página 4:
        - Fijan resultado de máxima probabilidad en partidos Ancla (>60%)
        - Variación controlada en Divisores y Neutros
        - 4-6 empates obligatorios por quiniela
        - Core deben ser IDÉNTICAS en ANCLAS
        
        Args:
            partidos_clasificados: Lista de partidos con clasificación
            
        Returns:
            List[Dict]: 4 quinielas Core
        """
        self.logger.info("Generando 4 quinielas Core...")
        
        # Verificar que tenemos exactamente 14 partidos
        if len(partidos_clasificados) != 14:
            raise ValueError(f"Se requieren exactamente 14 partidos, recibidos: {len(partidos_clasificados)}")
        
        core_quinielas = []
        
        # Generar cada una de las 4 Core
        for i in range(4):
            self.logger.debug(f"Generando Core-{i+1}")
            
            quiniela = self._generar_core_individual(partidos_clasificados, i)
            
            core_info = {
                "id": f"Core-{i+1}",
                "tipo": "Core",
                "resultados": quiniela,
                "empates": quiniela.count("E"),
                "distribución": {
                    "L": quiniela.count("L"),
                    "E": quiniela.count("E"), 
                    "V": quiniela.count("V")
                }
            }
            
            core_quinielas.append(core_info)
            
            self.logger.debug(f"  Core-{i+1}: L={core_info['distribución']['L']}, "
                             f"E={core_info['distribución']['E']}, V={core_info['distribución']['V']}")
        
        # Validar que todas las Core son válidas
        self._validar_quinielas_core(core_quinielas, partidos_clasificados)
        
        self.logger.info(f"✅ Generadas {len(core_quinielas)} quinielas Core válidas")
        return core_quinielas
    
    def _generar_core_individual(self, partidos: List[Dict[str, Any]], core_index: int) -> List[str]:
        """
        Genera una quiniela Core individual siguiendo las reglas exactas
        """
        quiniela = []
        empates_actuales = 0
        
        # FASE 1: Asignar resultados según clasificación
        for partido in partidos:
            clasificacion = partido["clasificacion"]
            
            if clasificacion == "Ancla":
                # OBLIGATORIO: Fijar resultado de máxima probabilidad
                resultado = self._get_resultado_max_prob(partido)
                self.logger.debug(f"    Ancla {partido['home']} vs {partido['away']}: {resultado}")
                
            elif clasificacion == "TendenciaEmpate":
                # Priorizar empate si no excede límite
                if empates_actuales < self.empates_max:
                    resultado = "E"
                    empates_actuales += 1
                    self.logger.debug(f"    TendenciaEmpate forzada a E: {empates_actuales} empates")
                else:
                    resultado = self._get_resultado_max_prob(partido)
                    self.logger.debug(f"    TendenciaEmpate -> max_prob (empates llenos)")
                    
            else:  # Divisor o Neutro
                # Core-1 siempre usa máxima probabilidad
                # Core 2-4 introducen variación mínima controlada
                if core_index == 0:
                    resultado = self._get_resultado_max_prob(partido)
                else:
                    # 20% de probabilidad de variación en Core 2-4
                    if random.random() < 0.2:
                        resultado = self._get_resultado_alternativo(partido)
                        self.logger.debug(f"    Variación Core-{core_index+1}: alternativo")
                    else:
                        resultado = self._get_resultado_max_prob(partido)
            
            if resultado == "E":
                empates_actuales += 1
            
            quiniela.append(resultado)
        
        # FASE 2: Ajustar empates para cumplir rango 4-6
        quiniela_ajustada = self._ajustar_empates_core(quiniela, partidos)
        
        return quiniela_ajustada
    
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """
        Obtiene el resultado de máxima probabilidad
        """
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        
        resultado_max = max(probs, key=probs.get)
        return resultado_max
    
    def _get_resultado_alternativo(self, partido: Dict[str, Any]) -> str:
        """
        Obtiene un resultado alternativo (segunda mayor probabilidad)
        """
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        
        # Ordenar por probabilidad descendente
        probs_ordenadas = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # Seleccionar segunda opción si la diferencia no es muy grande
        if len(probs_ordenadas) >= 2 and probs_ordenadas[1][1] > 0.15:
            return probs_ordenadas[1][0]
        else:
            # Si segunda opción es muy baja, usar primera
            return probs_ordenadas[0][0]
    
    def _ajustar_empates_core(self, quiniela: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """
        Ajusta la quiniela para cumplir exactamente el rango de empates 4-6
        """
        empates_actuales = quiniela.count("E")
        
        self.logger.debug(f"    Ajustando empates: actuales={empates_actuales}, rango=[{self.empates_min}-{self.empates_max}]")
        
        # Si está en rango, no hacer nada
        if self.empates_min <= empates_actuales <= self.empates_max:
            return quiniela
        
        quiniela_ajustada = quiniela.copy()
        
        # Demasiados empates: convertir algunos a L/V
        if empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            self._reducir_empates(quiniela_ajustada, partidos, exceso)
        
        # Muy pocos empates: convertir algunos L/V a E
        elif empates_actuales < self.empates_min:
            faltante = self.empates_min - empates_actuales
            self._aumentar_empates(quiniela_ajustada, partidos, faltante)
        
        empates_final = quiniela_ajustada.count("E")
        self.logger.debug(f"    Empates finales: {empates_final}")
        
        return quiniela_ajustada
    
    def _reducir_empates(self, quiniela: List[str], partidos: List[Dict[str, Any]], reducir: int):
        """
        Reduce el número de empates convirtiendo los de menor probabilidad a L/V
        """
        # Encontrar empates con menor probabilidad de empate
        empates_indices = [(i, partidos[i]["prob_empate"]) 
                          for i, res in enumerate(quiniela) 
                          if res == "E" and partidos[i]["clasificacion"] != "Ancla"]
        
        # Ordenar por probabilidad ascendente (cambiar los menos probables)
        empates_indices.sort(key=lambda x: x[1])
        
        # Cambiar los primeros 'reducir' empates
        for i in range(min(reducir, len(empates_indices))):
            idx = empates_indices[i][0]
            partido = partidos[idx]
            
            # Cambiar a L o V según cual tenga mayor probabilidad
            if partido["prob_local"] > partido["prob_visitante"]:
                quiniela[idx] = "L"
            else:
                quiniela[idx] = "V"
            
            self.logger.debug(f"      Empate -> {quiniela[idx]} en partido {idx}")
    
    def _aumentar_empates(self, quiniela: List[str], partidos: List[Dict[str, Any]], aumentar: int):
        """
        Aumenta el número de empates convirtiendo L/V con alta probabilidad de empate
        """
        # Encontrar L/V con alta probabilidad de empate
        candidatos = [(i, partidos[i]["prob_empate"]) 
                     for i, res in enumerate(quiniela) 
                     if res in ["L", "V"] and partidos[i]["clasificacion"] != "Ancla"]
        
        # Ordenar por probabilidad descendente (cambiar los más probables)
        candidatos.sort(key=lambda x: x[1], reverse=True)
        
        # Cambiar los primeros 'aumentar' candidatos
        for i in range(min(aumentar, len(candidatos))):
            idx = candidatos[i][0]
            quiniela[idx] = "E"
            self.logger.debug(f"      {quiniela[idx]} -> E en partido {idx}")
    
    def _validar_quinielas_core(self, core_quinielas: List[Dict[str, Any]], partidos: List[Dict[str, Any]]):
        """
        Valida que las 4 quinielas Core cumplan todas las reglas
        """
        self.logger.debug("Validando quinielas Core...")
        
        # Validar cada Core individual
        for core in core_quinielas:
            quiniela = core["resultados"]
            empates = quiniela.count("E")
            
            # Validar rango de empates
            if not (self.empates_min <= empates <= self.empates_max):
                raise ValueError(f"{core['id']}: empates {empates} fuera del rango [{self.empates_min}-{self.empates_max}]")
            
            # Validar longitud
            if len(quiniela) != 14:
                raise ValueError(f"{core['id']}: longitud {len(quiniela)} != 14")
        
        # Validar que ANCLAS son idénticas entre todas las Core
        anclas_core1 = []
        for i, partido in enumerate(partidos):
            if partido["clasificacion"] == "Ancla":
                anclas_core1.append((i, core_quinielas[0]["resultados"][i]))
        
        for core in core_quinielas[1:]:
            for idx, resultado_esperado in anclas_core1:
                resultado_actual = core["resultados"][idx]
                if resultado_actual != resultado_esperado:
                    raise ValueError(f"{core['id']}: Ancla en posición {idx} difiere de Core-1 "
                                   f"({resultado_actual} vs {resultado_esperado})")
        
        self.logger.debug("✅ Todas las quinielas Core son válidas")