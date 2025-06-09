# progol_optimizer/portfolio/satellite_generator.py
"""
Generador de Satélites - Implementación EXACTA de pares anticorrelados (página 4)
26 satélites en 13 pares con correlación negativa ~-0.35, invirtiendo DIVISORES
"""

import logging
import random
from typing import List, Dict, Any, Tuple

class SatelliteGenerator:
    """
    Genera pares de satélites anticorrelados según especificaciones del documento
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from ..config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.correlacion_max = self.config["ARQUITECTURA_PORTAFOLIO"]["correlacion_jaccard_max"]
        
        self.logger.debug(f"SatelliteGenerator inicializado: correlación_max={self.correlacion_max}")
    
    def generar_pares_satelites(self, partidos_clasificados: List[Dict[str, Any]], num_satelites: int) -> List[Dict[str, Any]]:
        """
        IMPLEMENTACIÓN CRÍTICA: Pares anticorrelados reales
        - Cada par invierte resultados en partidos DIVISOR
        - Mantiene ANCLAS idénticas 
        - Correlación Jaccard ≤ 0.57 entre pares
        
        Args:
            partidos_clasificados: Partidos con clasificación
            num_satelites: Número total de satélites (debe ser par)
            
        Returns:
            List[Dict]: Lista de satélites en pares anticorrelados
        """
        if num_satelites % 2 != 0:
            raise ValueError(f"Número de satélites debe ser par, recibido: {num_satelites}")
        
        num_pares = num_satelites // 2
        
        self.logger.info(f"Generando {num_satelites} satélites en {num_pares} pares anticorrelados...")
        
        # Identificar índices de partidos DIVISOR
        divisores_indices = [
            i for i, partido in enumerate(partidos_clasificados)
            if partido["clasificacion"] == "Divisor"
        ]
        
        if not divisores_indices:
            self.logger.warning("No hay partidos DIVISOR - generando con NEUTROS")
            divisores_indices = [
                i for i, partido in enumerate(partidos_clasificados)
                if partido["clasificacion"] == "Neutro"
            ]
        
        self.logger.debug(f"Divisores disponibles: {len(divisores_indices)} partidos")
        
        satelites = []
        
        # Generar cada par de satélites
        for par_id in range(num_pares):
            # Seleccionar DIVISOR específico para este par (rotatorio)
            divisor_principal = divisores_indices[par_id % len(divisores_indices)]
            
            # Crear par anticorrelado
            quiniela_a, quiniela_b = self._crear_par_anticorrelado(
                partidos_clasificados, divisor_principal, par_id
            )
            
            # Verificar correlación
            correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
            
            satelites.extend([
                {
                    "id": f"Sat-{par_id+1}A",
                    "tipo": "Satelite",
                    "resultados": quiniela_a,
                    "par_id": par_id,
                    "divisor_principal": divisor_principal,
                    "correlacion_jaccard": correlacion,
                    "empates": quiniela_a.count("E"),
                    "distribución": {
                        "L": quiniela_a.count("L"),
                        "E": quiniela_a.count("E"),
                        "V": quiniela_a.count("V")
                    }
                },
                {
                    "id": f"Sat-{par_id+1}B", 
                    "tipo": "Satelite",
                    "resultados": quiniela_b,
                    "par_id": par_id,
                    "divisor_principal": divisor_principal,
                    "correlacion_jaccard": correlacion,
                    "empates": quiniela_b.count("E"),
                    "distribución": {
                        "L": quiniela_b.count("L"),
                        "E": quiniela_b.count("E"),
                        "V": quiniela_b.count("V")
                    }
                }
            ])
            
            self.logger.debug(f"  Par {par_id+1}: divisor={divisor_principal}, correlación={correlacion:.3f}")
        
        # Validar todos los pares
        self._validar_satelites(satelites, partidos_clasificados)
        
        self.logger.info(f"✅ Generados {len(satelites)} satélites en {num_pares} pares válidos")
        return satelites
    
    def _crear_par_anticorrelado(self, partidos: List[Dict[str, Any]], divisor_principal: int, par_id: int) -> Tuple[List[str], List[str]]:
        """
        Crea par con anticorrelación REAL en DIVISOR específico
        """
        quiniela_a = []
        quiniela_b = []
        
        for i, partido in enumerate(partidos):
            clasificacion = partido["clasificacion"]
            
            if clasificacion == "Ancla":
                # IDÉNTICO en ambas quinielas del par
                resultado = self._get_resultado_max_prob(partido)
                quiniela_a.append(resultado)
                quiniela_b.append(resultado)
                
            elif i == divisor_principal:
                # ANTICORRELACIÓN: Invertir resultado en DIVISOR
                resultado_a = self._get_resultado_max_prob(partido)
                resultado_b = self._get_resultado_alternativo(partido)
                
                quiniela_a.append(resultado_a)
                quiniela_b.append(resultado_b)
                
                self.logger.debug(f"    Anticorrelación en partido {i}: {resultado_a} vs {resultado_b}")
                
            else:
                # Variación controlada en otros partidos
                if random.random() < 0.3:  # 30% de probabilidad de diferencia
                    resultado_a = self._get_resultado_max_prob(partido)
                    resultado_b = self._get_resultado_alternativo(partido)
                else:
                    # Misma elección en ambas
                    resultado = self._get_resultado_max_prob(partido)
                    resultado_a = resultado_b = resultado
                
                quiniela_a.append(resultado_a)
                quiniela_b.append(resultado_b)
        
        # Ajustar empates manteniendo anticorrelación
        quiniela_a = self._ajustar_empates_satelite(quiniela_a, partidos)
        quiniela_b = self._ajustar_empates_satelite(quiniela_b, partidos)
        
        return quiniela_a, quiniela_b
    
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        return max(probs, key=probs.get)
    
    def _get_resultado_alternativo(self, partido: Dict[str, Any]) -> str:
        """Obtiene resultado alternativo para crear anticorrelación"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        
        # Ordenar por probabilidad descendente
        probs_ordenadas = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # Estrategia: si primera es L/V, usar la otra; si primera es E, usar segunda opción
        primera = probs_ordenadas[0][0]
        
        if primera == "L":
            return "V"  # Intercambio directo L<->V
        elif primera == "V":
            return "L"  # Intercambio directo V<->L
        else:  # primera == "E"
            # Si empate es máximo, usar segunda opción
            return probs_ordenadas[1][0] if len(probs_ordenadas) > 1 else "L"
    
    def _ajustar_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """
        Ajusta empates en satélite manteniendo la lógica de anticorrelación
        """
        empates_actuales = quiniela.count("E")
        
        # Si está en rango, no hacer nada
        if self.empates_min <= empates_actuales <= self.empates_max:
            return quiniela
        
        quiniela_ajustada = quiniela.copy()
        
        # Ajustes similares a Core pero evitando tocar ANCLAS
        if empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            self._reducir_empates_satelite(quiniela_ajustada, partidos, exceso)
        elif empates_actuales < self.empates_min:
            faltante = self.empates_min - empates_actuales
            self._aumentar_empates_satelite(quiniela_ajustada, partidos, faltante)
        
        return quiniela_ajustada
    
    def _reducir_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]], reducir: int):
        """Reduce empates evitando ANCLAS"""
        candidatos = [(i, partidos[i]["prob_empate"]) 
                     for i, res in enumerate(quiniela) 
                     if res == "E" and partidos[i]["clasificacion"] != "Ancla"]
        
        candidatos.sort(key=lambda x: x[1])  # Menor probabilidad primero
        
        for i in range(min(reducir, len(candidatos))):
            idx = candidatos[i][0]
            partido = partidos[idx]
            
            if partido["prob_local"] > partido["prob_visitante"]:
                quiniela[idx] = "L"
            else:
                quiniela[idx] = "V"
    
    def _aumentar_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]], aumentar: int):
        """Aumenta empates evitando ANCLAS"""
        candidatos = [(i, partidos[i]["prob_empate"]) 
                     for i, res in enumerate(quiniela) 
                     if res in ["L", "V"] and partidos[i]["clasificacion"] != "Ancla"]
        
        candidatos.sort(key=lambda x: x[1], reverse=True)  # Mayor probabilidad primero
        
        for i in range(min(aumentar, len(candidatos))):
            idx = candidatos[i][0]
            quiniela[idx] = "E"
    
    def _calcular_correlacion_jaccard(self, quiniela_a: List[str], quiniela_b: List[str]) -> float:
        """
        Calcula correlación de Jaccard entre dos quinielas
        Jaccard = |intersección| / |unión|
        """
        if len(quiniela_a) != len(quiniela_b):
            return 0.0
        
        coincidencias = sum(1 for a, b in zip(quiniela_a, quiniela_b) if a == b)
        jaccard = coincidencias / len(quiniela_a)
        
        return jaccard
    
    def _validar_satelites(self, satelites: List[Dict[str, Any]], partidos: List[Dict[str, Any]]):
        """
        Valida que todos los satélites cumplan las reglas
        """
        self.logger.debug("Validando satélites...")
        
        # Validar cada satélite individual
        for satelite in satelites:
            quiniela = satelite["resultados"]
            empates = quiniela.count("E")
            
            # Validar rango de empates
            if not (self.empates_min <= empates <= self.empates_max):
                raise ValueError(f"{satelite['id']}: empates {empates} fuera del rango")
            
            # Validar longitud
            if len(quiniela) != 14:
                raise ValueError(f"{satelite['id']}: longitud incorrecta")
        
        # Validar correlaciones por pares
        pares = {}
        for satelite in satelites:
            par_id = satelite["par_id"]
            if par_id not in pares:
                pares[par_id] = []
            pares[par_id].append(satelite)
        
        for par_id, par_satelites in pares.items():
            if len(par_satelites) != 2:
                raise ValueError(f"Par {par_id}: debe tener exactamente 2 satélites")
            
            correlacion = par_satelites[0]["correlacion_jaccard"]
            if correlacion > self.correlacion_max:
                raise ValueError(f"Par {par_id}: correlación {correlacion:.3f} > {self.correlacion_max}")
        
        self.logger.debug("✅ Todos los satélites son válidos")
