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
        from config.constants import PROGOL_CONFIG
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
        
        satelites = []
        
        # Generar cada par de satélites
        for par_id in range(num_pares):
            # Crear par anticorrelado
            quiniela_a, quiniela_b = self._crear_par_anticorrelado(
                partidos_clasificados, par_id
            )
            
            # Verificar correlación
            correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
            
            satelites.extend([
                {
                    "id": f"Sat-{par_id+1}A",
                    "tipo": "Satelite",
                    "resultados": quiniela_a,
                    "par_id": par_id,
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
                    "correlacion_jaccard": correlacion,
                    "empates": quiniela_b.count("E"),
                    "distribución": {
                        "L": quiniela_b.count("L"),
                        "E": quiniela_b.count("E"),
                        "V": quiniela_b.count("V")
                    }
                }
            ])
            
            self.logger.debug(f"  Par {par_id+1}: correlación={correlacion:.3f}")
        
        # Validar todos los pares
        self._validar_satelites(satelites)
        
        self.logger.info(f"✅ Generados {len(satelites)} satélites en {num_pares} pares válidos")
        return satelites

    # --- INICIO DE CORRECCIÓN: Lógica mejorada para crear pares ---
    def _crear_par_anticorrelado(self, partidos: List[Dict[str, Any]], par_id: int) -> Tuple[List[str], List[str]]:
        """
        Crea par anticorrelado de forma robusta.
        - Invierte TODOS los partidos Divisor.
        - Si es necesario, invierte partidos Neutros para garantizar Jaccard <= 0.57 (máx 7 coincidencias).
        - Mantiene ANCLAS idénticas.
        """
        quiniela_a = [""] * 14
        quiniela_b = [""] * 14

        # 1. Clasificar índices de partidos
        anclas_indices = [i for i, p in enumerate(partidos) if p["clasificacion"] == "Ancla"]
        divisores_indices = [i for i, p in enumerate(partidos) if p["clasificacion"] == "Divisor"]
        neutros_indices = [i for i, p in enumerate(partidos) if p["clasificacion"] not in ["Ancla", "Divisor"]]

        # 2. Asignar ANCLAS (idénticas)
        for i in anclas_indices:
            resultado = self._get_resultado_max_prob(partidos[i])
            quiniela_a[i] = resultado
            quiniela_b[i] = resultado

        # 3. Asignar DIVISORES (anticorrelados)
        for i in divisores_indices:
            resultado_a = self._get_resultado_max_prob(partidos[i])
            resultado_b = self._get_resultado_alternativo(partidos[i])
            quiniela_a[i] = resultado_a
            quiniela_b[i] = resultado_b

        # 4. Determinar si se necesitan más diferencias en NEUTROS
        num_diferencias_actuales = len(divisores_indices)
        # Para Jaccard <= 0.57, se necesitan al menos 7 diferencias (14 - 7 = 7 coincidencias; 7/14 = 0.5)
        diferencias_necesarias = 7 
        diferencias_faltantes = max(0, diferencias_necesarias - num_diferencias_actuales)

        # Seleccionar neutros para invertir de forma aleatoria pero determinista para el par
        random.seed(par_id)
        neutros_a_invertir = random.sample(neutros_indices, min(diferencias_faltantes, len(neutros_indices)))
        
        self.logger.debug(f"    Par {par_id}: {num_diferencias_actuales} difs de Divisores. Invirtiendo {len(neutros_a_invertir)} Neutros.")

        # 5. Asignar NEUTROS
        for i in neutros_indices:
            if i in neutros_a_invertir:
                # Invertir para crear más diferencias
                resultado_a = self._get_resultado_max_prob(partidos[i])
                resultado_b = self._get_resultado_alternativo(partidos[i])
                quiniela_a[i] = resultado_a
                quiniela_b[i] = resultado_b
            else:
                # Mantener idénticos
                resultado = self._get_resultado_max_prob(partidos[i])
                quiniela_a[i] = resultado
                quiniela_b[i] = resultado

        # 6. Ajustar empates al final para ambas quinielas
        quiniela_a = self._ajustar_empates_satelite(quiniela_a, partidos)
        quiniela_b = self._ajustar_empates_satelite(quiniela_b, partidos)
        
        return quiniela_a, quiniela_b
    # --- FIN DE CORRECCIÓN ---
    
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
        
        probs_ordenadas = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        primera = probs_ordenadas[0][0]
        
        if primera == "L": return "V"
        if primera == "V": return "L"
        return probs_ordenadas[1][0] if len(probs_ordenadas) > 1 else "L"
    
    def _ajustar_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """
        Ajusta empates en satélite manteniendo la lógica de anticorrelación
        """
        empates_actuales = quiniela.count("E")
        
        if self.empates_min <= empates_actuales <= self.empates_max:
            return quiniela
        
        quiniela_ajustada = quiniela.copy()
        
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
        
        candidatos.sort(key=lambda x: x[1])
        
        for i in range(min(reducir, len(candidatos))):
            idx = candidatos[i][0]
            partido = partidos[idx]
            quiniela[idx] = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"
    
    def _aumentar_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]], aumentar: int):
        """Aumenta empates evitando ANCLAS"""
        candidatos = [(i, partidos[i]["prob_empate"]) 
                     for i, res in enumerate(quiniela) 
                     if res in ["L", "V"] and partidos[i]["clasificacion"] != "Ancla"]
        
        candidatos.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(min(aumentar, len(candidatos))):
            idx = candidatos[i][0]
            quiniela[idx] = "E"
    
    def _calcular_correlacion_jaccard(self, quiniela_a: List[str], quiniela_b: List[str]) -> float:
        """
        Calcula correlación de Jaccard entre dos quinielas
        """
        if len(quiniela_a) != len(quiniela_b): return 0.0
        coincidencias = sum(1 for a, b in zip(quiniela_a, quiniela_b) if a == b)
        return coincidencias / len(quiniela_a)
    
    def _validar_satelites(self, satelites: List[Dict[str, Any]]):
        """
        Valida que todos los satélites cumplan las reglas
        """
        self.logger.debug("Validando satélites...")
        
        for satelite in satelites:
            empates = satelite["resultados"].count("E")
            if not (self.empates_min <= empates <= self.empates_max):
                raise ValueError(f"{satelite['id']}: empates {empates} fuera del rango")
            if len(satelite["resultados"]) != 14:
                raise ValueError(f"{satelite['id']}: longitud incorrecta")
        
        pares = {}
        for satelite in satelites:
            par_id = satelite["par_id"]
            pares.setdefault(par_id, []).append(satelite)
        
        for par_id, par_satelites in pares.items():
            if len(par_satelites) != 2:
                raise ValueError(f"Par {par_id}: debe tener exactamente 2 satélites")
            
            # Recalcula la correlación final para ser extra seguro
            correlacion_final = self._calcular_correlacion_jaccard(par_satelites[0]['resultados'], par_satelites[1]['resultados'])
            if correlacion_final > self.correlacion_max:
                raise ValueError(f"Par {par_id}: correlación {correlacion_final:.3f} > {self.correlacion_max}")
        
        self.logger.debug("✅ Todos los satélites son válidos")
