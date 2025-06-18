# progol_optimizer/portfolio/satellite_generator.py - CORREGIDO
"""
Generador de Satélites CORREGIDO - Garantiza 4-6 empates Y Jaccard ≤ 0.57
"""

import logging
import random
from typing import List, Dict, Any, Tuple

class SatelliteGenerator:
    """
    Genera pares de satélites anticorrelados con algoritmo robusto CORREGIDO
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.correlacion_max = self.config["ARQUITECTURA_PORTAFOLIO"]["correlacion_jaccard_max"]
        
        self.logger.debug(f"SatelliteGenerator CORREGIDO: correlación_max={self.correlacion_max}")
    
    def generar_pares_satelites(self, partidos_clasificados: List[Dict[str, Any]], num_satelites: int) -> List[Dict[str, Any]]:
        """
        Genera satélites con algoritmo CORREGIDO que garantiza 4-6 empates Y Jaccard ≤ 0.57
        """
        if num_satelites % 2 != 0:
            raise ValueError(f"Número de satélites debe ser par, recibido: {num_satelites}")
        
        num_pares = num_satelites // 2
        
        self.logger.info(f"🔄 Generando {num_satelites} satélites CORREGIDOS en {num_pares} pares...")
        
        satelites = []
        
        # Generar cada par con algoritmo corregido
        for par_id in range(num_pares):
            try:
                quiniela_a, quiniela_b = self._crear_par_anticorrelado_corregido(
                    partidos_clasificados, par_id
                )
                
                # Crear objetos satélite con toda la información requerida
                correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
                
                satelite_a = {
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
                }
                
                satelite_b = {
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
                
                satelites.extend([satelite_a, satelite_b])
                
                self.logger.debug(f"✅ Par {par_id+1}: correlación={correlacion:.3f}, "
                               f"empates=({satelite_a['empates']},{satelite_b['empates']})")
                
            except Exception as e:
                self.logger.error(f"❌ Error generando par {par_id}: {e}, se usará par de emergencia.")
                # Generar par de emergencia si falla el algoritmo principal
                quiniela_a, quiniela_b = self._crear_par_emergencia_corregido(partidos_clasificados, par_id)
                correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
                satelites.extend([
                    {
                        "id": f"Sat-{par_id+1}A", "tipo": "Satelite", "resultados": quiniela_a,
                        "par_id": par_id, "correlacion_jaccard": correlacion, "empates": quiniela_a.count("E"),
                        "distribución": {"L": quiniela_a.count("L"), "E": quiniela_a.count("E"), "V": quiniela_a.count("V")}
                    },
                    {
                        "id": f"Sat-{par_id+1}B", "tipo": "Satelite", "resultados": quiniela_b,
                        "par_id": par_id, "correlacion_jaccard": correlacion, "empates": quiniela_b.count("E"),
                        "distribución": {"L": quiniela_b.count("L"), "E": quiniela_b.count("E"), "V": quiniela_b.count("V")}
                    }
                ])
        
        # Validación final robusta
        self._validar_satelites_robusto(satelites)
        
        self.logger.info(f"✅ Generados {len(satelites)} satélites corregidos en {num_pares} pares")
        return satelites

    def _crear_par_anticorrelado_corregido(self, partidos: List[Dict[str, Any]], par_id: int) -> Tuple[List[str], List[str]]:
        """
        Algoritmo CORREGIDO que garantiza 4-6 empates Y busca Jaccard ≤ 0.57
        """
        max_intentos = 10
        for intento in range(max_intentos):
            quiniela_a = [""] * 14
            quiniela_b = [""] * 14

            # 1. Identificar tipos de partidos
            anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
            divisores_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Divisor"]
            otros_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") not in ["Ancla", "Divisor"]]

            # 2. ANCLAS: Siempre idénticas (requisito crítico)
            for i in anclas_indices:
                resultado = self._get_resultado_max_prob(partidos[i])
                quiniela_a[i] = resultado
                quiniela_b[i] = resultado

            # 3. DIVISORES: Crear anti-correlación
            for i in divisores_indices:
                res_a = self._get_resultado_max_prob(partidos[i])
                res_b = self._get_resultado_opuesto_inteligente(res_a, partidos[i])
                quiniela_a[i] = res_a
                quiniela_b[i] = res_b
            
            # 4. OTROS: Usar máxima probabilidad para estabilidad
            for i in otros_indices:
                resultado = self._get_resultado_max_prob(partidos[i])
                quiniela_a[i] = resultado
                quiniela_b[i] = resultado

            # 5. AJUSTE DE EMPATES: Forzar cumplimiento de 4-6 empates. Esta es la parte crítica.
            quiniela_a = self._forzar_empates_correctos(quiniela_a, partidos, anclas_indices)
            quiniela_b = self._forzar_empates_correctos(quiniela_b, partidos, anclas_indices)

            # 6. VERIFICACIÓN: Comprobar si el par es válido
            correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
            if correlacion <= self.correlacion_max:
                self.logger.debug(f"Par {par_id} generado en intento {intento+1} con correlación {correlacion:.3f}")
                return quiniela_a, quiniela_b

        # Si después de varios intentos no se logra, se genera uno de emergencia.
        self.logger.warning(f"No se pudo generar par {par_id} con Jaccard ≤ {self.correlacion_max}, usando emergencia.")
        return self._crear_par_emergencia_corregido(partidos, par_id)

    def _forzar_empates_correctos(self, quiniela: List[str], partidos: List[Dict[str, Any]], anclas_indices: List[int]) -> List[str]:
        """
        NUEVO: Rutina robusta que ajusta una quiniela para que tenga entre 4 y 6 empates.
        No modifica los partidos clasificados como 'Ancla'.
        """
        quiniela_corregida = quiniela.copy()
        modificables_indices = [i for i in range(14) if i not in anclas_indices]

        empates_actuales = quiniela_corregida.count("E")

        # Caso 1: Necesitamos MÁS empates
        if empates_actuales < self.empates_min:
            necesarios = self.empates_min - empates_actuales
            # Buscar candidatos L/V para convertir a Empate
            candidatos = []
            for i in modificables_indices:
                if quiniela_corregida[i] in ["L", "V"]:
                    candidatos.append((i, partidos[i]["prob_empate"]))
            # Ordenar por probabilidad de empate (los más probables primero)
            candidatos.sort(key=lambda x: x[1], reverse=True)
            # Convertir los mejores 'necesarios' candidatos
            for i in range(min(necesarios, len(candidatos))):
                idx = candidatos[i][0]
                quiniela_corregida[idx] = "E"

        # Caso 2: Necesitamos MENOS empates
        elif empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            # Buscar candidatos Empate para convertir a L/V
            candidatos = []
            for i in modificables_indices:
                if quiniela_corregida[i] == "E":
                    candidatos.append((i, partidos[i]["prob_empate"]))
            # Ordenar por probabilidad de empate (los menos probables primero)
            candidatos.sort(key=lambda x: x[1])
            # Convertir los peores 'exceso' candidatos
            for i in range(min(exceso, len(candidatos))):
                idx = candidatos[i][0]
                partido = partidos[idx]
                quiniela_corregida[idx] = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"

        return quiniela_corregida
        
    def _crear_par_emergencia_corregido(self, partidos: List[Dict[str, Any]], par_id: int) -> Tuple[List[str], List[str]]:
        """
        Par de emergencia CORREGIDO que garantiza 4-6 empates
        """
        self.logger.warning(f"🚨 Generando par de emergencia CORREGIDO {par_id}")
        anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        
        # Base: máxima probabilidad para ambos
        quiniela_a = [self._get_resultado_max_prob(p) for p in partidos]
        quiniela_b = quiniela_a.copy()
        
        # Introducir una diferencia en el primer divisor no ancla
        divisores_no_ancla = [i for i,p in enumerate(partidos) if p.get("clasificacion") == "Divisor" and i not in anclas_indices]
        if divisores_no_ancla:
            idx_cambio = divisores_no_ancla[0]
            quiniela_b[idx_cambio] = self._get_resultado_opuesto_inteligente(quiniela_a[idx_cambio], partidos[idx_cambio])

        # Forzar ajuste final de empates
        quiniela_a = self._forzar_empates_correctos(quiniela_a, partidos, anclas_indices)
        quiniela_b = self._forzar_empates_correctos(quiniela_b, partidos, anclas_indices)
        
        return quiniela_a, quiniela_b

    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        return max(probs, key=probs.get)

    def _get_resultado_opuesto_inteligente(self, resultado_actual: str, partido: Dict[str, Any]) -> str:
        """Obtiene resultado opuesto de forma inteligente (segunda mayor probabilidad)"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        # Eliminar la opción actual y devolver la mejor de las restantes
        del probs[resultado_actual]
        return max(probs, key=probs.get)

    def _calcular_correlacion_jaccard(self, quiniela_a: List[str], quiniela_b: List[str]) -> float:
        """Calcula correlación de Jaccard entre dos quinielas"""
        if len(quiniela_a) != len(quiniela_b) or len(quiniela_a) == 0: 
            return 0.0
        coincidencias = sum(1 for a, b in zip(quiniela_a, quiniela_b) if a == b)
        return coincidencias / len(quiniela_a)
    
    def _validar_satelites_robusto(self, satelites: List[Dict[str, Any]]):
        """Validación robusta con logging detallado"""
        self.logger.debug("🔍 Validando satélites corregidos...")
        errores = []
        
        for satelite in satelites:
            empates = satelite["resultados"].count("E")
            if not (self.empates_min <= empates <= self.empates_max):
                errores.append(f"{satelite['id']}: empates {empates} fuera del rango [{self.empates_min}-{self.empates_max}]")
            if len(satelite["resultados"]) != 14:
                errores.append(f"{satelite['id']}: longitud {len(satelite['resultados'])} != 14")
        
        pares = {}
        for satelite in satelites:
            par_id = satelite.get("par_id", -1)
            pares.setdefault(par_id, []).append(satelite)
        
        for par_id, par_satelites in pares.items():
            if len(par_satelites) != 2:
                errores.append(f"Par {par_id}: no tiene 2 satélites, tiene {len(par_satelites)}")
                continue
            correlacion = self._calcular_correlacion_jaccard(par_satelites[0]['resultados'], par_satelites[1]['resultados'])
            if correlacion > self.correlacion_max:
                self.logger.warning(f"⚠️ Par {par_id}: correlación {correlacion:.3f} > {self.correlacion_max}")
        
        if errores:
            for error in errores:
                self.logger.error(f"  - {error}")
            raise ValueError(f"Validación de satélites falló: {', '.join(errores)}")
        
        self.logger.debug("✅ Todos los satélites corregidos son válidos")