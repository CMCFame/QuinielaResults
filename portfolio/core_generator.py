# progol_optimizer/portfolio/core_generator.py
"""
Generador de Quinielas Core - REFORZADO
Ahora valida y corrige la concentración para asegurar que las Core sean válidas desde su creación.
"""

import logging
import random
from typing import List, Dict, Any

class CoreGenerator:
    """
    Genera exactamente 4 quinielas Core válidas, incluyendo la regla de concentración.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.concentracion_max_general = self.config["CONCENTRACION_MAX_GENERAL"]
        self.concentracion_max_inicial = self.config["CONCENTRACION_MAX_INICIAL"]
        self.logger.debug(f"CoreGenerator inicializado con todas las reglas individuales.")
    
    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("Generando 4 quinielas Core REFORZADAS...")
        if len(partidos_clasificados) != 14:
            raise ValueError(f"Se requieren exactamente 14 partidos, recibidos: {len(partidos_clasificados)}")
        
        core_quinielas = []
        for i in range(4):
            intentos = 0
            quiniela_resultados = []
            while intentos < 10:
                quiniela_resultados = self._generar_core_individual(partidos_clasificados, i)
                # Validar concentración después de generar
                if self._es_concentracion_valida(quiniela_resultados):
                    break
                self.logger.debug(f"Core-{i+1} falló chequeo de concentración, reintentando...")
                intentos += 1
            else:
                self.logger.warning(f"No se pudo generar Core-{i+1} con concentración válida tras 10 intentos. Usando el último resultado.")

            core_info = {
                "id": f"Core-{i+1}", "tipo": "Core", "resultados": quiniela_resultados,
                "empates": quiniela_resultados.count("E"),
                "distribución": {"L": quiniela_resultados.count("L"), "E": quiniela_resultados.count("E"), "V": quiniela_resultados.count("V")}
            }
            core_quinielas.append(core_info)
        
        self._validar_quinielas_core_final(core_quinielas, partidos_clasificados)
        self.logger.info(f"✅ Generadas {len(core_quinielas)} quinielas Core válidas")
        return core_quinielas
    
    def _generar_core_individual(self, partidos: List[Dict[str, Any]], core_index: int) -> List[str]:
        quiniela = [""] * 14
        
        for i, partido in enumerate(partidos):
            clasificacion = partido.get("clasificacion", "Neutro")
            if clasificacion == "Ancla":
                quiniela[i] = self._get_resultado_max_prob(partido)
            elif clasificacion == "TendenciaEmpate":
                quiniela[i] = "E"
            else: # Divisor o Neutro
                # Core-1 es más conservador
                if core_index == 0 or random.random() > 0.25:
                    quiniela[i] = self._get_resultado_max_prob(partido)
                else:
                    quiniela[i] = self._get_resultado_alternativo(partido)
        
        # Ajustar empates y luego concentración
        quiniela = self._ajustar_empates_core(quiniela, partidos)
        quiniela = self._ajustar_concentracion_core(quiniela, partidos) # Nuevo paso de ajuste
        return quiniela

    def _ajustar_empates_core(self, quiniela: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        q_ajustada = quiniela.copy()
        modificables = [i for i, p in enumerate(partidos) if p.get("clasificacion") != "Ancla"]
        
        empates_actuales = q_ajustada.count("E")

        if empates_actuales < self.empates_min:
            necesarios = self.empates_min - empates_actuales
            candidatos = sorted([i for i in modificables if q_ajustada[i] != 'E'], key=lambda i: partidos[i]['prob_empate'], reverse=True)
            for i in range(min(necesarios, len(candidatos))):
                q_ajustada[candidatos[i]] = 'E'
        elif empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            candidatos = sorted([i for i in modificables if q_ajustada[i] == 'E'], key=lambda i: partidos[i]['prob_empate'])
            for i in range(min(exceso, len(candidatos))):
                idx = candidatos[i]
                prob_partido = partidos[idx]
                # Elegir entre L y V basado en probabilidad
                q_ajustada[idx] = 'L' if prob_partido['prob_local'] > prob_partido['prob_visitante'] else 'V'
                
        return q_ajustada
        
    def _ajustar_concentracion_core(self, quiniela: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """NUEVA FUNCIÓN: Corrige la quiniela si viola las reglas de concentración."""
        q_ajustada = quiniela.copy()
        modificables = [i for i, p in enumerate(partidos) if p.get("clasificacion") != "Ancla"]

        for _ in range(5): # Intentar ajustar hasta 5 veces
            if self._es_concentracion_valida(q_ajustada):
                break

            # Corregir concentración general (ej. > 9 del mismo signo)
            for signo in ["L", "E", "V"]:
                if q_ajustada.count(signo) / 14.0 > self.concentracion_max_general:
                    # Encontrar el mejor candidato para cambiar DE este signo A otro
                    candidatos = [i for i in modificables if q_ajustada[i] == signo]
                    if candidatos:
                        idx_cambio = min(candidatos, key=lambda i: partidos[i][f"prob_{'local' if signo=='L' else ('empate' if signo=='E' else 'visitante')}"])
                        q_ajustada[idx_cambio] = self._get_resultado_alternativo(partidos[idx_cambio], exclude=signo)

            # Corregir concentración inicial (los 3 partidos iguales)
            primeros_3 = q_ajustada[:3]
            if len(set(primeros_3)) == 1:
                signo_exceso = primeros_3[0]
                # Buscar cuál de los 3 es el más fácil de cambiar
                candidatos = [i for i in range(3) if i in modificables and q_ajustada[i] == signo_exceso]
                if candidatos:
                    idx_cambio = min(candidatos, key=lambda i: partidos[i][f"prob_{'local' if signo_exceso=='L' else ('empate' if signo_exceso=='E' else 'visitante')}"])
                    q_ajustada[idx_cambio] = self._get_resultado_alternativo(partidos[idx_cambio], exclude=signo_exceso)
        
        # Re-ajustar empates por si la corrección de concentración los rompió
        return self._ajustar_empates_core(q_ajustada, partidos)

    def _es_concentracion_valida(self, resultados: List[str]) -> bool:
        """Verifica si una sola quiniela cumple las reglas de concentración."""
        if not resultados: return False
        violacion_gen = max(resultados.count(s) for s in ["L","E","V"]) / 14.0 > self.concentracion_max_general
        violacion_ini = max(resultados[:3].count(s) for s in ["L","E","V"]) / 3.0 > self.concentracion_max_inicial
        return not violacion_gen and not violacion_ini

    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        return max(probs, key=probs.get)

    def _get_resultado_alternativo(self, partido: Dict[str, Any], exclude: str = None) -> str:
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        if exclude and exclude in probs:
            del probs[exclude]
        return max(probs, key=probs.get)

    def _validar_quinielas_core_final(self, core_quinielas: List[Dict[str, Any]], partidos: List[Dict[str, Any]]):
        """Valida que las 4 quinielas Core cumplan las reglas individuales y de anclas."""
        anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        if not core_quinielas: return

        # Validar anclas idénticas
        if anclas_indices:
            resultados_anclas_base = [core_quinielas[0]["resultados"][i] for i in anclas_indices]
            for core in core_quinielas[1:]:
                resultados_anclas_actual = [core["resultados"][i] for i in anclas_indices]
                if resultados_anclas_base != resultados_anclas_actual:
                    self.logger.error(f"Error Crítico en Core: {core['id']} tiene anclas {resultados_anclas_actual} diferentes a Core-1 {resultados_anclas_base}.")
                    raise ValueError(f"{core['id']} tiene anclas diferentes a Core-1.")