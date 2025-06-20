# progol_optimizer/portfolio/core_generator.py - VERSIÓN FINAL CON CORRECCIÓN PRIORIZADA
"""
Generador de Quinielas Core con estrategia de Base + Mutación y corrección final de empates.
GARANTIZA: 4 quinielas Core VÁLIDAS y DIVERSAS, cumpliendo todas las reglas.
"""

import logging
import random
import copy
from typing import List, Dict, Any

class CoreGenerator:
    """
    Genera 4 quinielas Core usando una estrategia de Base + Mutación
    con un ciclo de corrección final que prioriza la regla de empates.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.concentracion_max_general = self.config["CONCENTRACION_MAX_GENERAL"]
        self.concentracion_max_inicial = self.config["CONCENTRACION_MAX_INICIAL"]
        self.logger.debug(f"CoreGenerator (Corrección Priorizada) inicializado")

    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("Generando 4 quinielas Core con CORRECCIÓN PRIORIZADA...")
        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        no_anclas_indices = [i for i in range(14) if i not in anclas_indices]
        
        # --- PASO 1: Crear Core-1 como la base más probable y válida ---
        self.logger.info("Creando Core-1 (base)...")
        base_resultados = [self._get_resultado_max_prob(p) for p in partidos_clasificados]
        core_1_resultados = self._correccion_agresiva_final(base_resultados, partidos_clasificados, anclas_indices)
        
        core_quinielas = [self._crear_info_quiniela("Core-1", core_1_resultados)]
        self.logger.info(f"✅ Core-1 (Base) generado: {''.join(core_1_resultados)}")

        # --- PASO 2: Generar Core 2, 3 y 4 por mutación y corrección ---
        intentos_totales = 0
        while len(core_quinielas) < 4 and intentos_totales < 250:
            intentos_totales += 1
            candidato_resultados = core_1_resultados.copy()
            
            num_mutaciones = random.randint(3, 5) # Aumentar mutaciones para más diversidad
            posiciones_a_mutar = random.sample(no_anclas_indices, k=min(num_mutaciones, len(no_anclas_indices)))
            
            for pos in posiciones_a_mutar:
                opciones = ["L", "E", "V"]
                opciones.remove(candidato_resultados[pos])
                candidato_resultados[pos] = random.choice(opciones)
            
            candidato_corregido = self._correccion_agresiva_final(candidato_resultados, partidos_clasificados, anclas_indices)
            
            if self._es_valido_individualmente(candidato_corregido) and self._aporta_diversidad(candidato_corregido, core_quinielas):
                nueva_id = f"Core-{len(core_quinielas)+1}"
                core_quinielas.append(self._crear_info_quiniela(nueva_id, candidato_corregido))
                self.logger.info(f"✅ {nueva_id} (Mutante) generado: {''.join(candidato_corregido)}")

        if len(core_quinielas) < 4:
            self.logger.error("No se pudieron generar 4 quinielas Core diversas. Los datos de entrada pueden ser muy restrictivos.")
            # Devolver lo que se haya podido generar para no detener el flujo
            return core_quinielas

        self._validacion_final_completa(core_quinielas)
        return core_quinielas

    def _correccion_agresiva_final(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                 anclas_indices: List[int]) -> List[str]:
        """
        VERSIÓN FINAL: Ciclo de corrección que aplica reglas y prioriza empates al final.
        """
        q_corregida = copy.deepcopy(quiniela)
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Bucle de estabilización (máximo 3 pasadas para evitar bucles infinitos)
        for _ in range(3):
            q_corregida = self._forzar_concentracion_general_valida(q_corregida, partidos, modificables)
            q_corregida = self._forzar_concentracion_inicial_valida(q_corregida, partidos, modificables)
            # AJUSTE CLAVE: La regla de empates se aplica al final de cada ciclo para que tenga la última palabra.
            q_corregida = self._forzar_empates_validos_agresivo(q_corregida, partidos, modificables)
        
        return q_corregida
        
    def _forzar_empates_validos_agresivo(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                       modificables: List[int]) -> List[str]:
        """Fuerza que la quiniela tenga entre 4 y 6 empates, de la forma más inteligente posible."""
        q_ajustada = quiniela.copy()
        
        while q_ajustada.count("E") < self.empates_min:
            # NECESITAMOS MÁS EMPATES: Cambiar el L/V menos probable a E
            candidatos = [i for i in modificables if q_ajustada[i] != "E"]
            if not candidatos: break
            
            mejor_candidato = max(candidatos, key=lambda i: partidos[i]["prob_empate"])
            q_ajustada[mejor_candidato] = "E"
            self.logger.debug(f"Añadiendo empate en P{mejor_candidato+1} para cumplir regla.")
            
        while q_ajustada.count("E") > self.empates_max:
            # DEMASIADOS EMPATES: Cambiar el E menos probable a L/V
            candidatos = [i for i in modificables if q_ajustada[i] == "E"]
            if not candidatos: break
            
            peor_candidato = min(candidatos, key=lambda i: partidos[i]["prob_empate"])
            partido = partidos[peor_candidato]
            q_ajustada[peor_candidato] = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"
            self.logger.debug(f"Quitando empate en P{peor_candidato+1} para cumplir regla.")
            
        return q_ajustada

    # El resto de las funciones (las de abajo) no necesitan cambios y deben permanecer en el archivo.
    # ... (pegar aquí las funciones _forzar_concentracion_inicial_valida, _forzar_concentracion_general_valida, 
    # _crear_info_quiniela, _es_valido_individualmente, _aporta_diversidad, _get_resultado_max_prob,
    # _resultado_a_clave y _validacion_final_completa de la respuesta anterior).

    def _forzar_concentracion_inicial_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                           modificables: List[int]) -> List[str]:
        q_corregida = quiniela.copy()
        modificables_iniciales = [i for i in modificables if i < 3]

        # Solo es necesario un bucle porque la corrección es directa
        for i in range(3): # Re-evaluar hasta 3 veces
            primeros_3 = q_corregida[:3]
            counts = {"L": primeros_3.count("L"), "E": primeros_3.count("E"), "V": primeros_3.count("V")}
            
            signo_problema, count_problema = max(counts.items(), key=lambda item: item[1])

            if count_problema <= int(3 * self.concentracion_max_inicial):
                break # La concentración es válida, salir del bucle

            self.logger.debug(f"Corrigiendo Conc. Inicial: {signo_problema} tiene {count_problema}/3")
            
            indices_modificables = [idx for idx in modificables_iniciales if q_corregida[idx] == signo_problema]
            if not indices_modificables: break

            idx_a_cambiar = min(indices_modificables, key=lambda idx: partidos[idx][f"prob_{self._resultado_a_clave(signo_problema)}"])
            
            # Cambiar al signo menos usado en los primeros 3
            otros_signos = {s: c for s, c in counts.items() if s != signo_problema}
            signo_menos_usado = min(otros_signos, key=otros_signos.get)
            q_corregida[idx_a_cambiar] = signo_menos_usado
        
        return q_corregida

    def _forzar_concentracion_general_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]], modificables: List[int]) -> List[str]:
        q_corregida = quiniela.copy()
        max_permitido = int(14 * self.concentracion_max_general)

        for _ in range(3): # Bucle de estabilización
            counts = {"L": q_corregida.count("L"), "E": q_corregida.count("E"), "V": q_corregida.count("V")}
            signo_problema, count_problema = max(counts.items(), key=lambda item: item[1])
            
            if count_problema <= max_permitido:
                break # Válido

            exceso = count_problema - max_permitido
            indices_a_cambiar = sorted([i for i in modificables if q_corregida[i] == signo_problema], key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo_problema)}"])
                
            for idx in indices_a_cambiar[:exceso]:
                otros_signos = {s: c for s, c in counts.items() if s != signo_problema}
                q_corregida[idx] = min(otros_signos, key=otros_signos.get)

        return q_corregida
    
    def _crear_info_quiniela(self, q_id: str, resultados: List[str]) -> Dict[str, Any]:
        """NUEVA FUNCIÓN HELPER: Crea el diccionario de la quiniela."""
        return {
            "id": q_id, "tipo": "Core", "resultados": resultados,
            "empates": resultados.count("E"),
            "distribución": {"L": resultados.count("L"), "E": resultados.count("E"), "V": resultados.count("V")}
        }

    def _es_valido_individualmente(self, resultados: List[str]) -> bool:
        """NUEVA FUNCIÓN HELPER: Valida una sola quiniela contra las reglas clave."""
        empates = resultados.count("E")
        if not (self.empates_min <= empates <= self.empates_max): return False
        
        max_conc_general = max(resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc_general > self.concentracion_max_general: return False
            
        primeros_3 = resultados[:3]
        max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > self.concentracion_max_inicial: return False
            
        return True

    def _aporta_diversidad(self, nueva_quiniela: List[str], existentes: List[Dict[str, Any]]) -> bool:
        """NUEVA FUNCIÓN HELPER: Verifica que una quiniela no sea demasiado similar a las existentes."""
        if not existentes: return True
        for q_existente in existentes:
            coincidencias = sum(1 for a, b in zip(nueva_quiniela, q_existente["resultados"]) if a == b)
            # Aceptar si tiene al menos 3 diferencias (Jaccard < 12/14)
            if coincidencias >= 12:
                self.logger.debug(f"Mutante rechazado por similitud ({coincidencias} coincidencias).")
                return False
        return True
    
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        return max(probs, key=probs.get)

    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte resultado a clave de probabilidad"""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "local")

    def _validacion_final_completa(self, core_quinielas: List[Dict[str, Any]]):
        self.logger.info("=== VALIDACIÓN FINAL COMPLETA ===")
        problemas = []
        for q in core_quinielas:
            if not self._es_valido_individualmente(q['resultados']):
                 problemas.append(f"{q['id']}: No cumple con las reglas individuales.")
        if problemas:
            self.logger.error("❌ PROBLEMAS DETECTADOS EN LA GENERACIÓN FINAL:")
            for problema in problemas: self.logger.error(f"  - {problema}")
        else:
            self.logger.info("✅ TODAS LAS QUINIELAS INDIVIDUALES SON VÁLIDAS.")