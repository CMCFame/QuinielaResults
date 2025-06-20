# progol_optimizer/portfolio/core_generator.py - ESTRATEGIA BASE + MUTACIÓN
"""
Generador de Quinielas Core con estrategia de Base + Mutación
GARANTIZA: 4 quinielas Core VÁLIDAS y DIVERSAS
"""

import logging
import random
import copy
from typing import List, Dict, Any

class CoreGenerator:
    """
    Genera 4 quinielas Core usando una estrategia de Base + Mutación para
    garantizar la diversidad y el cumplimiento de todas las reglas.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.concentracion_max_general = self.config["CONCENTRACION_MAX_GENERAL"]
        self.concentracion_max_inicial = self.config["CONCENTRACION_MAX_INICIAL"]
        self.logger.debug(f"CoreGenerator (Base+Mutación) inicializado")

    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("Generando 4 quinielas Core con ESTRATEGIA DE MUTACIÓN...")
        if len(partidos_clasificados) != 14:
            raise ValueError(f"Se requieren 14 partidos, se recibieron {len(partidos_clasificados)}")

        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        no_anclas_indices = [i for i in range(14) if i not in anclas_indices]
        resultados_anclas = {i: self._get_resultado_max_prob(partidos_clasificados[i]) for i in anclas_indices}

        core_quinielas = []

        # --- PASO 1: Crear Core-1 como la base más probable y válida ---
        self.logger.info("Creando Core-1 (base)...")
        base_resultados = [self._get_resultado_max_prob(p) for p in partidos_clasificados]
        core_1_resultados = self._correccion_agresiva_definitiva(base_resultados, partidos_clasificados, anclas_indices)
        core_quinielas.append(self._crear_info_quiniela("Core-1", core_1_resultados))
        self.logger.info(f"✅ Core-1 (Base) generado: {''.join(core_1_resultados)}")

        # --- PASO 2: Generar Core 2, 3 y 4 por mutación de la base ---
        intentos_totales = 0
        while len(core_quinielas) < 4 and intentos_totales < 200:
            intentos_totales += 1
            
            # Crear un candidato a partir del Core-1
            candidato_resultados = core_1_resultados.copy()
            
            # Introducir de 2 a 4 mutaciones en posiciones no-ancla
            num_mutaciones = random.randint(2, 4)
            posiciones_a_mutar = random.sample(no_anclas_indices, k=min(num_mutaciones, len(no_anclas_indices)))
            
            for pos in posiciones_a_mutar:
                resultado_actual = candidato_resultados[pos]
                opciones = ["L", "E", "V"]
                opciones.remove(resultado_actual)
                candidato_resultados[pos] = random.choice(opciones)
            
            # Correr la corrección agresiva sobre el mutante
            candidato_corregido = self._correccion_agresiva_definitiva(candidato_resultados, partidos_clasificados, anclas_indices)
            
            # Verificar si el mutante corregido es válido y aporta diversidad
            if self._es_valido_individualmente(candidato_corregido) and self._aporta_diversidad(candidato_corregido, core_quinielas):
                nueva_id = f"Core-{len(core_quinielas)+1}"
                core_quinielas.append(self._crear_info_quiniela(nueva_id, candidato_corregido))
                self.logger.info(f"✅ {nueva_id} (Mutante) generado: {''.join(candidato_corregido)}")

        if len(core_quinielas) < 4:
            self.logger.error("No se pudieron generar 4 quinielas Core diversas. Pruebe con datos de entrada diferentes.")
            raise RuntimeError("Fallo al generar quinielas Core diversas.")

        self._validacion_final_completa(core_quinielas, anclas_indices)
        return core_quinielas

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
        
    def _correccion_agresiva_definitiva(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                      anclas_indices: List[int]) -> List[str]:
        q_corregida = copy.deepcopy(quiniela)
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        for _ in range(2): # Bucle de estabilización
            q_corregida = self._forzar_empates_validos_agresivo(q_corregida, partidos, modificables)
            q_corregida = self._forzar_concentracion_inicial_valida(q_corregida, partidos, modificables)
            q_corregida = self._forzar_concentracion_general_valida(q_corregida, partidos, modificables)
        
        return q_corregida

    def _forzar_empates_validos_agresivo(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                       modificables: List[int]) -> List[str]:
        q_ajustada = quiniela.copy()
        empates_actuales = q_ajustada.count("E")
        
        if empates_actuales < self.empates_min:
            necesarios = self.empates_min - empates_actuales
            candidatos = sorted([i for i in modificables if q_ajustada[i] != "E"], key=lambda i: partidos[i]["prob_empate"], reverse=True)
            for idx in candidatos[:necesarios]: q_ajustada[idx] = "E"
                
        elif empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            candidatos = sorted([i for i in modificables if q_ajustada[i] == "E"], key=lambda i: partidos[i]["prob_empate"])
            for idx in candidatos[:exceso]:
                partido = partidos[idx]
                q_ajustada[idx] = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"

        return q_ajustada

    def _forzar_concentracion_inicial_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                           modificables: List[int]) -> List[str]:
        q_corregida = quiniela.copy()
        modificables_iniciales = [i for i in modificables if i < 3]

        while True:
            primeros_3 = q_corregida[:3]
            counts = {"L": primeros_3.count("L"), "E": primeros_3.count("E"), "V": primeros_3.count("V")}
            signo_problema = max(counts, key=counts.get)

            if counts[signo_problema] <= int(3 * self.concentracion_max_inicial): break

            self.logger.debug(f"Corrigiendo Conc. Inicial: {signo_problema} tiene {counts[signo_problema]}/3")
            
            indices_modificables = [i for i in modificables_iniciales if q_corregida[i] == signo_problema]
            if not indices_modificables: break

            idx_a_cambiar = min(indices_modificables, key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo_problema)}"])
            
            signo_menos_usado = min(counts, key=lambda s: (counts[s], s != signo_problema))
            q_corregida[idx_a_cambiar] = signo_menos_usado
        
        return q_corregida

    def _forzar_concentracion_general_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]], modificables: List[int]) -> List[str]:
        q_corregida = quiniela.copy()
        max_permitido = int(14 * self.concentracion_max_general)

        for signo in ["L", "E", "V"]:
            while q_corregida.count(signo) > max_permitido:
                exceso = q_corregida.count(signo) - max_permitido
                
                indices_modificables = [i for i in modificables if q_corregida[i] == signo]
                if not indices_modificables: break
                
                # Cambiar los 'exceso' menos probables
                indices_a_cambiar = sorted(indices_modificables, key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])[:exceso]
                
                for idx in indices_a_cambiar:
                    otros_signos = [s for s in ["L", "E", "V"] if s != signo]
                    q_corregida[idx] = min(otros_signos, key=lambda s: q_corregida.count(s))

        return q_corregida
    
    # ... [Asegúrate de incluir las funciones _get_resultado_max_prob, _resultado_a_clave y _validacion_final_completa del archivo original] ...
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        return max(probs, key=probs.get)

    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte resultado a clave de probabilidad"""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "local")

    def _validacion_final_completa(self, core_quinielas: List[Dict[str, Any]], anclas_indices: List[int]):
        """
        Validación final completa con logging detallado
        """
        self.logger.info("=== VALIDACIÓN FINAL COMPLETA ===")
        
        problemas = []
        
        # Validar cada quiniela individualmente
        for q in core_quinielas:
            if not self._es_valido_individualmente(q['resultados']):
                 problemas.append(f"{q['id']}: No cumple con las reglas individuales.")
        
        # Validar distribución global
        total_L = sum(q["distribución"]["L"] for q in core_quinielas)
        total_E = sum(q["distribución"]["E"] for q in core_quinielas)
        total_V = sum(q["distribución"]["V"] for q in core_quinielas)
        total_partidos = len(core_quinielas) * 14
        
        porc_L = total_L / total_partidos
        porc_E = total_E / total_partidos
        porc_V = total_V / total_partidos
        
        if not (0.35 <= porc_L <= 0.41): problemas.append(f"Distribución L={porc_L:.1%} fuera de 35-41%")
        if not (0.25 <= porc_E <= 0.33): problemas.append(f"Distribución E={porc_E:.1%} fuera de 25-33%")
        if not (0.30 <= porc_V <= 0.36): problemas.append(f"Distribución V={porc_V:.1%} fuera de 30-36%")
        
        # Reportar resultados
        if problemas:
            self.logger.error("❌ PROBLEMAS DETECTADOS EN LA GENERACIÓN FINAL:")
            for problema in problemas:
                self.logger.error(f"  - {problema}")
        else:
            self.logger.info("✅ TODAS LAS VALIDACIONES FINALES PASARON")
        
        self.logger.info(f"Distribución FINAL: L={porc_L:.1%}, E={porc_E:.1%}, V={porc_V:.1%}")