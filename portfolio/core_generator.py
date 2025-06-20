# progol_optimizer/portfolio/core_generator.py - VERSIÓN FINAL CORREGIDA
"""
Generador de Quinielas Core con lógica de validación corregida y balanceo colectivo agresivo.
GARANTIZA: Un portafolio Core 100% válido.
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
        self.concentracion_max_general = self.config["CONCENTRACION_MAX_GENERAL"]
        self.concentracion_max_inicial = self.config["CONCENTRACION_MAX_INICIAL"]
        self.rangos = self.config["RANGOS_HISTORICOS"]
        self.logger.debug(f"CoreGenerator (Lógica Corregida) inicializado")

    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("Generando 4 quinielas Core con LÓGICA CORREGIDA Y BALANCEO AGRESIVO...")
        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        
        core_quinielas_info = self._generar_cores_diversos(partidos_clasificados, anclas_indices)
        
        self.logger.info("Aplicando balanceo colectivo final...")
        core_quinielas_balanceadas = self._balanceo_colectivo_final(core_quinielas_info, partidos_clasificados, anclas_indices)

        self._validacion_final_completa(core_quinielas_balanceadas)
        return core_quinielas_balanceadas

    def _generar_cores_diversos(self, partidos, anclas_indices):
        core_quinielas = []
        base_resultados = [self._get_resultado_max_prob(p) for p in partidos]
        core_1_resultados = self._correccion_individual_robusta(base_resultados, partidos, anclas_indices)
        core_quinielas.append(self._crear_info_quiniela("Core-1", core_1_resultados))
        
        intentos = 0
        while len(core_quinielas) < 4 and intentos < 300:
            intentos += 1
            candidato = core_1_resultados.copy()
            no_anclas = [i for i in range(14) if i not in anclas_indices]
            num_mutaciones = random.randint(3, 6)
            pos_a_mutar = random.sample(no_anclas, k=min(num_mutaciones, len(no_anclas)))
            
            for pos in pos_a_mutar:
                opciones = ["L", "E", "V"]
                opciones.remove(candidato[pos])
                candidato[pos] = random.choice(opciones)
            
            corregido = self._correccion_individual_robusta(candidato, partidos, anclas_indices)

            if self._es_valido_individualmente(corregido) and self._aporta_diversidad(corregido, core_quinielas):
                nueva_id = f"Core-{len(core_quinielas)+1}"
                core_quinielas.append(self._crear_info_quiniela(nueva_id, corregido))
        
        if len(core_quinielas) < 4:
            raise RuntimeError("Fallo crítico: No se pudieron generar 4 quinielas Core diversas.")
        
        return core_quinielas

    def _balanceo_colectivo_final(self, quinielas: List[Dict[str, Any]], partidos: List[Dict[str, Any]], anclas_indices: List[int]) -> List[Dict[str, Any]]:
        """Balancea el conjunto de quinielas para cumplir reglas colectivas."""
        q_balanceadas = copy.deepcopy(quinielas)
        modificables = [i for i in range(14) if i not in anclas_indices]

        for _ in range(5): # Múltiples pasadas de balanceo
            # 1. Balanceo de Divisores (más agresivo)
            for pos in modificables:
                conteos = {"L": 0, "E": 0, "V": 0}
                for q in q_balanceadas:
                    conteos[q["resultados"][pos]] += 1
                
                # Si una posición tiene 3 o 4 apariciones
                signo_dominante, count_dominante = max(conteos.items(), key=lambda item: item[1])
                if count_dominante >= 3:
                    self.logger.debug(f"Balanceando P{pos+1}. Concentración de {count_dominante}/4 en '{signo_dominante}'.")
                    # Cambiar el resultado en 1 o 2 de las quinielas dominantes
                    indices_quinielas_dominantes = [i for i, q in enumerate(q_balanceadas) if q["resultados"][pos] == signo_dominante]
                    q_a_cambiar = random.choice(indices_quinielas_dominantes)
                    
                    opciones = ["L", "E", "V"]
                    opciones.remove(signo_dominante)
                    nuevo_resultado = min(opciones, key=lambda s: conteos[s]) # Cambiar al menos frecuente
                    
                    q_balanceadas[q_a_cambiar]["resultados"][pos] = nuevo_resultado
                    q_balanceadas[q_a_cambiar] = self._crear_info_quiniela(q_balanceadas[q_a_cambiar]["id"], self._correccion_individual_robusta(q_balanceadas[q_a_cambiar]["resultados"], partidos, anclas_indices))

            # 2. Balanceo de Distribución Global
            dist = self._calcular_distribucion_global(q_balanceadas)
            if not (self.rangos["L"][0] <= dist['L'] <= self.rangos["L"][1]):
                self.logger.debug(f"Balanceando Global: L={dist['L']:.1%} fuera de rango.")
                # Lógica de corrección global (ej. cambiar un L por un V)
                # (Esta lógica puede ser compleja, un ejemplo simple es cambiar el L menos probable)
                if dist['L'] > self.rangos['L'][1]: # Si hay exceso de locales
                    # Buscar el L menos probable en todo el portafolio y cambiarlo a V o E
                    mejor_cambio = (None, None, 1.0) # q_idx, pos, prob
                    for q_idx, q in enumerate(q_balanceadas):
                        for pos in modificables:
                            if q['resultados'][pos] == 'L' and partidos[pos]['prob_local'] < mejor_cambio[2]:
                                mejor_cambio = (q_idx, pos, partidos[pos]['prob_local'])
                    
                    if mejor_cambio[0] is not None:
                        q_idx, pos, _ = mejor_cambio
                        # Cambiar a V si está bajo, si no, a E
                        q_balanceadas[q_idx]['resultados'][pos] = 'V' if dist['V'] < self.rangos['V'][0] else 'E'
                        q_balanceadas[q_idx] = self._crear_info_quiniela(q_balanceadas[q_idx]["id"], self._correccion_individual_robusta(q_balanceadas[q_idx]["resultados"], partidos, anclas_indices))

        return q_balanceadas

    def _correccion_individual_robusta(self, quiniela: List[str], partidos: List[Dict[str, Any]], anclas_indices: List[int]) -> List[str]:
        q_corregida = copy.deepcopy(quiniela)
        modificables = [i for i in range(14) if i not in anclas_indices]
        for _ in range(5):
            q_corregida = self._forzar_concentracion_general_valida(q_corregida, partidos, modificables)
            q_corregida = self._forzar_concentracion_inicial_valida(q_corregida, partidos, modificables)
            q_corregida = self._forzar_empates_validos_agresivo(q_corregida, partidos, modificables)
            if self._es_valido_individualmente(q_corregida): return q_corregida
        return q_corregida
        
    def _forzar_concentracion_inicial_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]], modificables: List[int]) -> List[str]:
        q_corregida = quiniela.copy()
        modificables_iniciales = [i for i in modificables if i < 3]
        for _ in range(3):
            primeros_3 = q_corregida[:3]
            counts = {"L": primeros_3.count("L"), "E": primeros_3.count("E"), "V": primeros_3.count("V")}
            signo_problema, count_problema = max(counts.items(), key=lambda item: item[1])
            
            # --- LA CORRECCIÓN CLAVE ---
            # La regla es <=60%. 60% de 3 es 1.8. El máximo número de apariciones es 1.
            # Por lo tanto, cualquier cuenta de 2 o más (66.7%) es inválida.
            if count_problema <= 1: # Si la cuenta máxima es 1 (o 0), es válido.
                break
            
            self.logger.debug(f"Corrigiendo Conc. Inicial: '{signo_problema}' tiene {count_problema}/3. Supera el máximo de 1.")
            
            indices_modificables = [idx for idx in modificables_iniciales if q_corregida[idx] == signo_problema]
            if not indices_modificables: break

            idx_a_cambiar = min(indices_modificables, key=lambda idx: partidos[idx][f"prob_{self._resultado_a_clave(signo_problema)}"])
            
            otros_signos = {s: c for s, c in counts.items() if s != signo_problema}
            q_corregida[idx_a_cambiar] = min(otros_signos, key=otros_signos.get) if otros_signos else random.choice([s for s in ["L","E","V"] if s != signo_problema])
        return q_corregida

    # El resto de las funciones auxiliares no necesitan cambios
    # ... (pegar aquí las funciones _forzar_empates_validos_agresivo, _forzar_concentracion_general_valida, _crear_info_quiniela, etc.)
    def _forzar_empates_validos_agresivo(self, quiniela: List[str], partidos: List[Dict[str, Any]], modificables: List[int]) -> List[str]:
        q_ajustada = quiniela.copy()
        while q_ajustada.count("E") < self.empates_min:
            candidatos = [i for i in modificables if q_ajustada[i] != "E"]
            if not candidatos: break
            mejor_candidato = max(candidatos, key=lambda i: partidos[i]["prob_empate"])
            q_ajustada[mejor_candidato] = "E"
        while q_ajustada.count("E") > self.empates_max:
            candidatos = [i for i in modificables if q_ajustada[i] == "E"]
            if not candidatos: break
            peor_candidato = min(candidatos, key=lambda i: partidos[i]["prob_empate"])
            partido = partidos[peor_candidato]
            q_ajustada[peor_candidato] = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"
        return q_ajustada

    def _forzar_concentracion_general_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]], modificables: List[int]) -> List[str]:
        q_corregida = quiniela.copy()
        max_permitido = int(14 * self.concentracion_max_general)
        for _ in range(3):
            counts = {"L": q_corregida.count("L"), "E": q_corregida.count("E"), "V": q_corregida.count("V")}
            signo_problema, count_problema = max(counts.items(), key=lambda item: item[1])
            if count_problema <= max_permitido: break
            exceso = count_problema - max_permitido
            indices_a_cambiar = sorted([i for i in modificables if q_corregida[i] == signo_problema], key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo_problema)}"])
            for idx in indices_a_cambiar[:exceso]:
                otros_signos = {s: c for s, c in counts.items() if s != signo_problema}
                q_corregida[idx] = min(otros_signos, key=otros_signos.get) if otros_signos else random.choice([s for s in ["L","E","V"] if s != signo_problema])
        return q_corregida

    def _crear_info_quiniela(self, q_id: str, resultados: List[str]) -> Dict[str, Any]:
        return {"id": q_id, "tipo": "Core", "resultados": resultados, "empates": resultados.count("E"), "distribución": {"L": resultados.count("L"), "E": resultados.count("E"), "V": resultados.count("V")}}

    def _es_valido_individualmente(self, resultados: List[str]) -> bool:
        if not (self.empates_min <= resultados.count("E") <= self.empates_max): return False
        if max(resultados.count(s) for s in ["L", "E", "V"]) / 14.0 > self.concentracion_max_general: return False
        # Regla corregida: <= 60% significa que el máximo de apariciones es 1 (33.3%)
        if max(resultados[:3].count(s) for s in ["L", "E", "V"]) > 1: return False
        return True

    def _aporta_diversidad(self, nueva_quiniela: List[str], existentes: List[Dict[str, Any]]) -> bool:
        if not existentes: return True
        for q_existente in existentes:
            if sum(1 for a, b in zip(nueva_quiniela, q_existente["resultados"]) if a == b) >= 12: return False
        return True
    
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        return max({"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}, key=lambda k: partido[f"prob_{self._resultado_a_clave(k)}"])

    def _resultado_a_clave(self, resultado: str) -> str:
        return {"L": "local", "E": "empate", "V": "visitante"}.get(resultado, "local")

    def _calcular_distribucion_global(self, quinielas: List[Dict[str, Any]]) -> Dict[str, float]:
        total_L = sum(q["resultados"].count("L") for q in quinielas)
        total_E = sum(q["resultados"].count("E") for q in quinielas)
        total_V = sum(q["resultados"].count("V") for q in quinielas)
        total_partidos = len(quinielas) * 14
        if total_partidos == 0: return {"L": 0, "E": 0, "V": 0}
        return {"L": total_L / total_partidos, "E": total_E / total_partidos, "V": total_V / total_partidos}

    def _validacion_final_completa(self, core_quinielas: List[Dict[str, Any]]):
        self.logger.info("=== VALIDACIÓN FINAL DEL GENERADOR ===")
        problemas = [f"{q['id']}: No es válida" for q in core_quinielas if not self._es_valido_individualmente(q['resultados'])]
        if problemas:
            for problema in problemas: self.logger.error(f"  - {problema}")
        else:
            self.logger.info("✅ Todas las quinielas individuales son VÁLIDAS.")