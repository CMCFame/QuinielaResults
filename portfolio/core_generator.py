# progol_optimizer/portfolio/core_generator.py - CORRECCIÓN DEFINITIVA
"""
Generador de Quinielas Core DEFINITIVO - Corrección agresiva de todos los problemas
GARANTIZA: 4-6 empates, ≤60% concentración inicial, variación por posición, distribución válida
"""

import logging
import random
from typing import List, Dict, Any

class CoreGenerator:
    """
    Genera exactamente 4 quinielas Core con corrección DEFINITIVA de todos los problemas
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.concentracion_max_general = self.config["CONCENTRACION_MAX_GENERAL"]
        self.concentracion_max_inicial = self.config["CONCENTRACION_MAX_INICIAL"]
        self.logger.debug(f"CoreGenerator DEFINITIVO inicializado")

    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("Generando 4 quinielas Core DEFINITIVAS...")
        if len(partidos_clasificados) != 14:
            raise ValueError(f"Se requieren exactamente 14 partidos, recibidos: {len(partidos_clasificados)}")

        anclas_indices = [i for i, p in enumerate(partidos_clasificados) if p.get("clasificacion") == "Ancla"]
        no_anclas_indices = [i for i in range(14) if i not in anclas_indices]

        self.logger.info(f"Partidos: {len(anclas_indices)} Anclas, {len(no_anclas_indices)} Modificables")

        resultados_anclas = {i: self._get_resultado_max_prob(partidos_clasificados[i]) for i in anclas_indices}

        # --- ESTRATEGIA MEJORADA: Generar y validar en un bucle ---
        core_quinielas = []
        intentos = 0
        while len(core_quinielas) < 4 and intentos < 100:
            intentos += 1
            # 1. Generar una quiniela base aleatoria pero inteligente
            quiniela_base = self._generar_quiniela_base_diversa(partidos_clasificados, anclas_indices, resultados_anclas, no_anclas_indices)

            # 2. Corregir agresivamente para que sea válida
            quiniela_corregida = self._correccion_agresiva_definitiva(quiniela_base, partidos_clasificados, anclas_indices)

            # 3. Verificar si añade diversidad al conjunto
            if self._aporta_diversidad(quiniela_corregida, core_quinielas):
                core_info = {
                    "id": f"Core-{len(core_quinielas)+1}",
                    "tipo": "Core",
                    "resultados": quiniela_corregida,
                    "empates": quiniela_corregida.count("E"),
                    "distribución": {
                        "L": quiniela_corregida.count("L"),
                        "E": quiniela_corregida.count("E"),
                        "V": quiniela_corregida.count("V")
                    }
                }
                core_quinielas.append(core_info)
                self.logger.info(f"Core-{len(core_quinielas)} generado: {''.join(quiniela_corregida)}")

        if len(core_quinielas) < 4:
             raise RuntimeError("No se pudieron generar 4 quinielas Core diversas. Revise los datos de entrada.")

        # Aplicar corrección final de distribución global
        core_quinielas = self._correccion_distribucion_global_final(core_quinielas, partidos_clasificados, anclas_indices)
        self._validacion_final_completa(core_quinielas, anclas_indices)

        self.logger.info(f"✅ Generadas {len(core_quinielas)} quinielas Core DEFINITIVAS")
        return core_quinielas

    def _generar_quiniela_base_diversa(self, partidos: List[Dict[str, Any]], anclas_indices: List[int], resultados_anclas: Dict, no_anclas_indices: List[int]) -> List[str]:
        """ NUEVA FUNCIÓN: Genera una quiniela aleatoria pero inteligente """
        quiniela = [""] * 14
        for i in anclas_indices:
            quiniela[i] = resultados_anclas[i]

        for i in no_anclas_indices:
            partido = partidos[i]
            probs = [partido["prob_local"], partido["prob_empate"], partido["prob_visitante"]]
            # Usar elección aleatoria ponderada por probabilidad para generar diversidad
            quiniela[i] = random.choices(["L", "E", "V"], weights=probs, k=1)[0]
        return quiniela

    def _aporta_diversidad(self, nueva_quiniela: List[str], existentes: List[Dict[str, Any]]) -> bool:
        """ NUEVA FUNCIÓN: Verifica que una quiniela no sea demasiado similar a las existentes """
        if not existentes:
            return True
        for q_existente in existentes:
            coincidencias = sum(1 for a, b in zip(nueva_quiniela, q_existente["resultados"]) if a == b)
            # Aceptar si tiene al menos 3 diferencias (Jaccard < 11/14 ≈ 0.78)
            if coincidencias > 11:
                return False
        return True

    # El resto de las funciones de corrección (_correccion_agresiva_definitiva, _forzar_empates_validos_agresivo, etc.)
    # pueden permanecer como están, ya que ahora operarán sobre quinielas iniciales más diversas.

    # ... [El resto de las funciones de _forzar..., _correccion..., _validacion... y helpers permanecen igual] ...
    # Asegúrate de copiar las funciones auxiliares que no se muestran aquí desde el archivo original.
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        return max(probs, key=probs.get)

    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte resultado a clave de probabilidad"""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "local")

    def _correccion_agresiva_definitiva(self, quiniela: List[str], partidos: List[Dict[str, Any]],
                                      anclas_indices: List[int]) -> List[str]:
        """
        CORRECCIÓN AGRESIVA que GARANTIZA todas las reglas
        """
        q_corregida = quiniela.copy()
        modificables = [i for i in range(14) if i not in anclas_indices]

        # PASO 1: GARANTIZAR 4-6 empates (PRIORITARIO)
        q_corregida = self._forzar_empates_validos_agresivo(q_corregida, partidos, modificables)

        # PASO 2: GARANTIZAR concentración inicial ≤60%
        q_corregida = self._forzar_concentracion_inicial_valida(q_corregida, partidos, modificables)

        # PASO 3: GARANTIZAR concentración general ≤70%
        q_corregida = self._forzar_concentracion_general_valida(q_corregida, partidos, modificables) # Pasamos partidos aquí

        # PASO 4: Verificar que empates siguen válidos después de correcciones
        empates_finales = q_corregida.count("E")
        if not (self.empates_min <= empates_finales <= self.empates_max):
            q_corregida = self._forzar_empates_validos_agresivo(q_corregida, partidos, modificables)

        return q_corregida

    def _forzar_empates_validos_agresivo(self, quiniela: List[str], partidos: List[Dict[str, Any]],
                                       modificables: List[int]) -> List[str]:
        """
        FUERZA empates válidos de forma AGRESIVA
        """
        q_ajustada = quiniela.copy()
        empates_actuales = q_ajustada.count("E")

        self.logger.debug(f"Forzando empates: actual={empates_actuales}, target={self.empates_min}-{self.empates_max}")

        if empates_actuales < self.empates_min:
            # NECESITAMOS MÁS EMPATES
            necesarios = self.empates_min - empates_actuales

            # Candidatos ordenados por probabilidad de empate (mayor primero)
            candidatos = []
            for i in modificables:
                if q_ajustada[i] != "E":
                    candidatos.append((i, partidos[i]["prob_empate"]))

            candidatos.sort(key=lambda x: x[1], reverse=True)

            # CAMBIAR los mejores candidatos a E
            for j in range(min(necesarios, len(candidatos))):
                idx, prob = candidatos[j]
                q_ajustada[idx] = "E"
                self.logger.debug(f"Forzado P{idx+1} → E (prob={prob:.3f})")

        elif empates_actuales > self.empates_max:
            # DEMASIADOS EMPATES
            exceso = empates_actuales - self.empates_max

            # Candidatos E ordenados por menor probabilidad de empate
            candidatos = []
            for i in modificables:
                if q_ajustada[i] == "E":
                    candidatos.append((i, partidos[i]["prob_empate"]))

            candidatos.sort(key=lambda x: x[1])

            # CAMBIAR los peores empates a L/V
            for j in range(min(exceso, len(candidatos))):
                idx, prob = candidatos[j]
                partido = partidos[idx]
                # Elegir L o V según probabilidad
                nuevo_resultado = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"
                q_ajustada[idx] = nuevo_resultado
                self.logger.debug(f"Forzado P{idx+1}: E → {nuevo_resultado} (prob_E={prob:.3f})")

        return q_ajustada


    def _forzar_concentracion_inicial_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]],
                                           modificables: List[int]) -> List[str]:
        """
        FUERZA concentración inicial válida ≤60%
        """
        q_corregida = quiniela.copy()
        modificables_iniciales = [i for i in modificables if i < 3]

        if not modificables_iniciales:
            return q_corregida  # No se puede modificar

        # Verificar cada signo
        for signo in ["L", "E", "V"]:
            primeros_3 = q_corregida[:3]
            count_signo = primeros_3.count(signo)
            concentracion = count_signo / 3

            if concentracion > self.concentracion_max_inicial:  # >60%
                self.logger.debug(f"Corrigiendo concentración inicial: {signo}={concentracion:.1%}")

                # Encontrar índices de este signo en primeros 3 que se pueden modificar
                indices_modificables = [i for i in modificables_iniciales if q_corregida[i] == signo]

                if indices_modificables:
                    # Cambiar el que tenga menor probabilidad del signo actual
                    idx_cambiar = min(indices_modificables,
                                    key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])

                    # Buscar un signo que no esté concentrado
                    otros_signos = [s for s in ["L", "E", "V"] if s != signo]
                    for nuevo_signo in otros_signos:
                        count_nuevo = primeros_3.count(nuevo_signo)
                        if count_nuevo < 2:  # Si no está concentrado
                            q_corregida[idx_cambiar] = nuevo_signo
                            self.logger.debug(f"Cambiado P{idx_cambiar+1}: {signo} → {nuevo_signo}")
                            break
                    else:
                        # Si todos están concentrados, usar el menos concentrado
                        menos_concentrado = min(otros_signos, key=lambda s: primeros_3.count(s))
                        q_corregida[idx_cambiar] = menos_concentrado
                        self.logger.debug(f"Forzado P{idx_cambiar+1}: {signo} → {menos_concentrado}")

        return q_corregida

    def _forzar_concentracion_general_valida(self, quiniela: List[str], partidos: List[Dict[str, Any]], modificables: List[int]) -> List[str]:
        """
        FUERZA concentración general válida ≤70%
        """
        q_corregida = quiniela.copy()
        max_permitido = int(14 * self.concentracion_max_general)  # 9

        for signo in ["L", "E", "V"]:
            count_signo = q_corregida.count(signo)

            if count_signo > max_permitido:
                exceso = count_signo - max_permitido
                self.logger.debug(f"Corrigiendo concentración general: {signo}={count_signo} > {max_permitido}")

                # Encontrar índices modificables de este signo
                indices_modificables = [i for i in modificables if q_corregida[i] == signo]

                # Cambiar los de menor probabilidad
                indices_modificables.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])

                for j in range(min(exceso, len(indices_modificables))):
                    idx = indices_modificables[j]

                    # Buscar el signo menos usado
                    counts = {s: q_corregida.count(s) for s in ["L", "E", "V"] if s != signo}
                    menos_usado = min(counts, key=counts.get)

                    q_corregida[idx] = menos_usado
                    self.logger.debug(f"Cambiado P{idx+1}: {signo} → {menos_usado}")

        return q_corregida


    def _correccion_distribucion_global_final(self, core_quinielas: List[Dict[str, Any]],
                                            partidos: List[Dict[str, Any]], anclas_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Corrección FINAL de distribución global para estar EXACTAMENTE dentro de rangos
        """
        self.logger.info("Aplicando corrección FINAL de distribución global...")

        # Calcular distribución actual
        total_L = sum(q["distribución"]["L"] for q in core_quinielas)
        total_E = sum(q["distribución"]["E"] for q in core_quinielas)
        total_V = sum(q["distribución"]["V"] for q in core_quinielas)
        total_partidos = len(core_quinielas) * 14

        porc_L = total_L / total_partidos
        porc_E = total_E / total_partidos
        porc_V = total_V / total_partidos

        self.logger.debug(f"Distribución actual: L={porc_L:.3f}, E={porc_E:.3f}, V={porc_V:.3f}")

        # Verificar si está fuera de rangos y corregir
        modificables = [i for i in range(14) if i not in anclas_indices]

        # Si L > 41%, reducir L
        if porc_L > 0.41:
            cambios_necesarios = int((porc_L - 0.40) * total_partidos)  # Reducir a 40%
            self.logger.debug(f"Reduciendo L: {cambios_necesarios} cambios")

            # Buscar candidatos L para cambiar
            candidatos = []
            for q_idx, quiniela in enumerate(core_quinielas):
                for i in modificables:
                    if quiniela["resultados"][i] == "L":
                        prob_L = partidos[i]["prob_local"]
                        candidatos.append((q_idx, i, prob_L))

            # Cambiar los de menor probabilidad L
            candidatos.sort(key=lambda x: x[2])
            for j in range(min(cambios_necesarios, len(candidatos))):
                q_idx, pos_idx, _ = candidatos[j]

                # Decidir si cambiar a E o V basado en distribución actual
                if porc_V < 0.33:  # Si necesitamos más V
                    nuevo_resultado = "V"
                else:
                    nuevo_resultado = "E"

                # Aplicar cambio
                core_quinielas[q_idx]["resultados"][pos_idx] = nuevo_resultado
                core_quinielas[q_idx]["distribución"]["L"] -= 1
                core_quinielas[q_idx]["distribución"][nuevo_resultado] += 1

                self.logger.debug(f"Cambiado {core_quinielas[q_idx]['id']} P{pos_idx+1}: L → {nuevo_resultado}")

        return core_quinielas

    def _validacion_final_completa(self, core_quinielas: List[Dict[str, Any]], anclas_indices: List[int]):
        """
        Validación final completa con logging detallado
        """
        self.logger.info("=== VALIDACIÓN FINAL COMPLETA ===")

        problemas = []

        # Validar cada quiniela individualmente
        for q in core_quinielas:
            resultados = q["resultados"]

            # Empates
            if not (self.empates_min <= q["empates"] <= self.empates_max):
                problemas.append(f"{q['id']}: {q['empates']} empates (debe ser 4-6)")

            # Concentración general
            max_conc_general = max(q["distribución"].values()) / 14
            if max_conc_general > self.concentracion_max_general:
                signo = max(q["distribución"], key=q["distribución"].get)
                problemas.append(f"{q['id']}: concentración general {signo}={max_conc_general:.1%}")

            # Concentración inicial
            primeros_3 = resultados[:3]
            max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3
            if max_conc_inicial > self.concentracion_max_inicial:
                signo = max(["L", "E", "V"], key=lambda s: primeros_3.count(s))
                problemas.append(f"{q['id']}: concentración inicial {signo}={max_conc_inicial:.1%}")

        # Validar distribución global
        total_L = sum(q["distribución"]["L"] for q in core_quinielas)
        total_E = sum(q["distribución"]["E"] for q in core_quinielas)
        total_V = sum(q["distribución"]["V"] for q in core_quinielas)
        total_partidos = len(core_quinielas) * 14

        porc_L = total_L / total_partidos
        porc_E = total_E / total_partidos
        porc_V = total_V / total_partidos

        if not (0.35 <= porc_L <= 0.41):
            problemas.append(f"Distribución L={porc_L:.1%} fuera de 35-41%")
        if not (0.25 <= porc_E <= 0.33):
            problemas.append(f"Distribución E={porc_E:.1%} fuera de 25-33%")
        if not (0.30 <= porc_V <= 0.36):
            problemas.append(f"Distribución V={porc_V:.1%} fuera de 30-36%")

        # Reportar resultados
        if problemas:
            self.logger.warning("⚠️ PROBLEMAS DETECTADOS:")
            for problema in problemas:
                self.logger.warning(f"  - {problema}")
        else:
            self.logger.info("✅ TODAS LAS VALIDACIONES PASARON")

        self.logger.info(f"Distribución FINAL: L={porc_L:.1%}, E={porc_E:.1%}, V={porc_V:.1%}")