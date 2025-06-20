# progol_optimizer/portfolio/core_generator.py - VERSIÓN BALANCEADA
"""
Generador de Quinielas Core BALANCEADO - Soluciona problemas específicos detectados
CORRECCIÓN: Reduce concentración inicial, balancea distribución global, crea más variación por posición
"""

import logging
import random
from typing import List, Dict, Any

class CoreGenerator:
    """
    Genera exactamente 4 quinielas Core con variación balanceada y distribución corregida
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.concentracion_max_general = self.config["CONCENTRACION_MAX_GENERAL"]
        self.concentracion_max_inicial = self.config["CONCENTRACION_MAX_INICIAL"]
        self.logger.debug(f"CoreGenerator BALANCEADO inicializado")
    
    def generar_quinielas_core(self, partidos_clasificados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("Generando 4 quinielas Core BALANCEADAS...")
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
        
        # NUEVO: Planificar distribución global target
        target_distribution = self._calcular_distribucion_target()
        
        core_quinielas = []
        for core_id in range(4):
            intentos = 0
            while intentos < 15:  # Más intentos para encontrar solución válida
                quiniela_resultados = self._generar_core_balanceado(
                    partidos_clasificados, core_id, anclas_indices, divisores_indices, 
                    otros_indices, resultados_anclas, target_distribution
                )
                
                # Validar que cumple TODAS las reglas
                if self._es_quiniela_completamente_valida(quiniela_resultados):
                    break
                
                self.logger.debug(f"Core-{core_id+1} falló validación completa, reintentando...")
                intentos += 1
            else:
                self.logger.warning(f"Core-{core_id+1} no pudo validarse completamente tras 15 intentos")

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
            conc_inicial = max(quiniela_resultados[:3].count(s) for s in ["L", "E", "V"]) / 3
            self.logger.debug(f"Core-{core_id+1}: {''.join(quiniela_resultados)} (E:{core_info['empates']}, ConInicial:{conc_inicial:.1%})")
        
        # Validación y balanceo final
        core_quinielas = self._balanceo_final_distribucion(core_quinielas, partidos_clasificados, anclas_indices)
        
        # Validación final
        self._validar_diferencias_y_balance(core_quinielas, anclas_indices)
        
        self.logger.info(f"✅ Generadas {len(core_quinielas)} quinielas Core balanceadas")
        return core_quinielas
    
    def _calcular_distribucion_target(self) -> Dict[str, int]:
        """Calcula distribución target para 4 quinielas (56 partidos totales)"""
        total_partidos = 4 * 14  # 56 partidos
        
        # Usar punto medio de los rangos históricos
        target_L = int(total_partidos * 0.39)  # 39% = punto medio de 35-41%
        target_E = int(total_partidos * 0.29)  # 29% = punto medio de 25-33%
        target_V = total_partidos - target_L - target_E  # El resto
        
        return {"L": target_L, "E": target_E, "V": target_V}
    
    def _generar_core_balanceado(self, partidos: List[Dict[str, Any]], core_id: int,
                                anclas_indices: List[int], divisores_indices: List[int], 
                                otros_indices: List[int], resultados_anclas: Dict[int, str],
                                target_distribution: Dict[str, int]) -> List[str]:
        """
        Genera una quiniela Core individual con balance específico
        """
        quiniela = [""] * 14
        
        # PASO 1: Fijar Anclas (iguales en todas las Core)
        for i in anclas_indices:
            quiniela[i] = resultados_anclas[i]
        
        # PASO 2: NUEVA ESTRATEGIA - Variar específicamente para balance
        
        # Estrategias por Core para crear diversidad controlada
        if core_id == 0:  # Core-1: Más conservador, pero variado en primeros 3
            self._aplicar_estrategia_conservadora_variada(quiniela, partidos, divisores_indices, otros_indices)
        elif core_id == 1:  # Core-2: Priorizar visitantes (corregir distribución)
            self._aplicar_estrategia_visitantes(quiniela, partidos, divisores_indices, otros_indices)
        elif core_id == 2:  # Core-3: Balance empates
            self._aplicar_estrategia_empates(quiniela, partidos, divisores_indices, otros_indices)
        else:  # Core-4: Mix balanceado
            self._aplicar_estrategia_mixta(quiniela, partidos, divisores_indices, otros_indices)
        
        # PASO 3: Ajustar empates manteniendo variación
        quiniela = self._ajustar_empates_con_balance(quiniela, partidos, anclas_indices, core_id)
        
        # PASO 4: NUEVO - Corregir concentración inicial específicamente
        quiniela = self._corregir_concentracion_inicial(quiniela, partidos, anclas_indices)
        
        return quiniela
    
    def _aplicar_estrategia_conservadora_variada(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                               divisores: List[int], otros: List[int]):
        """Core-1: Conservador pero con variación en primeros 3"""
        # Divisores: Máxima probabilidad
        for i in divisores:
            quiniela[i] = self._get_resultado_max_prob(partidos[i])
        
        # Otros: Máxima probabilidad con variación posicional
        for j, i in enumerate(otros):
            if i < 3:  # Primeros 3: crear variación intencional
                if j % 2 == 0:
                    quiniela[i] = self._get_resultado_max_prob(partidos[i])
                else:
                    quiniela[i] = self._get_resultado_segunda_opcion(partidos[i])
            else:
                quiniela[i] = self._get_resultado_max_prob(partidos[i])
    
    def _aplicar_estrategia_visitantes(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                     divisores: List[int], otros: List[int]):
        """Core-2: Priorizar visitantes para corregir distribución global"""
        # Divisores: Favor a visitantes cuando sea razonable
        for i in divisores:
            if partidos[i]["prob_visitante"] > 0.35:  # Si V es razonable
                quiniela[i] = "V"
            else:
                quiniela[i] = self._get_resultado_max_prob(partidos[i])
        
        # Otros: Mix con preferencia a V
        for j, i in enumerate(otros):
            if partidos[i]["prob_visitante"] > 0.25 and j % 3 == 1:  # 1 de cada 3 a V
                quiniela[i] = "V"
            else:
                quiniela[i] = self._get_resultado_max_prob(partidos[i])
    
    def _aplicar_estrategia_empates(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                  divisores: List[int], otros: List[int]):
        """Core-3: Balance con más empates"""
        # Divisores: Empates cuando sea razonable
        for i in divisores:
            if partidos[i]["prob_empate"] > 0.30:  # Si E es razonable
                quiniela[i] = "E"
            else:
                quiniela[i] = self._get_resultado_segunda_opcion(partidos[i])
        
        # Otros: Segunda opción para crear variación
        for i in otros:
            quiniela[i] = self._get_resultado_segunda_opcion(partidos[i])
    
    def _aplicar_estrategia_mixta(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                divisores: List[int], otros: List[int]):
        """Core-4: Mix balanceado"""
        # Divisores: Alternar estrategias
        for j, i in enumerate(divisores):
            if j % 2 == 0:
                quiniela[i] = self._get_resultado_max_prob(partidos[i])
            else:
                quiniela[i] = self._get_resultado_segunda_opcion(partidos[i])
        
        # Otros: Estrategia rotativa
        for j, i in enumerate(otros):
            if j % 3 == 0:
                quiniela[i] = self._get_resultado_max_prob(partidos[i])
            elif j % 3 == 1 and partidos[i]["prob_visitante"] > 0.25:
                quiniela[i] = "V"  # Favor a V para balance
            else:
                quiniela[i] = self._get_resultado_segunda_opcion(partidos[i])
    
    def _corregir_concentracion_inicial(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                      anclas_indices: List[int]) -> List[str]:
        """
        NUEVA FUNCIÓN: Corrige específicamente la concentración en primeros 3 partidos
        """
        primeros_3 = quiniela[:3]
        modificables_primeros_3 = [i for i in range(3) if i not in anclas_indices]
        
        # Verificar concentración actual
        for signo in ["L", "E", "V"]:
            count_signo = primeros_3.count(signo)
            concentracion = count_signo / 3
            
            if concentracion > self.concentracion_max_inicial:  # >60%
                self.logger.debug(f"Corrigiendo concentración inicial: {signo}={concentracion:.1%} en primeros 3")
                
                # Cambiar uno de los signos concentrados
                indices_este_signo = [i for i in modificables_primeros_3 if quiniela[i] == signo]
                
                if indices_este_signo:
                    # Cambiar el que tenga menor probabilidad del signo actual
                    idx_cambiar = min(indices_este_signo, 
                                    key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                    
                    # Cambiar a un signo que no esté concentrado
                    otros_signos = [s for s in ["L", "E", "V"] if s != signo]
                    for nuevo_signo in otros_signos:
                        if primeros_3.count(nuevo_signo) < 2:  # Si este signo no está concentrado
                            quiniela[idx_cambiar] = nuevo_signo
                            self.logger.debug(f"Cambiado P{idx_cambiar+1}: {signo} → {nuevo_signo}")
                            break
        
        return quiniela
    
    def _balanceo_final_distribucion(self, core_quinielas: List[Dict[str, Any]], 
                                   partidos: List[Dict[str, Any]], anclas_indices: List[int]) -> List[Dict[str, Any]]:
        """
        NUEVA FUNCIÓN: Balanceo final para ajustar distribución global
        """
        self.logger.info("Aplicando balanceo final de distribución...")
        
        # Calcular distribución actual
        total_L = sum(q["distribución"]["L"] for q in core_quinielas)
        total_E = sum(q["distribución"]["E"] for q in core_quinielas)
        total_V = sum(q["distribución"]["V"] for q in core_quinielas)
        total_partidos = len(core_quinielas) * 14
        
        # Calcular ajustes necesarios
        target = self._calcular_distribucion_target()
        ajuste_L = target["L"] - total_L
        ajuste_V = target["V"] - total_V
        
        self.logger.debug(f"Distribución actual: L={total_L}, E={total_E}, V={total_V}")
        self.logger.debug(f"Ajustes necesarios: L{ajuste_L:+d}, V{ajuste_V:+d}")
        
        # Aplicar ajustes si son necesarios
        if abs(ajuste_L) > 0 or abs(ajuste_V) > 0:
            core_quinielas = self._aplicar_ajustes_distribucion(
                core_quinielas, partidos, anclas_indices, ajuste_L, ajuste_V
            )
        
        return core_quinielas
    
    def _aplicar_ajustes_distribucion(self, core_quinielas: List[Dict[str, Any]], 
                                    partidos: List[Dict[str, Any]], anclas_indices: List[int],
                                    ajuste_L: int, ajuste_V: int) -> List[Dict[str, Any]]:
        """Aplica ajustes específicos a la distribución"""
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        # Si necesitamos menos L y más V
        if ajuste_L < 0 and ajuste_V > 0:
            cambios_necesarios = min(abs(ajuste_L), ajuste_V)
            
            # Buscar candidatos L para cambiar a V
            candidatos = []
            for q_idx, quiniela in enumerate(core_quinielas):
                for i in modificables:
                    if quiniela["resultados"][i] == "L" and partidos[i]["prob_visitante"] > 0.20:
                        prob_L = partidos[i]["prob_local"]
                        candidatos.append((q_idx, i, prob_L))
            
            # Ordenar por menor probabilidad de L (más fáciles de cambiar)
            candidatos.sort(key=lambda x: x[2])
            
            # Realizar cambios
            for j in range(min(cambios_necesarios, len(candidatos))):
                q_idx, pos_idx, _ = candidatos[j]
                
                # Cambiar de L a V
                core_quinielas[q_idx]["resultados"][pos_idx] = "V"
                
                # Actualizar distribución
                core_quinielas[q_idx]["distribución"]["L"] -= 1
                core_quinielas[q_idx]["distribución"]["V"] += 1
                
                self.logger.debug(f"Cambiado {core_quinielas[q_idx]['id']} P{pos_idx+1}: L → V")
        
        return core_quinielas
    
    def _es_quiniela_completamente_valida(self, resultados: List[str]) -> bool:
        """
        Validación COMPLETA de quiniela individual
        """
        # Validación básica de empates
        empates = resultados.count("E")
        if not (self.empates_min <= empates <= self.empates_max):
            return False
        
        # Concentración general
        max_conc_general = max(resultados.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc_general > self.concentracion_max_general:
            return False
        
        # NUEVA: Concentración inicial ESTRICTA
        primeros_3 = resultados[:3]
        max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > self.concentracion_max_inicial:
            return False
        
        return True
    
    def _resultado_a_clave(self, resultado: str) -> str:
        """Convierte resultado a clave de probabilidad"""
        mapeo = {"L": "local", "E": "empate", "V": "visitante"}
        return mapeo.get(resultado, "local")
    
    def _ajustar_empates_con_balance(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                   anclas_indices: List[int], core_id: int) -> List[str]:
        """Ajusta empates manteniendo balance y variación"""
        q_ajustada = quiniela.copy()
        modificables = [i for i in range(14) if i not in anclas_indices]
        
        empates_actuales = q_ajustada.count("E")
        
        if empates_actuales < self.empates_min:
            necesarios = self.empates_min - empates_actuales
            candidatos = [(i, partidos[i]["prob_empate"]) for i in modificables if q_ajustada[i] != 'E']
            candidatos.sort(key=lambda x: x[1], reverse=True)
            
            # Evitar cambiar primeros 3 si ya están balanceados
            candidatos_filtrados = []
            for idx, prob in candidatos:
                if idx < 3:  # Primeros 3
                    primeros_3_temp = q_ajustada[:3].copy()
                    primeros_3_temp[idx] = 'E'
                    max_conc = max(primeros_3_temp.count(s) for s in ["L", "E", "V"]) / 3
                    if max_conc <= self.concentracion_max_inicial:
                        candidatos_filtrados.append((idx, prob))
                else:
                    candidatos_filtrados.append((idx, prob))
            
            for i in range(min(necesarios, len(candidatos_filtrados))):
                idx, _ = candidatos_filtrados[i]
                q_ajustada[idx] = 'E'
                
        elif empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            candidatos = [(i, partidos[i]["prob_empate"]) for i in modificables if q_ajustada[i] == 'E']
            candidatos.sort(key=lambda x: x[1])
            
            for i in range(min(exceso, len(candidatos))):
                idx, _ = candidatos[i]
                partido = partidos[idx]
                # Balancear entre L y V
                q_ajustada[idx] = 'V' if partido['prob_visitante'] > partido['prob_local'] else 'L'
        
        return q_ajustada
    
    def _validar_diferencias_y_balance(self, core_quinielas: List[Dict[str, Any]], anclas_indices: List[int]):
        """Valida diferencias y balance final"""
        if len(core_quinielas) < 2:
            return
        
        primera_core = core_quinielas[0]["resultados"]
        
        # Verificar anclas iguales
        for i, core in enumerate(core_quinielas[1:], 1):
            resultados_actual = core["resultados"]
            for ancla_idx in anclas_indices:
                if primera_core[ancla_idx] != resultados_actual[ancla_idx]:
                    self.logger.error(f"ANCLA DIFERENTE: Core-1[{ancla_idx}]={primera_core[ancla_idx]} != Core-{i+1}[{ancla_idx}]={resultados_actual[ancla_idx]}")
        
        # Verificar diferencias y concentración inicial
        for i, core in enumerate(core_quinielas[1:], 1):
            diferencias = sum(1 for j in range(14) if primera_core[j] != core["resultados"][j])
            
            # Verificar concentración inicial
            primeros_3 = core["resultados"][:3]
            max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3
            
            self.logger.info(f"Core-1 vs Core-{i+1}: {diferencias} diferencias, ConInicial: {max_conc_inicial:.1%}")
            
            if diferencias == 0:
                self.logger.warning(f"⚠️ Core-1 y Core-{i+1} son IDÉNTICAS")
            if max_conc_inicial > self.concentracion_max_inicial:
                self.logger.warning(f"⚠️ Core-{i+1} concentración inicial: {max_conc_inicial:.1%} > {self.concentracion_max_inicial:.1%}")
        
        # Verificar distribución global final
        total_L = sum(q["distribución"]["L"] for q in core_quinielas)
        total_E = sum(q["distribución"]["E"] for q in core_quinielas)
        total_V = sum(q["distribución"]["V"] for q in core_quinielas)
        total_partidos = len(core_quinielas) * 14
        
        porc_L = total_L / total_partidos
        porc_E = total_E / total_partidos
        porc_V = total_V / total_partidos
        
        self.logger.info(f"Distribución final: L={porc_L:.1%}, E={porc_E:.1%}, V={porc_V:.1%}")
    
    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        return max(probs, key=probs.get)
    
    def _get_resultado_segunda_opcion(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de segunda mayor probabilidad"""
        probs = {"L": partido["prob_local"], "E": partido["prob_empate"], "V": partido["prob_visitante"]}
        sorted_results = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[1][0]  # Segunda opción