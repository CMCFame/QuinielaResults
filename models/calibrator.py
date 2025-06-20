# progol_optimizer/models/calibrator.py - VERSIÓN MEJORADA
"""
Calibrador Bayesiano MEJORADO - Regularización más agresiva y mejor control de distribución
CORRECCIÓN: Control más estricto de la distribución histórica y mejor balance L/E/V
"""

import logging
import numpy as np
from typing import Dict, Any, List

class BayesianCalibrator:
    """
    Implementa la calibración bayesiana exacta y la regularización global mejorada.
    MEJORADO: Control más estricto de distribución histórica y mejor diversidad
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.k1 = self.config["CALIBRACION_COEFICIENTES"]["k1_forma"]
        self.k2 = self.config["CALIBRACION_COEFICIENTES"]["k2_lesiones"]
        self.k3 = self.config["CALIBRACION_COEFICIENTES"]["k3_contexto"]
        self.distribucion_historica = self.config["DISTRIBUCION_HISTORICA"]
        self.rangos_historicos = self.config["RANGOS_HISTORICOS"]
        self.logger.debug(f"Calibrador MEJORADO inicializado: k1={self.k1}, k2={self.k2}, k3={self.k3}")

    def calibrar_concurso_completo(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        VERSIÓN MEJORADA: Calibra y regulariza con control más estricto de distribución
        """
        self.logger.info("Iniciando calibración completa MEJORADA del concurso...")
        
        if len(partidos) != 14:
            raise ValueError(f"Se requieren exactamente 14 partidos, recibidos: {len(partidos)}")
        
        # PASO 1: Calibración bayesiana individual
        partidos_calibrados = [self.aplicar_calibracion_bayesiana(p) for p in partidos]
        
        # PASO 2: Regularización agresiva hacia distribución histórica
        partidos_regularizados = self._aplicar_regularizacion_mejorada(partidos_calibrados)
        
        # PASO 3: Verificación y ajuste fino
        partidos_finales = self._ajuste_fino_distribución(partidos_regularizados)
        
        # Validar resultado final
        distribucion_final = self._calcular_distribucion_total(partidos_finales)
        self._validar_distribucion_final(distribucion_final)
        
        self.logger.info(f"✅ Calibración completada: L={distribucion_final['L']:.3f}, " +
                       f"E={distribucion_final['E']:.3f}, V={distribucion_final['V']:.3f}")
        
        return partidos_finales

    def aplicar_calibracion_bayesiana(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementa la fórmula de la página 3 del documento (sin cambios)
        """
        p_local_raw = partido["prob_local"]
        p_empate_raw = partido["prob_empate"]
        p_visitante_raw = partido["prob_visitante"]
        
        delta_forma = partido.get("forma_diferencia", 0)
        lesiones_impact = partido.get("lesiones_impact", 0)
        
        contexto = 0.0
        if partido.get("es_final", False): contexto += 1.0
        if partido.get("es_derbi", False): contexto += 0.5
        if partido.get("es_playoff", False): contexto += 0.3
        
        # Factor de ajuste según la fórmula exacta
        factor_ajuste = 1 + self.k1 * delta_forma + self.k2 * lesiones_impact + self.k3 * contexto
        
        # Ajuste diferenciado: el factor impacta principalmente a L y V
        p_local_ajustado = p_local_raw
        p_visitante_ajustado = p_visitante_raw
        if factor_ajuste > 1.0: # Beneficia al local
            p_local_ajustado *= factor_ajuste
        else: # Beneficia al visitante
            p_visitante_ajustado *= (1/factor_ajuste if factor_ajuste != 0 else 1)

        p_empate_ajustado = p_empate_raw
        
        # Aplicar Draw-Propensity Rule (página 3)
        p_local_ajustado, p_empate_ajustado, p_visitante_ajustado = self._aplicar_draw_propensity(
            p_local_ajustado, p_empate_ajustado, p_visitante_ajustado
        )
        
        # Normalización Z
        Z = p_local_ajustado + p_empate_ajustado + p_visitante_ajustado
        if Z == 0: Z = 1
        
        partido_calibrado = partido.copy()
        partido_calibrado.update({
            "prob_local": p_local_ajustado / Z,
            "prob_empate": p_empate_ajustado / Z,
            "prob_visitante": p_visitante_ajustado / Z,
            "prob_local_raw": p_local_raw, 
            "prob_empate_raw": p_empate_raw, 
            "prob_visitante_raw": p_visitante_raw,
            "factor_ajuste": factor_ajuste, 
            "Z_normalizacion": Z
        })
        return partido_calibrado

    def _aplicar_regularizacion_mejorada(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        VERSIÓN MEJORADA: Regularización más estricta hacia distribución histórica
        """
        self.logger.info("Aplicando regularización MEJORADA hacia distribución histórica...")
        
        # Targets históricos para 14 partidos (punto medio de los rangos)
        target_local_sum = 14 * 0.38      # 5.32
        target_empate_sum = 14 * 0.29     # 4.06
        target_visitante_sum = 14 * 0.33  # 4.62
        
        # Calcular diferencias actuales
        actual_local_sum = sum(p["prob_local"] for p in partidos)
        actual_empate_sum = sum(p["prob_empate"] for p in partidos)
        actual_visitante_sum = sum(p["prob_visitante"] for p in partidos)
        
        self.logger.debug(f"Antes regularización: L={actual_local_sum:.3f}, E={actual_empate_sum:.3f}, V={actual_visitante_sum:.3f}")
        self.logger.debug(f"Targets objetivo: L={target_local_sum:.3f}, E={target_empate_sum:.3f}, V={target_visitante_sum:.3f}")
        
        # Factores de corrección más agresivos
        factor_local = target_local_sum / max(actual_local_sum, 0.1)
        factor_empate = target_empate_sum / max(actual_empate_sum, 0.1)
        factor_visitante = target_visitante_sum / max(actual_visitante_sum, 0.1)
        
        # Aplicar corrección con smoothing para evitar cambios extremos
        smoothing = 0.7  # Factor de suavizado
        factor_local = 1.0 + smoothing * (factor_local - 1.0)
        factor_empate = 1.0 + smoothing * (factor_empate - 1.0)
        factor_visitante = 1.0 + smoothing * (factor_visitante - 1.0)
        
        self.logger.debug(f"Factores de corrección: L={factor_local:.3f}, E={factor_empate:.3f}, V={factor_visitante:.3f}")
        
        partidos_corregidos = []
        for i, partido in enumerate(partidos):
            p_local_corregido = partido["prob_local"] * factor_local
            p_empate_corregido = partido["prob_empate"] * factor_empate
            p_visitante_corregido = partido["prob_visitante"] * factor_visitante
            
            # Normalizar cada partido para que sus probabilidades sumen 1
            total_corregido = p_local_corregido + p_empate_corregido + p_visitante_corregido
            if total_corregido == 0: total_corregido = 1

            # Aplicar límites mínimos para evitar probabilidades demasiado bajas
            p_local_final = max(0.05, p_local_corregido / total_corregido)
            p_empate_final = max(0.05, p_empate_corregido / total_corregido)
            p_visitante_final = max(0.05, p_visitante_corregido / total_corregido)
            
            # Re-normalizar después de aplicar límites
            total_final = p_local_final + p_empate_final + p_visitante_final
            
            partido_corregido = partido.copy()
            partido_corregido.update({
                "prob_local": p_local_final / total_final,
                "prob_empate": p_empate_final / total_final,
                "prob_visitante": p_visitante_final / total_final,
                "regularizado": True,
                "factor_local_aplicado": factor_local,
                "factor_empate_aplicado": factor_empate,
                "factor_visitante_aplicado": factor_visitante
            })
            partidos_corregidos.append(partido_corregido)
        
        return partidos_corregidos

    def _ajuste_fino_distribución(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        NUEVO: Ajuste fino para asegurar que estamos dentro de los rangos históricos exactos
        """
        self.logger.info("Aplicando ajuste fino de distribución...")
        
        partidos_ajustados = partidos.copy()
        max_iteraciones = 10
        
        for iteracion in range(max_iteraciones):
            distribucion = self._calcular_distribucion_total(partidos_ajustados)
            
            # Verificar si estamos en rangos válidos
            if self._esta_en_rangos_historicos(distribucion):
                self.logger.debug(f"✅ Distribución en rangos correctos en iteración {iteracion}")
                break
            
            # Aplicar correcciones específicas
            partidos_ajustados = self._corregir_fuera_de_rangos(partidos_ajustados, distribucion)
            
        return partidos_ajustados

    def _esta_en_rangos_historicos(self, distribucion: Dict[str, float]) -> bool:
        """Verifica si la distribución está en los rangos históricos válidos"""
        sum_L = distribucion["L"]
        sum_E = distribucion["E"]
        sum_V = distribucion["V"]
        
        # Rangos para 14 partidos
        rango_L = [14 * self.rangos_historicos["L"][0], 14 * self.rangos_historicos["L"][1]]
        rango_E = [14 * self.rangos_historicos["E"][0], 14 * self.rangos_historicos["E"][1]]
        rango_V = [14 * self.rangos_historicos["V"][0], 14 * self.rangos_historicos["V"][1]]
        
        return (rango_L[0] <= sum_L <= rango_L[1] and 
                rango_E[0] <= sum_E <= rango_E[1] and 
                rango_V[0] <= sum_V <= rango_V[1])

    def _corregir_fuera_de_rangos(self, partidos: List[Dict[str, Any]], distribucion: Dict[str, float]) -> List[Dict[str, Any]]:
        """Corrige distribuciones que están fuera de rangos históricos"""
        partidos_corregidos = []
        
        # Calcular qué tan fuera de rango estamos
        sum_L, sum_E, sum_V = distribucion["L"], distribucion["E"], distribucion["V"]
        
        target_L = 14 * 0.38
        target_E = 14 * 0.29
        target_V = 14 * 0.33
        
        # Calcular ajustes necesarios
        ajuste_L = (target_L - sum_L) / 14  # Ajuste por partido
        ajuste_E = (target_E - sum_E) / 14
        ajuste_V = (target_V - sum_V) / 14
        
        for partido in partidos:
            # Aplicar micro-ajustes
            nuevo_local = max(0.05, partido["prob_local"] + ajuste_L * 0.1)
            nuevo_empate = max(0.05, partido["prob_empate"] + ajuste_E * 0.1)
            nuevo_visitante = max(0.05, partido["prob_visitante"] + ajuste_V * 0.1)
            
            # Normalizar
            total = nuevo_local + nuevo_empate + nuevo_visitante
            
            partido_corregido = partido.copy()
            partido_corregido.update({
                "prob_local": nuevo_local / total,
                "prob_empate": nuevo_empate / total,
                "prob_visitante": nuevo_visitante / total,
                "ajuste_fino_aplicado": True
            })
            partidos_corregidos.append(partido_corregido)
        
        return partidos_corregidos

    def _validar_distribucion_final(self, distribucion: Dict[str, float]):
        """Valida que la distribución final sea válida"""
        sum_L, sum_E, sum_V = distribucion["L"], distribucion["E"], distribucion["V"]
        
        if not self._esta_en_rangos_historicos(distribucion):
            self.logger.warning(f"⚠️ Distribución final fuera de rangos históricos:")
            self.logger.warning(f"  L: {sum_L:.3f} (esperado: {14*self.rangos_historicos['L'][0]:.3f}-{14*self.rangos_historicos['L'][1]:.3f})")
            self.logger.warning(f"  E: {sum_E:.3f} (esperado: {14*self.rangos_historicos['E'][0]:.3f}-{14*self.rangos_historicos['E'][1]:.3f})")
            self.logger.warning(f"  V: {sum_V:.3f} (esperado: {14*self.rangos_historicos['V'][0]:.3f}-{14*self.rangos_historicos['V'][1]:.3f})")

    def _calcular_distribucion_total(self, partidos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula la suma total de probabilidades del concurso."""
        return {
            "L": sum(p["prob_local"] for p in partidos),
            "E": sum(p["prob_empate"] for p in partidos),
            "V": sum(p["prob_visitante"] for p in partidos)
        }

    def _aplicar_draw_propensity(self, p_local, p_empate, p_visitante):
        """
        Implementa la Draw-Propensity Rule de la página 3 (sin cambios)
        """
        umbral_diff = self.config["DRAW_PROPENSITY"]["umbral_diferencia"]
        boost_empate = self.config["DRAW_PROPENSITY"]["boost_empate"]
        
        if abs(p_local - p_visitante) < umbral_diff and p_empate > max(p_local, p_visitante):
            self.logger.debug(f"  Aplicando Draw-Propensity Rule...")
            p_empate_new = min(p_empate + boost_empate, 0.95)
            reduccion = (p_empate_new - p_empate)
            # Reducir de L y V en proporción a su probabilidad
            total_lv = p_local + p_visitante
            if total_lv > 0:
                p_local_new = max(p_local - reduccion * (p_local/total_lv), 0.05)
                p_visitante_new = max(p_visitante - reduccion * (p_visitante/total_lv), 0.05)
            else:
                p_local_new, p_visitante_new = p_local, p_visitante
            return p_local_new, p_empate_new, p_visitante_new
        
        return p_local, p_empate, p_visitante