# progol_optimizer/models/calibrator.py
"""
Calibrador Bayesiano - VERSIÓN CORREGIDA CON REGULARIZACIÓN AGRESIVA
Fórmula: p_final = p_raw * (1 + k1*ΔForma + k2*Lesiones + k3*Contexto) / Z
CORRECCIÓN CRÍTICA: Regularización agresiva hacia distribución histórica 38%L, 29%E, 33%V
"""

import logging
import numpy as np
from typing import Dict, Any, List

class BayesianCalibrator:
    """
    Implementa la calibración bayesiana exacta y la regularización global.
    CORREGIDO: Regularización agresiva para forzar distribución histórica
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        self.k1 = self.config["CALIBRACION_COEFICIENTES"]["k1_forma"]
        self.k2 = self.config["CALIBRACION_COEFICIENTES"]["k2_lesiones"]
        self.k3 = self.config["CALIBRACION_COEFICIENTES"]["k3_contexto"]
        self.distribucion_historica = self.config["DISTRIBUCION_HISTORICA"]
        self.logger.debug(f"Coeficientes bayesianos: k1={self.k1}, k2={self.k2}, k3={self.k3}")

    def calibrar_concurso_completo(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        NUEVA FUNCIÓN: Calibra todos los partidos y luego aplica regularización global.
        Garantiza que la distribución final del concurso respete los priors históricos.
        """
        self.logger.info("Iniciando calibración completa del concurso...")
        
        # PASO 1: Calibración bayesiana individual de cada partido
        partidos_calibrados_individualmente = [self.aplicar_calibracion_bayesiana(p) for p in partidos]
        
        # PASO 2: Regularización global para ajustar la suma total de probabilidades
        partidos_regularizados = self._aplicar_regularizacion_agresiva(partidos_calibrados_individualmente)
        
        distribucion_final = self._calcular_distribucion_total(partidos_regularizados)
        self.logger.info(f"Distribución final tras regularización: L={distribucion_final['L']:.3f}, " +
                       f"E={distribucion_final['E']:.3f}, V={distribucion_final['V']:.3f}")
        
        return partidos_regularizados

    def aplicar_calibracion_bayesiana(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementa la fórmula de la página 3 del documento.
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
            "prob_local_raw": p_local_raw, "prob_empate_raw": p_empate_raw, "prob_visitante_raw": p_visitante_raw,
            "factor_ajuste": factor_ajuste, "Z_normalizacion": Z
        })
        return partido_calibrado

    def _aplicar_regularizacion_agresiva(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aplica regularización para forzar la suma de probabilidades del concurso
        a alinearse con la distribución histórica.
        """
        self.logger.info("Aplicando regularización agresiva hacia distribución histórica...")
        
        # Targets históricos para 14 partidos
        target_local_sum = 14 * self.distribucion_historica["L"]      # ~5.32
        target_empate_sum = 14 * self.distribucion_historica["E"]     # ~4.06
        target_visitante_sum = 14 * self.distribucion_historica["V"]  # ~4.62
        
        # Sumas actuales
        actual_local_sum = sum(p["prob_local"] for p in partidos)
        actual_empate_sum = sum(p["prob_empate"] for p in partidos)
        actual_visitante_sum = sum(p["prob_visitante"] for p in partidos)
        
        # Factores de corrección
        factor_local = target_local_sum / max(actual_local_sum, 0.1)
        factor_empate = target_empate_sum / max(actual_empate_sum, 0.1)
        factor_visitante = target_visitante_sum / max(actual_visitante_sum, 0.1)
        
        partidos_corregidos = []
        for partido in partidos:
            p_local_corregido = partido["prob_local"] * factor_local
            p_empate_corregido = partido["prob_empate"] * factor_empate
            p_visitante_corregido = partido["prob_visitante"] * factor_visitante
            
            # Normalizar cada partido para que sus probabilidades sumen 1
            total_corregido = p_local_corregido + p_empate_corregido + p_visitante_corregido
            if total_corregido == 0: total_corregido = 1

            partido_corregido = partido.copy()
            partido_corregido.update({
                "prob_local": p_local_corregido / total_corregido,
                "prob_empate": p_empate_corregido / total_corregido,
                "prob_visitante": p_visitante_corregido / total_corregido,
                "regularizado": True
            })
            partidos_corregidos.append(partido_corregido)
        
        return partidos_corregidos

    def _calcular_distribucion_total(self, partidos: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula la suma total de probabilidades del concurso."""
        return {
            "L": sum(p["prob_local"] for p in partidos),
            "E": sum(p["prob_empate"] for p in partidos),
            "V": sum(p["prob_visitante"] for p in partidos)
        }

    def _aplicar_draw_propensity(self, p_local, p_empate, p_visitante):
        """
        Implementa la Draw-Propensity Rule de la página 3.
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