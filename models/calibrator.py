# progol_optimizer/models/calibrator.py
"""
Calibrador Bayesiano - Implementación EXACTA de la página 3 del documento
Fórmula: p_final = p_raw * (1 + k1*ΔForma + k2*Lesiones + k3*Contexto) / Z
"""

import logging
import numpy as np
from typing import Dict, Any

class BayesianCalibrator:
    """
    Implementa la calibración bayesiana exacta según la metodología del documento
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from progol_optimizer.config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        # Coeficientes de calibración (página 3)
        self.k1 = self.config["CALIBRACION_COEFICIENTES"]["k1_forma"]
        self.k2 = self.config["CALIBRACION_COEFICIENTES"]["k2_lesiones"]
        self.k3 = self.config["CALIBRACION_COEFICIENTES"]["k3_contexto"]
        
        self.logger.debug(f"Coeficientes bayesianos: k1={self.k1}, k2={self.k2}, k3={self.k3}")
    
    def aplicar_calibracion_bayesiana(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementa EXACTAMENTE la fórmula de la página 3:
        p_final = p_raw * (1 + k1*ΔForma + k2*Lesiones + k3*Contexto) / Z
        
        Args:
            partido: Diccionario con datos del partido
            
        Returns:
            Dict: Partido con probabilidades calibradas
        """
        self.logger.debug(f"Calibrando partido: {partido['home']} vs {partido['away']}")
        
        # Probabilidades base (p_raw)
        p_local_raw = partido["prob_local"]
        p_empate_raw = partido["prob_empate"]
        p_visitante_raw = partido["prob_visitante"]
        
        # Factores contextuales del partido
        delta_forma = partido.get("forma_diferencia", 0)
        lesiones_impact = partido.get("lesiones_impact", 0)
        
        # Contexto binario (final, derbi, playoff)
        contexto = 0.0
        if partido.get("es_final", False):
            contexto += 1.0
        if partido.get("es_derbi", False):
            contexto += 0.5
        if partido.get("es_playoff", False):
            contexto += 0.3
        
        # Factor de ajuste según fórmula exacta
        factor_ajuste = 1 + self.k1 * delta_forma + self.k2 * lesiones_impact + self.k3 * contexto
        
        self.logger.debug(f"  Factor ajuste: {factor_ajuste:.3f} "
                         f"(forma={delta_forma:.2f}, lesiones={lesiones_impact:.2f}, contexto={contexto:.1f})")
        
        # Aplicar ajuste diferenciado
        # Local se beneficia del factor positivo, visitante se penaliza
        p_local_ajustado = p_local_raw * max(factor_ajuste, 0.3)  # Mínimo 30% para evitar colapso
        p_empate_ajustado = p_empate_raw  # No se ajusta directamente
        p_visitante_ajustado = p_visitante_raw / max(factor_ajuste, 0.5)  # Ajuste inverso
        
        # Aplicar Draw-Propensity Rule (página 3)
        p_local_ajustado, p_empate_ajustado, p_visitante_ajustado = self._aplicar_draw_propensity(
            p_local_ajustado, p_empate_ajustado, p_visitante_ajustado
        )
        
        # Factor de normalización Z
        Z = p_local_ajustado + p_empate_ajustado + p_visitante_ajustado
        
        # Probabilidades finales normalizadas
        p_local_final = p_local_ajustado / Z
        p_empate_final = p_empate_ajustado / Z
        p_visitante_final = p_visitante_ajustado / Z
        
        # Validar que suman 1.0
        suma_final = p_local_final + p_empate_final + p_visitante_final
        assert abs(suma_final - 1.0) < 0.001, f"Error normalización: suma={suma_final}"
        
        self.logger.debug(f"  Probabilidades calibradas: L={p_local_final:.3f}, "
                         f"E={p_empate_final:.3f}, V={p_visitante_final:.3f}")
        
        # Retornar partido con probabilidades calibradas
        partido_calibrado = partido.copy()
        partido_calibrado.update({
            "prob_local": p_local_final,
            "prob_empate": p_empate_final,
            "prob_visitante": p_visitante_final,
            
            # Metadatos de calibración
            "prob_local_raw": p_local_raw,
            "prob_empate_raw": p_empate_raw,
            "prob_visitante_raw": p_visitante_raw,
            "factor_ajuste": factor_ajuste,
            "Z_normalizacion": Z
        })
        
        return partido_calibrado
    
    def _aplicar_draw_propensity(self, p_local, p_empate, p_visitante):
        """
        Implementa la Draw-Propensity Rule de la página 3:
        Si |pL - pV| < 0.08 y pE > max(pL, pV) entonces boost empate +6pp
        """
        umbral_diff = self.config["DRAW_PROPENSITY"]["umbral_diferencia"]
        boost_empate = self.config["DRAW_PROPENSITY"]["boost_empate"]
        
        diferencia_lv = abs(p_local - p_visitante)
        max_lv = max(p_local, p_visitante)
        
        if diferencia_lv < umbral_diff and p_empate > max_lv:
            self.logger.debug(f"  Aplicando Draw-Propensity: diff={diferencia_lv:.3f} < {umbral_diff}")
            
            # Boost al empate
            p_empate_new = min(p_empate + boost_empate, 0.95)  # Máximo 95%
            
            # Reducir proporcionalmente local y visitante
            reduccion = (p_empate_new - p_empate) / 2
            p_local_new = max(p_local - reduccion, 0.05)  # Mínimo 5%
            p_visitante_new = max(p_visitante - reduccion, 0.05)
            
            return p_local_new, p_empate_new, p_visitante_new
        
        return p_local, p_empate, p_visitante
