# progol_optimizer/models/calibrator.py
"""
Calibrador Bayesiano - Implementación COMPLETA con Bivariate-Poisson integrado
Fórmula: p_final = p_raw * (1 + k1*ΔForma + k2*Lesiones + k3*Contexto) / Z
+ Stacking con Bivariate-Poisson + Ajuste diferenciado
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple

class BayesianCalibrator:
    """
    Implementa la calibración bayesiana exacta + Bivariate-Poisson integrado
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        # Coeficientes de calibración (página 3)
        self.k1 = self.config["CALIBRACION_COEFICIENTES"]["k1_forma"]
        self.k2 = self.config["CALIBRACION_COEFICIENTES"]["k2_lesiones"]
        self.k3 = self.config["CALIBRACION_COEFICIENTES"]["k3_contexto"]
        
        # Pesos de stacking del documento (página 3)
        self.w_raw = 0.58  # Peso del mercado
        self.w_pois = 0.42  # Peso Bivariate-Poisson
        
        # Inicializar Bivariate-Poisson
        self.usar_bivariate_poisson = True
        try:
            from models.probability import BivariatePoisson
            self.poisson_model = BivariatePoisson()
            self.logger.info("✅ Bivariate-Poisson disponible y cargado")
        except ImportError:
            self.usar_bivariate_poisson = False
            self.logger.warning("⚠️ Bivariate-Poisson no disponible, usando solo calibración bayesiana")
        
        # Sistema Elo para λ1, λ2
        self.elo_system = self._inicializar_elo()
        
        self.logger.debug(f"Coeficientes bayesianos: k1={self.k1}, k2={self.k2}, k3={self.k3}")
        self.logger.debug(f"Pesos stacking: mercado={self.w_raw}, poisson={self.w_pois}")
    
    def aplicar_calibracion_bayesiana(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        CALIBRACIÓN COMPLETA: Bayesiana + Bivariate-Poisson + Stacking
        
        Args:
            partido: Diccionario con datos del partido
            
        Returns:
            Dict: Partido con probabilidades calibradas finales
        """
        self.logger.debug(f"Calibrando partido: {partido['home']} vs {partido['away']}")
        
        # PASO 1: Calibración Bayesiana (método original mejorado)
        partido_bayesiano = self._aplicar_calibracion_bayesiana_mejorada(partido)
        
        # PASO 2: Modelo Bivariate-Poisson (NUEVO)
        if self.usar_bivariate_poisson:
            probs_poisson = self._calcular_probabilidades_poisson(partido)
            
            # PASO 3: Stacking según pesos del documento
            partido_final = self._fusionar_probabilidades(
                partido_bayesiano, probs_poisson
            )
        else:
            partido_final = partido_bayesiano
        
        # PASO 4: Draw-Propensity Rule (sin cambios)
        partido_final = self._aplicar_draw_propensity_completo(partido_final)
        
        return partido_final
    
    def _aplicar_calibracion_bayesiana_mejorada(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibración bayesiana con AJUSTE DIFERENCIADO del documento
        """
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
        
        # MEJORA CRÍTICA: Ajuste diferenciado del documento
        # Local se beneficia del factor positivo, visitante se penaliza
        p_local_ajustado = p_local_raw * max(factor_ajuste, 0.3)  # Mínimo 30%
        p_empate_ajustado = p_empate_raw  # No se ajusta directamente
        p_visitante_ajustado = p_visitante_raw / max(factor_ajuste, 0.5)  # Ajuste inverso
        
        # Factor de normalización Z
        Z = p_local_ajustado + p_empate_ajustado + p_visitante_ajustado
        
        # Probabilidades finales normalizadas
        p_local_final = p_local_ajustado / Z
        p_empate_final = p_empate_ajustado / Z
        p_visitante_final = p_visitante_ajustado / Z
        
        # Validar que suman 1.0
        suma_final = p_local_final + p_empate_final + p_visitante_final
        assert abs(suma_final - 1.0) < 0.001, f"Error normalización: suma={suma_final}"
        
        self.logger.debug(f"  Probabilidades bayesianas: L={p_local_final:.3f}, "
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
    
    def _calcular_probabilidades_poisson(self, partido: Dict[str, Any]) -> Dict[str, float]:
        """
        NUEVO: Calcula probabilidades usando Bivariate-Poisson
        """
        # Obtener parámetros λ1, λ2, λ3
        lambda1, lambda2, lambda3 = self._calcular_lambdas_poisson(partido)
        
        # Calcular probabilidades L/E/V
        probs_poisson = self.poisson_model.calcular_probabilidades_resultado(
            lambda1, lambda2, lambda3
        )
        
        self.logger.debug(f"  Probabilidades Poisson: L={probs_poisson['prob_local']:.3f}, "
                         f"E={probs_poisson['prob_empate']:.3f}, V={probs_poisson['prob_visitante']:.3f}")
        
        return probs_poisson
    
    def _calcular_lambdas_poisson(self, partido: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Calcula λ1, λ2, λ3 para Bivariate-Poisson usando Elo + contexto
        """
        # Obtener ratings Elo (o usar valores por defecto)
        rating_local = self.elo_system.get(partido['home'], 1500)
        rating_visitante = self.elo_system.get(partido['away'], 1500)
        
        # Fórmula del documento: log λ = μ + α_H - β_A + γ(Elo_H - Elo_A) + δ*factor_local
        diferencia_elo = (rating_local - rating_visitante) / 400  # Normalizar
        
        # Parámetros base (calibrados empíricamente según documento)
        mu = 0.3  # Base de goles
        gamma = 0.15  # Sensibilidad a Elo
        home_factor = 0.1  # Ventaja local
        
        # Ajustar por liga (diferentes factores locales)
        liga = partido.get('liga', 'Liga MX')
        factor_local_liga = self._obtener_factor_local_liga(liga)
        
        lambda1 = np.exp(mu + home_factor + factor_local_liga + gamma * diferencia_elo)
        lambda2 = np.exp(mu - gamma * diferencia_elo)
        
        # λ3 (covarianza) basado en contexto del partido
        lambda3 = 0.1  # Base
        if partido.get("es_final", False):
            lambda3 += 0.05  # Finales más impredecibles
        if partido.get("es_derbi", False):
            lambda3 += 0.03  # Derbis más volátiles
            
        # Asegurar λ3 ≤ min(λ1, λ2) para validez matemática
        lambda3 = min(lambda3, min(lambda1, lambda2) * 0.8)
        
        return lambda1, lambda2, lambda3
    
    def _obtener_factor_local_liga(self, liga: str) -> float:
        """
        Factores de localía por liga según el documento
        """
        factores = {
            'Liga MX': 0.45,
            'Premier League': 0.35,
            'La Liga': 0.40,
            'Serie A': 0.42,
            'Bundesliga': 0.38,
            'MLS': 0.55,
            'Liga Brasileira': 0.55,
        }
        return factores.get(liga, 0.45)  # Default Liga MX
    
    def _fusionar_probabilidades(self, partido_bayesiano: Dict[str, Any], 
                                probs_poisson: Dict[str, float]) -> Dict[str, Any]:
        """
        Stacking según pesos del documento: 58% mercado, 42% Poisson
        """
        # Extraer probabilidades bayesianas
        p_bay = [
            partido_bayesiano["prob_local"],
            partido_bayesiano["prob_empate"],
            partido_bayesiano["prob_visitante"]
        ]
        
        # Extraer probabilidades Poisson
        p_pois = [
            probs_poisson["prob_local"],
            probs_poisson["prob_empate"],
            probs_poisson["prob_visitante"]
        ]
        
        # Stacking lineal con pesos del documento
        p_final = [
            self.w_raw * p_bay[i] + self.w_pois * p_pois[i]
            for i in range(3)
        ]
        
        # Normalizar para asegurar suma = 1
        suma = sum(p_final)
        p_final = [p / suma for p in p_final]
        
        self.logger.debug(f"  Probabilidades fusionadas: L={p_final[0]:.3f}, "
                         f"E={p_final[1]:.3f}, V={p_final[2]:.3f}")
        
        # Actualizar partido con probabilidades finales
        partido_final = partido_bayesiano.copy()
        partido_final.update({
            "prob_local": p_final[0],
            "prob_empate": p_final[1],
            "prob_visitante": p_final[2],
            
            # Metadatos adicionales
            "prob_local_poisson": probs_poisson["prob_local"],
            "prob_empate_poisson": probs_poisson["prob_empate"],
            "prob_visitante_poisson": probs_poisson["prob_visitante"],
            "peso_mercado": self.w_raw,
            "peso_poisson": self.w_pois
        })
        
        return partido_final
    
    def _aplicar_draw_propensity_completo(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draw-Propensity Rule completa del documento
        """
        p_local = partido["prob_local"]
        p_empate = partido["prob_empate"]
        p_visitante = partido["prob_visitante"]
        
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
            
            # Normalizar
            suma = p_local_new + p_empate_new + p_visitante_new
            
            partido["prob_local"] = p_local_new / suma
            partido["prob_empate"] = p_empate_new / suma
            partido["prob_visitante"] = p_visitante_new / suma
            partido["draw_propensity_aplicada"] = True
        else:
            partido["draw_propensity_aplicada"] = False
        
        return partido
    
    def _inicializar_elo(self) -> Dict[str, float]:
        """
        Inicializa sistema Elo básico
        TODO: Integrar con sistema Elo completo cuando esté disponible
        """
        # Por ahora, ratings básicos por liga
        equipos_ratings = {
            # Liga MX
            'América': 1650, 'Chivas': 1600, 'Cruz Azul': 1620, 'Pumas': 1580,
            'Monterrey': 1640, 'Tigres': 1630, 'Atlas': 1550, 'Santos': 1540,
            
            # Premier League
            'Manchester City': 1800, 'Arsenal': 1750, 'Liverpool': 1780, 'Chelsea': 1720,
            
            # La Liga
            'Real Madrid': 1820, 'Barcelona': 1800, 'Atlético Madrid': 1720,
            
            # Champions League
            'PSG': 1760, 'Bayern': 1790, 'Inter': 1700,
        }
        
        return equipos_ratings