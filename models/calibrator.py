# progol_optimizer/models/calibrator.py - CORREGIDO
"""
Calibrador Bayesiano CORREGIDO con regularización global para evitar sesgos extremos
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, List

class BayesianCalibrator:
    """
    Implementa la calibración bayesiana exacta + regularización global
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        # Coeficientes de calibración (página 3) - REDUCIDOS para evitar sesgos extremos
        self.k1 = self.config["CALIBRACION_COEFICIENTES"]["k1_forma"] * 0.7  # Reducido
        self.k2 = self.config["CALIBRACION_COEFICIENTES"]["k2_lesiones"] * 0.7  # Reducido  
        self.k3 = self.config["CALIBRACION_COEFICIENTES"]["k3_contexto"] * 0.7  # Reducido
        
        # Pesos de stacking del documento (página 3)
        self.w_raw = 0.58  # Peso del mercado
        self.w_pois = 0.42  # Peso Bivariate-Poisson
        
        # NUEVO: Acumuladores para regularización global
        self.partidos_calibrados = []
        
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
        
        self.logger.debug(f"Coeficientes bayesianos REDUCIDOS: k1={self.k1}, k2={self.k2}, k3={self.k3}")
        self.logger.debug(f"Pesos stacking: mercado={self.w_raw}, poisson={self.w_pois}")
    
    def calibrar_concurso_completo(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        NUEVO: Calibra todo el concurso aplicando regularización global
        """
        self.logger.info("🎯 Calibrando concurso completo con regularización global...")
        
        # PASO 1: Calibración individual
        partidos_calibrados = []
        for partido in partidos:
            partido_calibrado = self._aplicar_calibracion_individual(partido)
            partidos_calibrados.append(partido_calibrado)
        
        # PASO 2: Verificar distribución global y regularizar si es necesario
        partidos_regularizados = self._aplicar_regularizacion_global(partidos_calibrados)
        
        self.logger.info("✅ Calibración completa con regularización aplicada")
        return partidos_regularizados
    
    def aplicar_calibracion_bayesiana(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método individual (mantiene compatibilidad)
        """
        return self._aplicar_calibracion_individual(partido)
    
    def _aplicar_calibracion_individual(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        CALIBRACIÓN INDIVIDUAL: Bayesiana + Bivariate-Poisson + Stacking
        """
        self.logger.debug(f"Calibrando partido: {partido['home']} vs {partido['away']}")
        
        # PASO 1: Calibración Bayesiana (método mejorado)
        partido_bayesiano = self._aplicar_calibracion_bayesiana_controlada(partido)
        
        # PASO 2: Modelo Bivariate-Poisson (si está disponible)
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
    
    def _aplicar_calibracion_bayesiana_controlada(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibración bayesiana CONTROLADA para evitar sesgos extremos
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
        
        # Factor de ajuste según fórmula exacta - LIMITADO para evitar sesgos
        factor_ajuste_raw = 1 + self.k1 * delta_forma + self.k2 * lesiones_impact + self.k3 * contexto
        
        # CORRECCIÓN CRÍTICA: Limitar factor de ajuste para evitar sesgos extremos
        factor_ajuste = max(0.7, min(1.4, factor_ajuste_raw))  # Límite ±40%
        
        self.logger.debug(f"  Factor ajuste: {factor_ajuste:.3f} (raw={factor_ajuste_raw:.3f}) "
                         f"(forma={delta_forma:.2f}, lesiones={lesiones_impact:.2f}, contexto={contexto:.1f})")
        
        # MEJORA: Ajuste diferenciado CONTROLADO
        if factor_ajuste > 1.0:
            # Factor positivo: beneficia local moderadamente
            ajuste_local = 1.0 + (factor_ajuste - 1.0) * 0.6  # Solo 60% del ajuste
            ajuste_visitante = 1.0 / (1.0 + (factor_ajuste - 1.0) * 0.4)  # Solo 40% penalización
        else:
            # Factor negativo: beneficia visitante moderadamente  
            ajuste_local = 1.0 / (1.0 + (1.0 - factor_ajuste) * 0.4)  # Penalización moderada
            ajuste_visitante = 1.0 + (1.0 - factor_ajuste) * 0.6  # Beneficio moderado
        
        p_local_ajustado = p_local_raw * ajuste_local
        p_empate_ajustado = p_empate_raw  # Empate no se ajusta directamente
        p_visitante_ajustado = p_visitante_raw * ajuste_visitante
        
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
    
    def _aplicar_regularizacion_global(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        NUEVO: Aplica regularización global para forzar distribución histórica
        """
        # Calcular distribución actual
        total_prob_L = sum(p['prob_local'] for p in partidos)
        total_prob_E = sum(p['prob_empate'] for p in partidos)
        total_prob_V = sum(p['prob_visitante'] for p in partidos)
        
        total_actual = total_prob_L + total_prob_E + total_prob_V
        dist_actual = {
            'L': total_prob_L / total_actual,
            'E': total_prob_E / total_actual,
            'V': total_prob_V / total_actual
        }
        
        # Distribución objetivo del documento
        dist_objetivo = self.config["DISTRIBUCION_HISTORICA"]
        
        self.logger.debug(f"Distribución actual: L={dist_actual['L']:.3f}, E={dist_actual['E']:.3f}, V={dist_actual['V']:.3f}")
        self.logger.debug(f"Distribución objetivo: L={dist_objetivo['L']:.3f}, E={dist_objetivo['E']:.3f}, V={dist_objetivo['V']:.3f}")
        
        # Verificar si necesita regularización
        desviacion_L = abs(dist_actual['L'] - dist_objetivo['L'])
        desviacion_E = abs(dist_actual['E'] - dist_objetivo['E'])
        desviacion_V = abs(dist_actual['V'] - dist_objetivo['V'])
        
        max_desviacion = max(desviacion_L, desviacion_E, desviacion_V)
        
        if max_desviacion > 0.15:  # Si desviación > 15pp, regularizar
            self.logger.warning(f"⚠️ Distribución desbalanceada (max desv: {max_desviacion:.1%}), aplicando regularización...")
            return self._regularizar_hacia_distribucion_historica(partidos, dist_actual, dist_objetivo)
        else:
            self.logger.debug("✅ Distribución dentro de rangos aceptables")
            return partidos
    
    def _regularizar_hacia_distribucion_historica(self, partidos: List[Dict[str, Any]], 
                                                 dist_actual: Dict[str, float], 
                                                 dist_objetivo: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Regulariza las probabilidades hacia la distribución histórica
        """
        partidos_regularizados = []
        
        # Calcular factores de corrección
        factor_L = dist_objetivo['L'] / dist_actual['L'] if dist_actual['L'] > 0 else 1.0
        factor_E = dist_objetivo['E'] / dist_actual['E'] if dist_actual['E'] > 0 else 1.0
        factor_V = dist_objetivo['V'] / dist_actual['V'] if dist_actual['V'] > 0 else 1.0
        
        # Limitar factores para evitar cambios extremos
        factor_L = max(0.7, min(1.4, factor_L))
        factor_E = max(0.7, min(1.4, factor_E))
        factor_V = max(0.7, min(1.4, factor_V))
        
        self.logger.debug(f"Factores de regularización: L={factor_L:.3f}, E={factor_E:.3f}, V={factor_V:.3f}")
        
        for partido in partidos:
            # Aplicar factores de corrección
            p_local_reg = partido['prob_local'] * factor_L
            p_empate_reg = partido['prob_empate'] * factor_E
            p_visitante_reg = partido['prob_visitante'] * factor_V
            
            # Normalizar
            total_reg = p_local_reg + p_empate_reg + p_visitante_reg
            
            partido_regularizado = partido.copy()
            partido_regularizado.update({
                'prob_local': p_local_reg / total_reg,
                'prob_empate': p_empate_reg / total_reg,
                'prob_visitante': p_visitante_reg / total_reg,
                'regularizacion_aplicada': True,
                'factor_regularizacion_L': factor_L,
                'factor_regularizacion_E': factor_E,
                'factor_regularizacion_V': factor_V
            })
            
            partidos_regularizados.append(partido_regularizado)
        
        # Verificar distribución final
        total_final_L = sum(p['prob_local'] for p in partidos_regularizados)
        total_final_E = sum(p['prob_empate'] for p in partidos_regularizados)
        total_final_V = sum(p['prob_visitante'] for p in partidos_regularizados)
        total_final = total_final_L + total_final_E + total_final_V
        
        dist_final = {
            'L': total_final_L / total_final,
            'E': total_final_E / total_final,
            'V': total_final_V / total_final
        }
        
        self.logger.info(f"✅ Distribución regularizada: L={dist_final['L']:.3f}, E={dist_final['E']:.3f}, V={dist_final['V']:.3f}")
        
        return partidos_regularizados
    
    # ... [Resto de métodos sin cambios: _calcular_probabilidades_poisson, _fusionar_probabilidades, etc.]
    def _calcular_probabilidades_poisson(self, partido: Dict[str, Any]) -> Dict[str, float]:
        """Calcula probabilidades usando Bivariate-Poisson"""
        lambda1, lambda2, lambda3 = self._calcular_lambdas_poisson(partido)
        probs_poisson = self.poisson_model.calcular_probabilidades_resultado(lambda1, lambda2, lambda3)
        self.logger.debug(f"  Probabilidades Poisson: L={probs_poisson['prob_local']:.3f}, "
                         f"E={probs_poisson['prob_empate']:.3f}, V={probs_poisson['prob_visitante']:.3f}")
        return probs_poisson
    
    def _calcular_lambdas_poisson(self, partido: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calcula λ1, λ2, λ3 para Bivariate-Poisson usando Elo + contexto"""
        rating_local = self.elo_system.get(partido['home'], 1500)
        rating_visitante = self.elo_system.get(partido['away'], 1500)
        diferencia_elo = (rating_local - rating_visitante) / 400
        
        mu = 0.3
        gamma = 0.15
        home_factor = 0.1
        liga = partido.get('liga', 'Liga MX')
        factor_local_liga = self._obtener_factor_local_liga(liga)
        
        lambda1 = np.exp(mu + home_factor + factor_local_liga + gamma * diferencia_elo)
        lambda2 = np.exp(mu - gamma * diferencia_elo)
        
        lambda3 = 0.1
        if partido.get("es_final", False):
            lambda3 += 0.05
        if partido.get("es_derbi", False):
            lambda3 += 0.03
        lambda3 = min(lambda3, min(lambda1, lambda2) * 0.8)
        
        return lambda1, lambda2, lambda3
    
    def _obtener_factor_local_liga(self, liga: str) -> float:
        """Factores de localía por liga según el documento"""
        factores = {
            'Liga MX': 0.45, 'Premier League': 0.35, 'La Liga': 0.40,
            'Serie A': 0.42, 'Bundesliga': 0.38, 'MLS': 0.55, 'Liga Brasileira': 0.55,
        }
        return factores.get(liga, 0.45)
    
    def _fusionar_probabilidades(self, partido_bayesiano: Dict[str, Any], probs_poisson: Dict[str, float]) -> Dict[str, Any]:
        """Stacking según pesos del documento: 58% mercado, 42% Poisson"""
        p_bay = [partido_bayesiano["prob_local"], partido_bayesiano["prob_empate"], partido_bayesiano["prob_visitante"]]
        p_pois = [probs_poisson["prob_local"], probs_poisson["prob_empate"], probs_poisson["prob_visitante"]]
        
        p_final = [self.w_raw * p_bay[i] + self.w_pois * p_pois[i] for i in range(3)]
        suma = sum(p_final)
        p_final = [p / suma for p in p_final]
        
        self.logger.debug(f"  Probabilidades fusionadas: L={p_final[0]:.3f}, E={p_final[1]:.3f}, V={p_final[2]:.3f}")
        
        partido_final = partido_bayesiano.copy()
        partido_final.update({
            "prob_local": p_final[0], "prob_empate": p_final[1], "prob_visitante": p_final[2],
            "prob_local_poisson": probs_poisson["prob_local"], "prob_empate_poisson": probs_poisson["prob_empate"],
            "prob_visitante_poisson": probs_poisson["prob_visitante"], "peso_mercado": self.w_raw, "peso_poisson": self.w_pois
        })
        return partido_final
    
    def _aplicar_draw_propensity_completo(self, partido: Dict[str, Any]) -> Dict[str, Any]:
        """Draw-Propensity Rule completa del documento"""
        p_local = partido["prob_local"]
        p_empate = partido["prob_empate"]
        p_visitante = partido["prob_visitante"]
        
        umbral_diff = self.config["DRAW_PROPENSITY"]["umbral_diferencia"]
        boost_empate = self.config["DRAW_PROPENSITY"]["boost_empate"]
        
        diferencia_lv = abs(p_local - p_visitante)
        max_lv = max(p_local, p_visitante)
        
        if diferencia_lv < umbral_diff and p_empate > max_lv:
            self.logger.debug(f"  Aplicando Draw-Propensity: diff={diferencia_lv:.3f} < {umbral_diff}")
            
            p_empate_new = min(p_empate + boost_empate, 0.95)
            reduccion = (p_empate_new - p_empate) / 2
            p_local_new = max(p_local - reduccion, 0.05)
            p_visitante_new = max(p_visitante - reduccion, 0.05)
            
            suma = p_local_new + p_empate_new + p_visitante_new
            partido["prob_local"] = p_local_new / suma
            partido["prob_empate"] = p_empate_new / suma
            partido["prob_visitante"] = p_visitante_new / suma
            partido["draw_propensity_aplicada"] = True
        else:
            partido["draw_propensity_aplicada"] = False
        
        return partido
    
    def _inicializar_elo(self) -> Dict[str, float]:
        """Inicializa sistema Elo básico"""
        equipos_ratings = {
            'América': 1650, 'Chivas': 1600, 'Cruz Azul': 1620, 'Pumas': 1580,
            'Monterrey': 1640, 'Tigres': 1630, 'Atlas': 1550, 'Santos': 1540,
            'Manchester City': 1800, 'Arsenal': 1750, 'Liverpool': 1780, 'Chelsea': 1720,
            'Real Madrid': 1820, 'Barcelona': 1800, 'Atlético Madrid': 1720,
            'PSG': 1760, 'Bayern': 1790, 'Inter': 1700,
        }
        return equipos_ratings