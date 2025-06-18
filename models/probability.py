# progol_optimizer/models/probability.py - CORREGIDO
"""
Modelo Bivariate-Poisson (OPCIONAL - Avanzado)
Implementación de la distribución Bivariate-Poisson de la página 3
Solo necesario para implementación completa con momios de cierre
"""

import numpy as np
import math  # CORRECCIÓN: Importar math directamente
import logging
from scipy.special import comb
from typing import Tuple, Dict, Any

class BivariatePoisson:
    """
    Implementación opcional del modelo Bivariate-Poisson
    para casos avanzados con momios de cierre
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calcular_probabilidades_resultado(self, lambda1: float, lambda2: float, lambda3: float) -> Dict[str, float]:
        """
        Calcula P(L), P(E), P(V) usando Bivariate-Poisson
        
        Fórmula de la página 3:
        P(X=x, Y=y) = e^-(λ1+λ2+λ3) * λ1^x * λ2^y / (x! * y!) * Σ(k=0 to min(x,y)) λ3^k/k! * C(x,k) * C(y,k)
        
        Args:
            lambda1: Intensidad goles local
            lambda2: Intensidad goles visitante  
            lambda3: Covarianza
            
        Returns:
            Dict con probabilidades L, E, V
        """
        self.logger.debug(f"Calculando Bivariate-Poisson: λ1={lambda1:.3f}, λ2={lambda2:.3f}, λ3={lambda3:.3f}")
        
        # Límites de cálculo (raramente se superan 6 goles)
        max_goles = 8
        
        prob_matriz = np.zeros((max_goles + 1, max_goles + 1))
        
        # Calcular P(X=x, Y=y) para cada combinación
        for x in range(max_goles + 1):
            for y in range(max_goles + 1):
                prob_matriz[x, y] = self._bivariate_poisson_pmf(x, y, lambda1, lambda2, lambda3)
        
        # Sumar probabilidades por resultado
        prob_local = np.sum(prob_matriz[1:, 0])  # Local gana (x > y, empezando desde x=1, y=0)
        for x in range(2, max_goles + 1):
            for y in range(x):
                prob_local += prob_matriz[x, y]
        
        prob_empate = np.sum([prob_matriz[i, i] for i in range(max_goles + 1)])  # Empates (x = y)
        
        prob_visitante = 1.0 - prob_local - prob_empate  # V = 1 - L - E
        
        self.logger.debug(f"  Resultados: L={prob_local:.3f}, E={prob_empate:.3f}, V={prob_visitante:.3f}")
        
        return {
            "prob_local": prob_local,
            "prob_empate": prob_empate,
            "prob_visitante": prob_visitante
        }
    
    def _bivariate_poisson_pmf(self, x: int, y: int, lambda1: float, lambda2: float, lambda3: float) -> float:
        """
        Función de masa de probabilidad Bivariate-Poisson - CORREGIDA
        """
        if x < 0 or y < 0:
            return 0.0
        
        # Término exponencial
        exp_term = np.exp(-(lambda1 + lambda2 + lambda3))
        
        # Términos de potencia - CORRECCIÓN: usar math.factorial en lugar de np.math.factorial
        lambda1_term = (lambda1 ** x) / math.factorial(x)
        lambda2_term = (lambda2 ** y) / math.factorial(y)
        
        # Suma de covarianza
        min_xy = min(x, y)
        suma_cov = 0.0
        
        for k in range(min_xy + 1):
            if k <= x and k <= y:
                lambda3_term = (lambda3 ** k) / math.factorial(k)  # CORRECCIÓN: math.factorial
                comb_x = comb(x, k, exact=True)
                comb_y = comb(y, k, exact=True)
                suma_cov += lambda3_term * comb_x * comb_y
        
        resultado = exp_term * lambda1_term * lambda2_term * suma_cov
        
        return resultado