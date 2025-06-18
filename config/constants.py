# progol_optimizer/config/constants.py
"""
Configuración exacta extraída del documento técnico
Página 2: Distribución histórica de 1,497 concursos
Página 3: Coeficientes de calibración bayesiana
Página 4: Umbrales de clasificación de partidos
"""

PROGOL_CONFIG = {
    # Página 2 del documento - distribución histórica 1,497 concursos
    "DISTRIBUCION_HISTORICA": {"L": 0.38, "E": 0.29, "V": 0.33},
    "RANGOS_HISTORICOS": {
        "L": [0.35, 0.41],  # 35-41% locales
        "E": [0.25, 0.33],  # 25-33% empates  
        "V": [0.30, 0.36]   # 30-36% visitantes
    },
    "EMPATES_PROMEDIO": 4.33,
    "EMPATES_MIN": 4,
    "EMPATES_MAX": 6,
    "CONCENTRACION_MAX_GENERAL": 0.70,      # ≤70% mismo signo
    "CONCENTRACION_MAX_INICIAL": 0.60,      # ≤60% en partidos 1-3
    
    # Página 3 - Calibración Bayesiana k1, k2, k3
    "CALIBRACION_COEFICIENTES": {
        "k1_forma": 0.15,
        "k2_lesiones": 0.10, 
        "k3_contexto": 0.20
    },
    
    # Página 3 - Draw-Propensity Rule
    "DRAW_PROPENSITY": {
        "umbral_diferencia": 0.08,  # |pL - pV| < 0.08
        "boost_empate": 0.06        # +6 pp a empate
    },
    
    # Página 4 - Taxonomía de partidos
    "UMBRALES_CLASIFICACION": {
        "ancla_prob_min": 0.60,           # p_max > 60%
        "divisor_prob_min": 0.40,         # 40% < p_max < 60%
        "divisor_prob_max": 0.60,
        "tendencia_empate_min": 0.30      # p_empate > 30%
    },
    
    # Página 5 - Parámetros GRASP-Annealing
    "OPTIMIZACION": {
        "max_iteraciones": 2000,
        "temperatura_inicial": 0.05,
        "factor_enfriamiento": 0.92,
        "alpha_grasp": 0.15,  # Top 15% candidatos
        "iteraciones_sin_mejora": 100
    },
    
    # Página 4 - Arquitectura del portafolio
    "ARQUITECTURA_PORTAFOLIO": {
        "num_core": 4,
        "num_satelites": 26,
        "num_total": 30,
        "correlacion_jaccard_max": 0.57
    },
    
    # Logging y debug
    "DEBUG": True,
    "LOG_LEVEL": "INFO"
}