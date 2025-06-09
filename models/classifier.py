# progol_optimizer/models/classifier.py
"""
Clasificador de Partidos - Implementación EXACTA de la taxonomía de la página 4
Ancla / Divisor / TendenciaX / Neutro según umbrales específicos
"""

import logging
from typing import Dict, Any, List

class PartidoClassifier:
    """
    Clasifica partidos según la taxonomía exacta del documento (página 4)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.umbrales = PROGOL_CONFIG["UMBRALES_CLASIFICACION"]
        
        self.logger.debug(f"Umbrales de clasificación: {self.umbrales}")
    
    def clasificar_partido(self, partido_calibrado: Dict[str, Any]) -> str:
        """
        Implementa EXACTAMENTE la taxonomía de la página 4:
        
        - ANCLA: p_max > 60% y alta confianza
        - DIVISOR: 40% < p_max < 60% o volatilidad
        - TENDENCIA X: Regla draw-propensity activa
        - NEUTRO: Todo lo demás
        
        Args:
            partido_calibrado: Partido con probabilidades ya calibradas
            
        Returns:
            str: Clasificación ('Ancla', 'Divisor', 'TendenciaEmpate', 'Neutro')
        """
        probs = [
            partido_calibrado["prob_local"],
            partido_calibrado["prob_empate"], 
            partido_calibrado["prob_visitante"]
        ]
        max_prob = max(probs)
        prob_empate = partido_calibrado["prob_empate"]
        
        self.logger.debug(f"Clasificando {partido_calibrado['home']} vs {partido_calibrado['away']}: "
                         f"max_prob={max_prob:.3f}, prob_empate={prob_empate:.3f}")
        
        # ANCLA: p_max > 60% y alta confianza
        if max_prob > self.umbrales["ancla_prob_min"]:
            # Verificar alta confianza (diferencia significativa con segunda opción)
            probs_sorted = sorted(probs, reverse=True)
            diferencia = probs_sorted[0] - probs_sorted[1]
            
            if diferencia > 0.15:  # Diferencia mínima del 15% para alta confianza
                clasificacion = "Ancla"
                self.logger.debug(f"  -> ANCLA (max_prob={max_prob:.3f}, diff={diferencia:.3f})")
                return clasificacion
        
        # TENDENCIA EMPATE: condiciones especiales empate
        if self._es_tendencia_empate(partido_calibrado):
            clasificacion = "TendenciaEmpate"
            self.logger.debug(f"  -> TENDENCIA EMPATE (prob_empate={prob_empate:.3f})")
            return clasificacion
        
        # DIVISOR: 40% < p_max < 60% o volatilidad detectada
        if (self.umbrales["divisor_prob_min"] < max_prob < self.umbrales["divisor_prob_max"] or
            self._tiene_volatilidad(partido_calibrado)):
            clasificacion = "Divisor"
            self.logger.debug(f"  -> DIVISOR (max_prob={max_prob:.3f})")
            return clasificacion
        
        # NEUTRO: todo lo demás
        clasificacion = "Neutro"
        self.logger.debug(f"  -> NEUTRO (max_prob={max_prob:.3f})")
        return clasificacion
    
    def _es_tendencia_empate(self, partido: Dict[str, Any]) -> bool:
        """
        Determina si el partido tiene tendencia al empate según criterios específicos
        """
        prob_empate = partido["prob_empate"]
        prob_local = partido["prob_local"]
        prob_visitante = partido["prob_visitante"]
        
        # Criterio 1: Empate es la opción más probable
        if prob_empate >= max(prob_local, prob_visitante):
            return True
        
        # Criterio 2: Empate > 30% y diferencia entre L-V es pequeña
        if (prob_empate > self.umbrales["tendencia_empate_min"] and
            abs(prob_local - prob_visitante) < 0.10):
            return True
        
        # Criterio 3: Partido especial (final, derbi) con empate alto
        if (partido.get("es_final", False) or partido.get("es_derbi", False)) and prob_empate > 0.25:
            return True
        
        return False
    
    def _tiene_volatilidad(self, partido: Dict[str, Any]) -> bool:
        """
        Detecta volatilidad en las probabilidades que califique como Divisor
        """
        # Si existe factor de ajuste significativo (> 10%)
        if "factor_ajuste" in partido:
            factor = partido["factor_ajuste"]
            if abs(factor - 1.0) > 0.10:
                return True
        
        # Si hay diferencias importantes entre raw y calibrado
        if all(key in partido for key in ["prob_local_raw", "prob_empate_raw", "prob_visitante_raw"]):
            diff_local = abs(partido["prob_local"] - partido["prob_local_raw"])
            diff_empate = abs(partido["prob_empate"] - partido["prob_empate_raw"])
            diff_visitante = abs(partido["prob_visitante"] - partido["prob_visitante_raw"])
            
            if max(diff_local, diff_empate, diff_visitante) > 0.05:  # Cambio > 5pp
                return True
        
        return False
    
    def obtener_estadisticas_clasificacion(self, partidos_clasificados: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula estadísticas de la clasificación realizada
        """
        clasificaciones = {}
        for partido in partidos_clasificados:
            clase = partido.get("clasificacion", "Sin clasificar")
            clasificaciones[clase] = clasificaciones.get(clase, 0) + 1
        
        total = len(partidos_clasificados)
        estadisticas = {
            "total_partidos": total,
            "distribución": clasificaciones,
            "porcentajes": {clase: (count/total)*100 for clase, count in clasificaciones.items()}
        }
        
        self.logger.info(f"Estadísticas de clasificación:")
        for clase, count in clasificaciones.items():
            porcentaje = (count/total)*100
            self.logger.info(f"  {clase}: {count} partidos ({porcentaje:.1f}%)")
        
        return estadisticas