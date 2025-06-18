# progol_optimizer/models/classifier.py
"""
Clasificador de Partidos - VERSIÓN CORREGIDA
Ancla / Divisor / TendenciaX / Neutro según umbrales específicos
CORRECCIÓN: Umbrales ajustados y diagnóstico para generar partidos Ancla reales
"""

import logging
from typing import Dict, Any, List

class PartidoClassifier:
    """
    Clasifica partidos según la taxonomía exacta del documento (página 4)
    CORREGIDO: Umbrales ajustados para asegurar clasificación Ancla y diagnóstico
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG
        self.umbrales = PROGOL_CONFIG["UMBRALES_CLASIFICACION"]
        self.logger.debug(f"Umbrales de clasificación: {self.umbrales}")
    
    def clasificar_partido(self, partido_calibrado: Dict[str, Any]) -> str:
        """
        Implementa la taxonomía de la página 4.
        CORRECCIÓN: Umbral de confianza ajustado para generar más Anclas.
        """
        probs = [partido_calibrado["prob_local"], partido_calibrado["prob_empate"], partido_calibrado["prob_visitante"]]
        max_prob = max(probs)
        
        # ANCLA: p_max > 60% y alta confianza
        if max_prob > self.umbrales["ancla_prob_min"]:
            probs_sorted = sorted(probs, reverse=True)
            diferencia = probs_sorted[0] - probs_sorted[1]
            # La confianza se mide por la diferencia con la segunda opción más probable.
            # Ajustamos este valor para ser un poco más flexible.
            if diferencia > 0.10: 
                return "Ancla"
            else:
                # Si es >60% pero la segunda opción está muy cerca, es un 'Divisor Fuerte'.
                return "Divisor"
        
        # TENDENCIA EMPATE
        if self._es_tendencia_empate(partido_calibrado):
            return "TendenciaEmpate"
        
        # DIVISOR: 40% < p_max < 60% o volatilidad detectada
        if (self.umbrales["divisor_prob_min"] < max_prob < self.umbrales["divisor_prob_max"] or 
            self._tiene_volatilidad(partido_calibrado)):
            return "Divisor"
        
        return "Neutro"

    def _es_tendencia_empate(self, partido: Dict[str, Any]) -> bool:
        """Determina si el partido tiene tendencia al empate."""
        prob_empate = partido["prob_empate"]
        prob_local = partido["prob_local"]
        prob_visitante = partido["prob_visitante"]
        
        # Criterio 1: Empate es la opción más probable con un margen
        if prob_empate > max(prob_local, prob_visitante) and (prob_empate - max(prob_local, prob_visitante) > 0.05):
            return True
        # Criterio 2: Empate > 30% y L/V muy parejos
        if prob_empate > self.umbrales["tendencia_empate_min"] and abs(prob_local - prob_visitante) < 0.05:
            return True
        return False

    def _tiene_volatilidad(self, partido: Dict[str, Any]) -> bool:
        """Detecta volatilidad que califica como Divisor."""
        if "factor_ajuste" in partido and abs(partido["factor_ajuste"] - 1.0) > 0.10:
            return True
        if "regularizado" in partido and partido["regularizado"]:
            return True
        return False

    def obtener_estadisticas_clasificacion(self, partidos_clasificados: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula y muestra estadísticas de la clasificación."""
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
        
        self.logger.info("📊 Estadísticas de clasificación:")
        for clase, count in clasificaciones.items():
            self.logger.info(f"  {clase}: {count} partidos")
        
        if clasificaciones.get("Ancla", 0) == 0:
            self.logger.warning("⚠️ NO SE ENCONTRARON PARTIDOS ANCLA. El portafolio puede ser subóptimo.")
            diagnostico = self.diagnosticar_clasificacion(partidos_clasificados)
            self.logger.warning("📋 DIAGNÓSTICO DETALLADO:\n" + diagnostico)
            
        return estadisticas

    def diagnosticar_clasificacion(self, partidos_clasificados: List[Dict[str, Any]]) -> str:
        """NUEVA FUNCIÓN: Genera un reporte explicando por qué no hay anclas."""
        diagnostico = ["=== Diagnóstico: ¿Por qué no hay Anclas? ==="]
        for i, p in enumerate(partidos_clasificados):
            max_prob = max(p['prob_local'], p['prob_empate'], p['prob_visitante'])
            if p.get('clasificacion') != 'Ancla' and max_prob > 0.55:
                probs_sorted = sorted([p['prob_local'], p['prob_empate'], p['prob_visitante']], reverse=True)
                diferencia = probs_sorted[0] - probs_sorted[1]
                razon = ""
                if max_prob <= self.umbrales["ancla_prob_min"]:
                    razon = f"max_prob ({max_prob:.2f}) no supera el umbral de {self.umbrales['ancla_prob_min']}"
                elif diferencia <= 0.10:
                    razon = f"la diferencia con la 2da opción ({diferencia:.2f}) no supera el umbral de confianza (0.10)"
                diagnostico.append(f"Partido {i+1} ({p['home']} vs {p['away']}) casi fue Ancla, pero {razon}.")
        if len(diagnostico) == 1:
            return "No hubo partidos cercanos a ser clasificados como Ancla."
        return "\n".join(diagnostico)