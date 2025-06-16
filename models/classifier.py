# progol_optimizer/models/classifier.py
"""
Clasificador de Partidos - VERSIÓN CORREGIDA
Ancla / Divisor / TendenciaX / Neutro según umbrales específicos
CORRECCIÓN: Umbrales ajustados para generar partidos Ancla reales
"""

import logging
from typing import Dict, Any, List

class PartidoClassifier:
    """
    Clasifica partidos según la taxonomía exacta del documento (página 4)
    CORREGIDO: Umbrales ajustados para asegurar clasificación Ancla
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.umbrales = PROGOL_CONFIG["UMBRALES_CLASIFICACION"]
        
        self.logger.debug(f"Umbrales de clasificación: {self.umbrales}")
    
    def clasificar_partido(self, partido_calibrado: Dict[str, Any]) -> str:
        """
        Implementa EXACTAMENTE la taxonomía de la página 4 - VERSIÓN CORREGIDA:
        
        - ANCLA: p_max > 60% y alta confianza (AJUSTADO: diferencia >10% en lugar de 15%)
        - DIVISOR: 40% < p_max < 60% o volatilidad
        - TENDENCIA X: Regla draw-propensity activa
        - NEUTRO: Todo lo demás
        
        CORRECCIÓN: Umbral de alta confianza reducido de 15% a 10% para generar más Anclas
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
        
        # ANCLA: p_max > 60% y alta confianza - CORREGIDO
        if max_prob > self.umbrales["ancla_prob_min"]:
            # CORRECCIÓN: Reducir umbral de alta confianza de 15% a 10%
            probs_sorted = sorted(probs, reverse=True)
            diferencia = probs_sorted[0] - probs_sorted[1]
            
            # ANTES: diferencia > 0.15, AHORA: diferencia > 0.10
            if diferencia > 0.10:  # Diferencia mínima del 10% para alta confianza
                clasificacion = "Ancla"
                self.logger.debug(f"  -> ANCLA (max_prob={max_prob:.3f}, diff={diferencia:.3f})")
                return clasificacion
            else:
                # Si supera 60% pero diferencia < 10%, es Divisor
                self.logger.debug(f"  -> DIVISOR (max_prob={max_prob:.3f} pero diff={diferencia:.3f} < 0.10)")
                return "Divisor"
        
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
        MEJORADO: Criterios más estrictos para asegurar que realmente sea tendencia empate
        """
        prob_empate = partido["prob_empate"]
        prob_local = partido["prob_local"]
        prob_visitante = partido["prob_visitante"]
        
        # Criterio 1: Empate es la opción más probable CON margen significativo
        max_no_empate = max(prob_local, prob_visitante)
        if prob_empate > max_no_empate and prob_empate - max_no_empate > 0.05:
            self.logger.debug(f"    TendenciaEmpate: empate dominante ({prob_empate:.3f} vs {max_no_empate:.3f})")
            return True
        
        # Criterio 2: Empate > 30% y diferencia entre L-V es muy pequeña
        if (prob_empate > self.umbrales["tendencia_empate_min"] and
            abs(prob_local - prob_visitante) < 0.05):  # Más estricto: < 5%
            self.logger.debug(f"    TendenciaEmpate: empate alto + L-V equilibrado")
            return True
        
        # Criterio 3: Partido especial (final, derbi) con empate muy alto
        if (partido.get("es_final", False) or partido.get("es_derbi", False)) and prob_empate > 0.35:
            self.logger.debug(f"    TendenciaEmpate: partido especial con empate alto")
            return True
        
        return False
    
    def _tiene_volatilidad(self, partido: Dict[str, Any]) -> bool:
        """
        Detecta volatilidad en las probabilidades que califique como Divisor
        MEJORADO: Detección más sensible de volatilidad
        """
        # Si existe factor de ajuste significativo (> 8% en lugar de 10%)
        if "factor_ajuste" in partido:
            factor = partido["factor_ajuste"]
            if abs(factor - 1.0) > 0.08:  # Más sensible
                self.logger.debug(f"    Volatilidad: factor_ajuste={factor:.3f}")
                return True
        
        # Si hay diferencias importantes entre raw y calibrado
        if all(key in partido for key in ["prob_local_raw", "prob_empate_raw", "prob_visitante_raw"]):
            diff_local = abs(partido["prob_local"] - partido["prob_local_raw"])
            diff_empate = abs(partido["prob_empate"] - partido["prob_empate_raw"])
            diff_visitante = abs(partido["prob_visitante"] - partido["prob_visitante_raw"])
            
            max_diff = max(diff_local, diff_empate, diff_visitante)
            if max_diff > 0.03:  # Más sensible: cambio > 3pp en lugar de 5pp
                self.logger.debug(f"    Volatilidad: max_diff_calibracion={max_diff:.3f}")
                return True
        
        # Si hay regularización aplicada
        if partido.get("regularizado", False):
            self.logger.debug(f"    Volatilidad: partido regularizado")
            return True
        
        return False
    
    def obtener_estadisticas_clasificacion(self, partidos_clasificados: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula estadísticas de la clasificación realizada
        MEJORADO: Logging más detallado para debug
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
        
        self.logger.info(f"📊 Estadísticas de clasificación:")
        for clase, count in clasificaciones.items():
            porcentaje = (count/total)*100
            self.logger.info(f"  {clase}: {count} partidos ({porcentaje:.1f}%)")
        
        # NUEVO: Advertencias si la distribución no es óptima
        num_anclas = clasificaciones.get("Ancla", 0)
        num_divisores = clasificaciones.get("Divisor", 0)
        num_empates = clasificaciones.get("TendenciaEmpate", 0)
        
        if num_anclas == 0:
            self.logger.warning("⚠️ NO HAY PARTIDOS ANCLA - Esto puede indicar probabilidades mal calibradas")
        elif num_anclas < 2:
            self.logger.warning(f"⚠️ Solo {num_anclas} partido Ancla - Se recomienda al menos 2-4 Anclas")
        else:
            self.logger.info(f"✅ {num_anclas} partidos Ancla - Distribución adecuada")
        
        if num_divisores < 8:
            self.logger.warning(f"⚠️ Solo {num_divisores} Divisores - Se esperan 8-12 para buena diversificación")
        
        if num_empates == 0:
            self.logger.warning("⚠️ NO HAY PARTIDOS TendenciaEmpate - Puede afectar balance de empates")
        
        return estadisticas
    
    def diagnosticar_clasificacion(self, partidos_clasificados: List[Dict[str, Any]]) -> str:
        """
        NUEVA FUNCIÓN: Genera diagnóstico detallado de la clasificación
        """
        diagnostico = ["=== DIAGNÓSTICO DE CLASIFICACIÓN ==="]
        
        for i, partido in enumerate(partidos_clasificados):
            probs = [partido["prob_local"], partido["prob_empate"], partido["prob_visitante"]]
            max_prob = max(probs)
            probs_sorted = sorted(probs, reverse=True)
            diferencia = probs_sorted[0] - probs_sorted[1]
            
            diagnostico.append(f"P{i+1:2d} {partido['home'][:12]:12s} vs {partido['away'][:12]:12s}")
            diagnostico.append(f"     L={partido['prob_local']:.3f} E={partido['prob_empate']:.3f} V={partido['prob_visitante']:.3f}")
            diagnostico.append(f"     Max={max_prob:.3f}, Diff={diferencia:.3f}, Clase={partido.get('clasificacion', 'N/A')}")
            
            # Explicar por qué no es Ancla si max_prob > 0.55
            if max_prob > 0.55 and partido.get('clasificacion') != 'Ancla':
                if max_prob <= 0.60:
                    diagnostico.append(f"     → No Ancla: max_prob {max_prob:.3f} ≤ 0.60")
                elif diferencia <= 0.10:
                    diagnostico.append(f"     → No Ancla: diferencia {diferencia:.3f} ≤ 0.10")
            
            diagnostico.append("")
        
        return "\n".join(diagnostico)