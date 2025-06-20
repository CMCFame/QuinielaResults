# progol_optimizer/data/validator.py
"""
Validador de datos según especificaciones del documento técnico
Verifica integridad de datos de entrada antes del procesamiento
"""

import logging
from typing import List, Dict, Any, Tuple

class DataValidator:
    """
    Valida la estructura y contenido de los datos de entrada
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validar_estructura(self, partidos: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Valida que los datos cumplan con la estructura requerida
        
        Args:
            partidos: Lista de partidos cargados
            
        Returns:
            Tuple[bool, List[str]]: (es_válido, lista_errores)
        """
        errores = []
        
        # Validación 1: Número exacto de partidos
        if len(partidos) != 14:
            errores.append(f"Se requieren exactamente 14 partidos, se recibieron {len(partidos)}")
        
        # Validación 2: Estructura de cada partido
        campos_requeridos = [
            'id', 'home', 'away', 'liga',
            'prob_local', 'prob_empate', 'prob_visitante',
            'forma_diferencia', 'lesiones_impact',
            'es_final', 'es_derbi', 'es_playoff'
        ]
        
        for i, partido in enumerate(partidos):
            # Campos obligatorios
            campos_faltantes = [campo for campo in campos_requeridos if campo not in partido]
            if campos_faltantes:
                errores.append(f"Partido {i}: campos faltantes {campos_faltantes}")
            
            # Validar probabilidades
            if all(campo in partido for campo in ['prob_local', 'prob_empate', 'prob_visitante']):
                prob_sum = partido['prob_local'] + partido['prob_empate'] + partido['prob_visitante']
                if abs(prob_sum - 1.0) > 0.01:  # Tolerancia de 1%
                    errores.append(f"Partido {i}: probabilidades no suman 1.0 (suma={prob_sum:.3f})")
                
                # Validar rangos
                for prob_tipo in ['prob_local', 'prob_empate', 'prob_visitante']:
                    prob = partido[prob_tipo]
                    if not (0.0 <= prob <= 1.0):
                        errores.append(f"Partido {i}: {prob_tipo}={prob:.3f} fuera del rango [0,1]")
        
        # Validación 3: Distribución global aproximada
        if len(partidos) == 14 and not errores:
            total_prob_local = sum(p['prob_local'] for p in partidos)
            total_prob_empate = sum(p['prob_empate'] for p in partidos)
            total_prob_visitante = sum(p['prob_visitante'] for p in partidos)
            
            # Rangos esperados según documento (página 2)
            if not (4.5 <= total_prob_local <= 6.0):
                errores.append(f"Suma probabilidades locales fuera de rango esperado [4.5-6.0]: {total_prob_local:.2f}")
            
            if not (3.0 <= total_prob_empate <= 5.0):
                errores.append(f"Suma probabilidades empates fuera de rango esperado [3.0-5.0]: {total_prob_empate:.2f}")
            
            if not (4.0 <= total_prob_visitante <= 6.0):
                errores.append(f"Suma probabilidades visitantes fuera de rango esperado [4.0-6.0]: {total_prob_visitante:.2f}")
        
        es_valido = len(errores) == 0
        
        if es_valido:
            self.logger.info("✅ Validación de datos exitosa")
        else:
            self.logger.error(f"❌ Validación fallida: {len(errores)} errores encontrados")
            for error in errores:
                self.logger.error(f"  - {error}")
        
        return es_valido, errores
    
    def validar_integridad_referencial(self, partidos: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validaciones adicionales de integridad referencial
        """
        errores = []
        
        # IDs únicos
        ids = [p['id'] for p in partidos]
        if len(set(ids)) != len(ids):
            errores.append("IDs de partidos duplicados")
        
        # Equipos no pueden jugar contra sí mismos
        for i, partido in enumerate(partidos):
            if partido['home'] == partido['away']:
                errores.append(f"Partido {i}: equipo no puede jugar contra sí mismo")
        
        # Equipos no pueden aparecer múltiples veces
        equipos_usados = set()
        for i, partido in enumerate(partidos):
            home, away = partido['home'], partido['away']
            if home in equipos_usados:
                errores.append(f"Partido {i}: equipo {home} aparece múltiples veces")
            if away in equipos_usados:
                errores.append(f"Partido {i}: equipo {away} aparece múltiples veces")
            equipos_usados.add(home)
            equipos_usados.add(away)
        
        return len(errores) == 0, errores