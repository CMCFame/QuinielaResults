# progol_optimizer/data/loader.py
"""
Cargador de datos según especificaciones del documento técnico
Maneja CSV con estructura: concurso_id,fecha,match_no,liga,home,away,l_g,a_g,resultado,premio_1,premio_2
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

class DataLoader:
    """
    Carga datos desde CSV y los convierte al formato requerido para el pipeline
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def cargar_datos(self, archivo_path: str) -> List[Dict[str, Any]]:
        """
        Carga datos desde CSV o genera datos de ejemplo si no existe el archivo
        
        Args:
            archivo_path: Ruta al archivo CSV
            
        Returns:
            List[Dict]: Lista de partidos con probabilidades y metadatos
        """
        self.logger.info(f"Cargando datos desde: {archivo_path}")
        
        if not Path(archivo_path).exists():
            self.logger.warning(f"Archivo {archivo_path} no encontrado. Generando datos de ejemplo...")
            return self._generar_datos_ejemplo()
        
        try:
            # Cargar CSV con estructura del documento
            df = pd.read_csv(archivo_path)
            
            # Validar columnas obligatorias
            columnas_requeridas = ['home', 'away', 'liga']
            columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
            
            if columnas_faltantes:
                raise ValueError(f"Columnas faltantes en CSV: {columnas_faltantes}")
            
            # Convertir a formato interno
            partidos = []
            for idx, row in df.iterrows():
                partido = self._procesar_fila_csv(row, idx)
                partidos.append(partido)
            
            self.logger.info(f"✅ Cargados {len(partidos)} partidos desde CSV")
            return partidos
            
        except Exception as e:
            self.logger.error(f"Error cargando CSV: {e}")
            self.logger.info("Generando datos de ejemplo como fallback...")
            return self._generar_datos_ejemplo()
    
    def _procesar_fila_csv(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """
        Convierte una fila del CSV al formato interno requerido
        """
        # Probabilidades base - Si no están en CSV, usar distribución histórica + ruido
        if all(col in row and pd.notna(row[col]) for col in ['prob_local', 'prob_empate', 'prob_visitante']):
            prob_local = float(row['prob_local'])
            prob_empate = float(row['prob_empate'])
            prob_visitante = float(row['prob_visitante'])
        else:
            # Generar probabilidades basadas en distribución histórica + ruido aleatorio
            prob_local, prob_empate, prob_visitante = self._generar_probabilidades_realistas()
        
        # Metadatos contextuales
        partido = {
            'id': idx,
            'home': str(row['home']),
            'away': str(row['away']),
            'liga': str(row.get('liga', 'Liga Desconocida')),
            
            # Probabilidades base (antes de calibración)
            'prob_local': prob_local,
            'prob_empate': prob_empate,
            'prob_visitante': prob_visitante,
            
            # Factores para calibración bayesiana
            'forma_diferencia': float(row.get('forma_diferencia', np.random.normal(0, 1))),
            'lesiones_impact': float(row.get('lesiones_impact', np.random.normal(0, 0.5))),
            'es_final': bool(row.get('es_final', False)),
            'es_derbi': bool(row.get('es_derbi', False)),
            'es_playoff': bool(row.get('es_playoff', False)),
            
            # Metadatos adicionales
            'fecha': str(row.get('fecha', '2025-06-07')),
            'jornada': int(row.get('jornada', 1)),
            'concurso_id': str(row.get('concurso_id', '2283'))
        }
        
        return partido
    
    def _generar_probabilidades_realistas(self) -> tuple:
        """
        Genera probabilidades realistas basadas en la distribución histórica
        del documento (38% L, 29% E, 33% V) con variación natural
        """
        from config.constants import PROGOL_CONFIG
        
        # Usar distribución histórica como base
        hist_dist = PROGOL_CONFIG["DISTRIBUCION_HISTORICA"]
        
        # Agregar ruido realista
        ruido = np.random.normal(0, 0.05, 3)  # ±5% de variación
        
        prob_local = max(0.1, min(0.8, hist_dist["L"] + ruido[0]))
        prob_empate = max(0.1, min(0.6, hist_dist["E"] + ruido[1]))
        prob_visitante = max(0.1, min(0.8, hist_dist["V"] + ruido[2]))
        
        # Normalizar para que sume 1
        total = prob_local + prob_empate + prob_visitante
        prob_local /= total
        prob_empate /= total
        prob_visitante /= total
        
        return prob_local, prob_empate, prob_visitante
    
    def _generar_datos_ejemplo(self) -> List[Dict[str, Any]]:
        """
        Genera 14 partidos de ejemplo con datos realistas para testing
        Simula un concurso típico de Progol
        """
        self.logger.info("Generando 14 partidos de ejemplo...")
        
        # Equipos realistas para diferentes ligas
        equipos = {
            'Liga MX': [
                ('América', 'Chivas'), ('Cruz Azul', 'Pumas'), 
                ('Monterrey', 'Tigres'), ('Atlas', 'Santos')
            ],
            'Premier League': [
                ('Manchester City', 'Arsenal'), ('Liverpool', 'Chelsea')
            ],
            'UEFA CL': [
                ('Real Madrid', 'Barcelona'), ('PSG', 'Bayern')
            ],
            'Copa MX': [
                ('León', 'Pachuca'), ('Toluca', 'Necaxa'),
                ('FC Juárez', 'Mazatlán'), ('Puebla', 'Querétaro')
            ]
        }
        
        partidos = []
        idx = 0
        
        for liga, enfrentamientos in equipos.items():
            for home, away in enfrentamientos:
                # Generar probabilidades variadas pero realistas
                if 'final' in home.lower() or 'derbi' in liga.lower():
                    # Partidos especiales más equilibrados
                    prob_local, prob_empate, prob_visitante = self._generar_probabilidades_equilibradas()
                    es_final = True
                else:
                    prob_local, prob_empate, prob_visitante = self._generar_probabilidades_realistas()
                    es_final = False
                
                partido = {
                    'id': idx,
                    'home': home,
                    'away': away,
                    'liga': liga,
                    'prob_local': prob_local,
                    'prob_empate': prob_empate,
                    'prob_visitante': prob_visitante,
                    'forma_diferencia': np.random.normal(0, 1),
                    'lesiones_impact': np.random.normal(0, 0.5),
                    'es_final': es_final,
                    'es_derbi': 'clásico' in f"{home} vs {away}".lower(),
                    'es_playoff': liga == 'UEFA CL',
                    'fecha': '2025-06-07',
                    'jornada': 1,
                    'concurso_id': '2283'
                }
                
                partidos.append(partido)
                idx += 1
                
                if len(partidos) >= 14:  # Máximo 14 partidos por concurso
                    break
            
            if len(partidos) >= 14:
                break
        
        # Asegurar exactamente 14 partidos
        while len(partidos) < 14:
            partidos.append(partidos[-1].copy())
            partidos[-1]['id'] = len(partidos) - 1
            partidos[-1]['home'] = f"Equipo {len(partidos)}A"
            partidos[-1]['away'] = f"Equipo {len(partidos)}B"
        
        self.logger.info(f"✅ Generados {len(partidos)} partidos de ejemplo")
        return partidos[:14]  # Exactamente 14
    
    def _generar_probabilidades_equilibradas(self) -> tuple:
        """
        Genera probabilidades más equilibradas para finales/derbis
        """
        # Partidos más equilibrados
        prob_local = np.random.uniform(0.25, 0.45)
        prob_empate = np.random.uniform(0.25, 0.40)
        prob_visitante = 1.0 - prob_local - prob_empate
        
        # Asegurar valores válidos
        if prob_visitante < 0.15:
            prob_visitante = 0.15
            total = prob_local + prob_empate + prob_visitante
            prob_local /= total
            prob_empate /= total
            prob_visitante /= total
        
        return prob_local, prob_empate, prob_visitante