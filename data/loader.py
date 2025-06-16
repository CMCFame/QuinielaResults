# progol_optimizer/data/loader.py - CORREGIDO
"""
Cargador de datos CORREGIDO con distribución balanceada según documento técnico
Genera datos realistas que respetan 38%L, 29%E, 33%V
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

class DataLoader:
    """
    Carga datos desde CSV o genera datos de ejemplo BALANCEADOS
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
        CORREGIDO: Genera probabilidades realistas basadas en la distribución histórica
        del documento (38% L, 29% E, 33% V) con variación natural BALANCEADA
        """
        from config.constants import PROGOL_CONFIG
        
        # Usar distribución histórica como base
        hist_dist = PROGOL_CONFIG["DISTRIBUCION_HISTORICA"]
        
        # CORRECCIÓN: Generar probabilidades más variadas y balanceadas
        # En lugar de usar la media siempre, crear variedad realista
        
        # Tipos de partidos con diferentes perfiles
        tipo_partido = np.random.choice([
            'equilibrado',      # 40% - Partidos muy disputados  
            'favorito_local',   # 30% - Local tiene ventaja
            'favorito_visitante', # 20% - Visitante favorito
            'empate_probable'   # 10% - Empate muy probable
        ], p=[0.4, 0.3, 0.2, 0.1])
        
        if tipo_partido == 'equilibrado':
            # Partidos equilibrados (35-45% L, 25-35% E, 25-40% V)
            prob_local = np.random.uniform(0.30, 0.45)
            prob_empate = np.random.uniform(0.25, 0.35)
            prob_visitante = 1.0 - prob_local - prob_empate
            
        elif tipo_partido == 'favorito_local':
            # Local favorito (45-65% L, 20-30% E, 15-35% V)
            prob_local = np.random.uniform(0.45, 0.65)
            prob_empate = np.random.uniform(0.20, 0.30)
            prob_visitante = 1.0 - prob_local - prob_empate
            
        elif tipo_partido == 'favorito_visitante':
            # Visitante favorito (15-35% L, 20-30% E, 45-65% V)
            prob_visitante = np.random.uniform(0.45, 0.65)
            prob_empate = np.random.uniform(0.20, 0.30)
            prob_local = 1.0 - prob_visitante - prob_empate
            
        else:  # empate_probable
            # Empate muy probable (25-35% L, 35-50% E, 25-35% V)
            prob_empate = np.random.uniform(0.35, 0.50)
            prob_local = np.random.uniform(0.25, 0.35)
            prob_visitante = 1.0 - prob_empate - prob_local
        
        # Asegurar valores válidos
        prob_local = max(0.05, min(0.80, prob_local))
        prob_empate = max(0.05, min(0.60, prob_empate))
        prob_visitante = max(0.05, min(0.80, prob_visitante))
        
        # Normalizar para que sume exactamente 1
        total = prob_local + prob_empate + prob_visitante
        prob_local /= total
        prob_empate /= total
        prob_visitante /= total
        
        return prob_local, prob_empate, prob_visitante
    
    def _generar_datos_ejemplo(self) -> List[Dict[str, Any]]:
        """
        CORREGIDO: Genera 14 partidos de ejemplo con distribución BALANCEADA
        """
        self.logger.info("Generando 14 partidos de ejemplo BALANCEADOS...")
        
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
        
        # CORRECCIÓN: Pre-definir algunos partidos específicos para balancear
        partidos_especiales = [
            # Favoritos locales claros
            {'tipo': 'favorito_local', 'count': 4},
            # Favoritos visitantes
            {'tipo': 'favorito_visitante', 'count': 4}, 
            # Equilibrados
            {'tipo': 'equilibrado', 'count': 4},
            # Empates probables
            {'tipo': 'empate_probable', 'count': 2}
        ]
        
        contador_especiales = 0
        tipo_actual = partidos_especiales[0]['tipo']
        count_actual = 0
        
        for liga, enfrentamientos in equipos.items():
            for home, away in enfrentamientos:
                
                # Determinar tipo de partido para forzar balance
                if count_actual >= partidos_especiales[contador_especiales]['count']:
                    contador_especiales = min(contador_especiales + 1, len(partidos_especiales) - 1)
                    tipo_actual = partidos_especiales[contador_especiales]['tipo']
                    count_actual = 0
                
                # Generar probabilidades según tipo
                prob_local, prob_empate, prob_visitante = self._generar_probabilidades_por_tipo(tipo_actual)
                count_actual += 1
                
                # Contextualizar partido
                es_final = 'final' in liga.lower() or idx == 13  # Último partido como final
                es_derbi = 'clásico' in f"{home} vs {away}".lower() or home in ['Real Madrid', 'Barcelona']
                
                partido = {
                    'id': idx,
                    'home': home,
                    'away': away,
                    'liga': liga,
                    'prob_local': prob_local,
                    'prob_empate': prob_empate,
                    'prob_visitante': prob_visitante,
                    'forma_diferencia': np.random.normal(0, 0.8),  # Reducir varianza
                    'lesiones_impact': np.random.normal(0, 0.3),   # Reducir varianza
                    'es_final': es_final,
                    'es_derbi': es_derbi,
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
        
        # Asegurar exactamente 14 partidos con balance forzado
        while len(partidos) < 14:
            # Generar partidos adicionales equilibrados
            prob_local, prob_empate, prob_visitante = self._generar_probabilidades_por_tipo('equilibrado')
            
            partidos.append({
                'id': len(partidos),
                'home': f"Equipo {len(partidos) + 1}A",
                'away': f"Equipo {len(partidos) + 1}B",
                'liga': 'Liga Ejemplo',
                'prob_local': prob_local,
                'prob_empate': prob_empate,
                'prob_visitante': prob_visitante,
                'forma_diferencia': np.random.normal(0, 0.8),
                'lesiones_impact': np.random.normal(0, 0.3),
                'es_final': False,
                'es_derbi': False,
                'es_playoff': False,
                'fecha': '2025-06-07',
                'jornada': 1,
                'concurso_id': '2283'
            })
        
        # Verificar distribución final
        self._verificar_distribucion_balanceada(partidos)
        
        self.logger.info(f"✅ Generados {len(partidos)} partidos de ejemplo BALANCEADOS")
        return partidos[:14]  # Exactamente 14
    
    def _generar_probabilidades_por_tipo(self, tipo: str) -> tuple:
        """
        NUEVO: Genera probabilidades específicas por tipo para forzar balance
        """
        if tipo == 'favorito_local':
            prob_local = np.random.uniform(0.50, 0.70)
            prob_empate = np.random.uniform(0.20, 0.30)
            prob_visitante = 1.0 - prob_local - prob_empate
            
        elif tipo == 'favorito_visitante':
            prob_visitante = np.random.uniform(0.50, 0.70)
            prob_empate = np.random.uniform(0.20, 0.30)
            prob_local = 1.0 - prob_visitante - prob_empate
            
        elif tipo == 'empate_probable':
            prob_empate = np.random.uniform(0.40, 0.55)
            prob_local = np.random.uniform(0.22, 0.35)
            prob_visitante = 1.0 - prob_empate - prob_local
            
        else:  # equilibrado
            prob_local = np.random.uniform(0.30, 0.45)
            prob_empate = np.random.uniform(0.25, 0.35)
            prob_visitante = 1.0 - prob_local - prob_empate
        
        # Normalizar
        total = prob_local + prob_empate + prob_visitante
        return prob_local/total, prob_empate/total, prob_visitante/total
    
    def _verificar_distribucion_balanceada(self, partidos: List[Dict[str, Any]]):
        """
        NUEVO: Verifica que la distribución esté en rangos aceptables
        """
        if not partidos:
            return
            
        # Calcular distribución esperada si todas las quinielas siguieran máxima probabilidad
        total_prob_L = sum(p['prob_local'] for p in partidos)
        total_prob_E = sum(p['prob_empate'] for p in partidos)
        total_prob_V = sum(p['prob_visitante'] for p in partidos)
        
        total = total_prob_L + total_prob_E + total_prob_V
        dist_L = total_prob_L / total
        dist_E = total_prob_E / total
        dist_V = total_prob_V / total
        
        self.logger.info(f"📊 Distribución esperada datos: L={dist_L:.1%}, E={dist_E:.1%}, V={dist_V:.1%}")
        
        # Warnings si está muy desbalanceado
        if dist_L > 0.50:
            self.logger.warning(f"⚠️ Datos sesgados hacia locales: {dist_L:.1%}")
        if dist_V < 0.20:
            self.logger.warning(f"⚠️ Pocos visitantes en datos: {dist_V:.1%}")
    
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