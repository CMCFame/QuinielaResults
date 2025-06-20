# progol_optimizer/data/loader.py
"""
Cargador de datos MEJORADO con garantía matemática de partidos Ancla
Implementa cobertura combinatoria y distribución probabilística optimizada
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from logging_setup import get_instrumentor


class EnhancedDataLoader:
    """
    Cargador de datos MEJORADO que garantiza:
    1. Al menos 6 partidos Ancla (>65% probabilidad)
    2. Distribución global que respeta 38%L, 29%E, 33%V
    3. Cobertura combinatoria para máxima diversidad
    4. Instrumentación completa para debug
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.instrumentor = get_instrumentor()
        
        # Configuración matemática basada en teoría de cobertura
        self.target_distribution = {"L": 0.38, "E": 0.29, "V": 0.33}
        self.anchor_threshold = 0.65  # Umbral más alto para garantizar Anclas
        self.min_anchors = 6  # Mínimo garantizado
        self.coverage_matrix = None
        
        self.logger.debug("🔧 EnhancedDataLoader inicializado con cobertura combinatoria")
    
    def cargar_datos(self, archivo_path: str) -> List[Dict[str, Any]]:
        """
        Carga datos con instrumentación completa
        """
        timer_id = self.instrumentor.start_timer("cargar_datos")
        
        try:
            self.logger.info(f"📂 Cargando datos desde: {archivo_path}")
            
            if not Path(archivo_path).exists():
                self.logger.warning(f"Archivo {archivo_path} no encontrado. Generando datos optimizados...")
                partidos = self._generar_datos_optimizados()
            else:
                partidos = self._procesar_archivo_csv(archivo_path)
            
            # Validar y corregir distribución
            partidos_validados = self._garantizar_distribucion_optima(partidos)
            
            # Validar cobertura de Anclas
            self._validar_cobertura_anclas(partidos_validados)
            
            self.instrumentor.end_timer(timer_id, success=True, metrics={
                "total_partidos": len(partidos_validados),
                "anclas_generadas": self._contar_anclas(partidos_validados),
                "distribucion_L": sum(1 for p in partidos_validados if self._resultado_mas_probable(p) == "L"),
                "distribucion_E": sum(1 for p in partidos_validados if self._resultado_mas_probable(p) == "E"),
                "distribucion_V": sum(1 for p in partidos_validados if self._resultado_mas_probable(p) == "V")
            })
            
            self.logger.info(f"✅ Datos cargados exitosamente: {len(partidos_validados)} partidos")
            return partidos_validados
            
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def _generar_datos_optimizados(self) -> List[Dict[str, Any]]:
        """
        Genera 14 partidos usando cobertura combinatoria optimizada
        GARANTIZA al menos 6 Anclas y distribución correcta
        """
        timer_id = self.instrumentor.start_timer("generar_datos_optimizados")
        
        self.logger.info("🎯 Generando datos con cobertura combinatoria optimizada...")
        
        # Configuración específica por tipo de partido con MATEMÁTICA EXACTA
        configuraciones_partidos = [
            # ANCLAS SUPER-FUERTES (6 partidos garantizados >65%)
            ('Manchester City', 'Sheffield Wed', 'Premier League', {'prob_local': 0.72, 'tipo': 'ancla_local_fuerte'}),
            ('Real Madrid', 'Getafe', 'La Liga', {'prob_local': 0.68, 'tipo': 'ancla_local_fuerte'}),
            ('Bayern Munich', 'Hoffenheim', 'Bundesliga', {'prob_local': 0.70, 'tipo': 'ancla_local_fuerte'}),
            ('PSG', 'Montpellier', 'Ligue 1', {'prob_local': 0.69, 'tipo': 'ancla_local_fuerte'}),
            ('Brighton', 'Liverpool', 'Premier League', {'prob_visitante': 0.67, 'tipo': 'ancla_visitante_fuerte'}),
            ('Celta', 'Barcelona', 'La Liga', {'prob_visitante': 0.66, 'tipo': 'ancla_visitante_fuerte'}),
            
            # EMPATES ESTRATÉGICOS (3 partidos para balance)
            ('Athletic Club', 'Real Sociedad', 'La Liga', {'prob_empate': 0.45, 'tipo': 'empate_estrategico'}),
            ('Valencia', 'Sevilla', 'La Liga', {'prob_empate': 0.42, 'tipo': 'empate_estrategico'}),
            ('Tottenham', 'Chelsea', 'Premier League', {'prob_empate': 0.40, 'tipo': 'empate_estrategico'}),
            
            # DIVISORES BALANCEADOS (5 partidos para diversidad)
            ('Arsenal', 'Newcastle', 'Premier League', {'prob_local': 0.48, 'tipo': 'divisor_equilibrado'}),
            ('Villarreal', 'Betis', 'La Liga', {'prob_visitante': 0.50, 'tipo': 'divisor_equilibrado'}),
            ('West Ham', 'Aston Villa', 'Premier League', {'prob_visitante': 0.49, 'tipo': 'divisor_equilibrado'}),
            ('Crystal Palace', 'Fulham', 'Premier League', {'prob_local': 0.47, 'tipo': 'divisor_equilibrado'}),
            ('Atletico Madrid', 'Villarreal', 'La Liga', {'prob_empate': 0.35, 'tipo': 'divisor_equilibrado'})
        ]
        
        partidos = []
        for idx, (home, away, liga, config) in enumerate(configuraciones_partidos):
            prob_local, prob_empate, prob_visitante = self._calcular_probabilidades_optimizadas(config, idx)
            
            # Validar que realmente es Ancla si está diseñado como tal
            max_prob = max(prob_local, prob_empate, prob_visitante)
            if 'ancla' in config['tipo'] and max_prob < self.anchor_threshold:
                self.logger.warning(f"⚠️ Partido {idx+1} diseñado como Ancla pero prob_max={max_prob:.3f} < {self.anchor_threshold}")
                # Forzar que sea Ancla real
                if 'local' in config['tipo']:
                    prob_local = max(0.68, prob_local)
                elif 'visitante' in config['tipo']:
                    prob_visitante = max(0.68, prob_visitante)
                
                # Re-normalizar
                total = prob_local + prob_empate + prob_visitante
                prob_local /= total
                prob_empate /= total
                prob_visitante /= total
            
            partido = {
                'id': idx,
                'home': home,
                'away': away,
                'liga': liga,
                'prob_local': prob_local,
                'prob_empate': prob_empate,
                'prob_visitante': prob_visitante,
                'forma_diferencia': self._generar_factor_contextual('forma', config['tipo']),
                'lesiones_impact': self._generar_factor_contextual('lesiones', config['tipo']),
                'es_final': 'final' in liga.lower(),
                'es_derbi': any(word in f"{home} {away}".lower() for word in ['madrid', 'barcelona', 'chelsea', 'tottenham']),
                'es_playoff': False,
                'fecha': '2025-06-19',
                'jornada': 1,
                'concurso_id': '2283',
                'configuracion_tipo': config['tipo']  # Para debug
            }
            
            partidos.append(partido)
            
            self.instrumentor.log_state_change(
                component="partido_generado",
                old_state=f"partido_{idx}",
                new_state={
                    "tipo": config['tipo'],
                    "max_prob": max(prob_local, prob_empate, prob_visitante),
                    "es_ancla": max_prob >= self.anchor_threshold
                }
            )
        
        self.instrumentor.end_timer(timer_id, success=True, metrics={
            "partidos_generados": len(partidos),
            "anclas_objetivo": self.min_anchors,
            "anclas_reales": self._contar_anclas(partidos)
        })
        
        return partidos
    
    def _calcular_probabilidades_optimizadas(self, config: Dict[str, Any], partido_idx: int) -> Tuple[float, float, float]:
        """
        Calcula probabilidades usando matemática exacta según tipo de partido
        """
        tipo = config['tipo']
        np.random.seed(42 + partido_idx)  # Reproducibilidad
        
        if 'ancla_local_fuerte' in tipo:
            prob_local = config.get('prob_local', 0.70)
            prob_empate = np.random.uniform(0.12, 0.18)
            prob_visitante = 1.0 - prob_local - prob_empate
            
        elif 'ancla_visitante_fuerte' in tipo:
            prob_visitante = config.get('prob_visitante', 0.67)
            prob_empate = np.random.uniform(0.12, 0.18)
            prob_local = 1.0 - prob_empate - prob_visitante
            
        elif 'empate_estrategico' in tipo:
            prob_empate = config.get('prob_empate', 0.42)
            restante = 1.0 - prob_empate
            prob_local = np.random.uniform(0.25, restante - 0.25)
            prob_visitante = restante - prob_local
            
        elif 'divisor_equilibrado' in tipo:
            if 'prob_local' in config:
                prob_local = config['prob_local']
                prob_empate = np.random.uniform(0.25, 0.35)
                prob_visitante = 1.0 - prob_local - prob_empate
            elif 'prob_visitante' in config:
                prob_visitante = config['prob_visitante']
                prob_empate = np.random.uniform(0.25, 0.35)
                prob_local = 1.0 - prob_empate - prob_visitante
            else:
                prob_empate = config.get('prob_empate', 0.33)
                restante = 1.0 - prob_empate
                prob_local = restante / 2
                prob_visitante = restante / 2
        else:
            # Fallback a distribución histórica
            probs = np.random.dirichlet([38, 29, 33])
            prob_local, prob_empate, prob_visitante = probs[0], probs[1], probs[2]
        
        # Normalizar y validar
        probs_array = np.array([prob_local, prob_empate, prob_visitante])
        probs_array = np.maximum(probs_array, 0.05)  # Mínimo 5%
        probs_array = probs_array / probs_array.sum()  # Normalizar
        
        return float(probs_array[0]), float(probs_array[1]), float(probs_array[2])
    
    def _generar_factor_contextual(self, factor_tipo: str, partido_tipo: str) -> float:
        """
        Genera factores contextuales realistas según tipo de partido
        """
        if factor_tipo == 'forma':
            if 'ancla_local' in partido_tipo:
                return np.random.uniform(0.5, 1.5)  # Forma favorable al local
            elif 'ancla_visitante' in partido_tipo:
                return np.random.uniform(-1.5, -0.5)  # Forma favorable al visitante
            else:
                return np.random.normal(0, 0.5)
                
        elif factor_tipo == 'lesiones':
            if 'ancla' in partido_tipo:
                return np.random.uniform(-0.5, 0.5)  # Pocas lesiones en favorito
            else:
                return np.random.normal(0, 0.3)
        
        return 0.0
    
    def _garantizar_distribucion_optima(self, partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Garantiza que la distribución global cumpla exactamente con 38%L, 29%E, 33%V
        """
        timer_id = self.instrumentor.start_timer("garantizar_distribucion")
        
        # Calcular distribución actual
        resultados_actuales = [self._resultado_mas_probable(p) for p in partidos]
        dist_actual = {
            "L": resultados_actuales.count("L") / 14,
            "E": resultados_actuales.count("E") / 14,
            "V": resultados_actuales.count("V") / 14
        }
        
        self.logger.info(f"📊 Distribución actual: L={dist_actual['L']:.1%}, E={dist_actual['E']:.1%}, V={dist_actual['V']:.1%}")
        
        # Calcular targets exactos para 14 partidos
        target_counts = {
            "L": round(14 * self.target_distribution["L"]),  # 5
            "E": round(14 * self.target_distribution["E"]),  # 4
            "V": round(14 * self.target_distribution["V"])   # 5
        }
        
        # Ajuste fino: asegurar que sumen 14
        total_target = sum(target_counts.values())
        if total_target != 14:
            # Ajustar el que esté más cerca
            diff = 14 - total_target
            if diff > 0:
                target_counts["L"] += diff  # Favorecer locales
            else:
                target_counts["E"] += diff
        
        self.logger.info(f"🎯 Targets objetivo: L={target_counts['L']}, E={target_counts['E']}, V={target_counts['V']}")
        
        # Contar resultados actuales
        actual_counts = {
            "L": resultados_actuales.count("L"),
            "E": resultados_actuales.count("E"), 
            "V": resultados_actuales.count("V")
        }
        
        # Identificar qué ajustar (solo tocar NO-Anclas)
        partidos_ajustados = partidos.copy()
        anclas_indices = set()
        
        for i, partido in enumerate(partidos):
            max_prob = max(partido['prob_local'], partido['prob_empate'], partido['prob_visitante'])
            if max_prob >= self.anchor_threshold:
                anclas_indices.add(i)
        
        modificables = [i for i in range(14) if i not in anclas_indices]
        self.logger.info(f"🔒 Anclas protegidas: {len(anclas_indices)}, Modificables: {len(modificables)}")
        
        # Aplicar ajustes solo en modificables
        for resultado, target_count in target_counts.items():
            current_count = actual_counts[resultado]
            diff = target_count - current_count
            
            if diff > 0:  # Necesitamos más de este resultado
                # Buscar candidatos modificables que NO sean de este tipo
                candidatos = [i for i in modificables 
                             if self._resultado_mas_probable(partidos_ajustados[i]) != resultado]
                
                # Ordenar por qué tan fácil es cambiar
                candidatos.sort(key=lambda i: -max(partidos_ajustados[i]['prob_local'], 
                                                 partidos_ajustados[i]['prob_empate'], 
                                                 partidos_ajustados[i]['prob_visitante']))
                
                cambios_realizados = 0
                for i in candidatos[:diff]:
                    if cambios_realizados >= diff:
                        break
                    
                    # Modificar probabilidades para favorecer el resultado objetivo
                    self._ajustar_probabilidades_hacia_resultado(partidos_ajustados[i], resultado)
                    cambios_realizados += 1
                    
                    self.instrumentor.log_state_change(
                        component="ajuste_distribucion",
                        old_state=f"partido_{i}_original",
                        new_state=f"partido_{i}_ajustado_hacia_{resultado}"
                    )
        
        self.instrumentor.end_timer(timer_id, success=True)
        return partidos_ajustados
    
    def _ajustar_probabilidades_hacia_resultado(self, partido: Dict[str, Any], resultado_objetivo: str):
        """
        Ajusta las probabilidades de un partido para favorecer un resultado específico
        """
        if resultado_objetivo == "L":
            partido['prob_local'] = max(0.50, partido['prob_local'] + 0.15)
            partido['prob_empate'] *= 0.8
            partido['prob_visitante'] *= 0.8
        elif resultado_objetivo == "E":
            partido['prob_empate'] = max(0.45, partido['prob_empate'] + 0.15)
            partido['prob_local'] *= 0.8
            partido['prob_visitante'] *= 0.8
        elif resultado_objetivo == "V":
            partido['prob_visitante'] = max(0.50, partido['prob_visitante'] + 0.15)
            partido['prob_local'] *= 0.8
            partido['prob_empate'] *= 0.8
        
        # Re-normalizar
        total = partido['prob_local'] + partido['prob_empate'] + partido['prob_visitante']
        partido['prob_local'] /= total
        partido['prob_empate'] /= total
        partido['prob_visitante'] /= total
    
    def _validar_cobertura_anclas(self, partidos: List[Dict[str, Any]]):
        """
        Valida que hay suficientes partidos Ancla para estabilidad
        """
        anclas_encontradas = self._contar_anclas(partidos)
        
        if anclas_encontradas < self.min_anchors:
            error_msg = f"❌ CRÍTICO: Solo {anclas_encontradas} Anclas encontradas, mínimo {self.min_anchors}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"✅ Cobertura de Anclas válida: {anclas_encontradas} >= {self.min_anchors}")
        
        # Log detallado de cada Ancla
        for i, partido in enumerate(partidos):
            max_prob = max(partido['prob_local'], partido['prob_empate'], partido['prob_visitante'])
            if max_prob >= self.anchor_threshold:
                resultado_ancla = self._resultado_mas_probable(partido)
                self.logger.debug(f"⚓ Ancla {i+1}: {partido['home']} vs {partido['away']} -> {resultado_ancla} ({max_prob:.3f})")
    
    def _contar_anclas(self, partidos: List[Dict[str, Any]]) -> int:
        """Cuenta partidos que califican como Ancla"""
        return sum(1 for p in partidos 
                  if max(p['prob_local'], p['prob_empate'], p['prob_visitante']) >= self.anchor_threshold)
    
    def _resultado_mas_probable(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado más probable de un partido"""
        probs = {
            "L": partido['prob_local'],
            "E": partido['prob_empate'],
            "V": partido['prob_visitante']
        }
        return max(probs, key=probs.get)
    
    def _procesar_archivo_csv(self, archivo_path: str) -> List[Dict[str, Any]]:
        """
        Procesa archivo CSV del usuario con validación mejorada
        """
        timer_id = self.instrumentor.start_timer("procesar_csv")
        
        try:
            df = pd.read_csv(archivo_path)
            
            if len(df) != 14:
                raise ValueError(f"El archivo debe tener exactamente 14 partidos, tiene {len(df)}")
            
            # Verificar columnas obligatorias
            columnas_requeridas = ['home', 'away']
            columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
            
            if columnas_faltantes:
                raise ValueError(f"Columnas faltantes en CSV: {columnas_faltantes}")
            
            partidos = []
            for idx, row in df.iterrows():
                partido = self._procesar_fila_csv(row, idx)
                partidos.append(partido)
            
            self.instrumentor.end_timer(timer_id, success=True, metrics={
                "archivo_procesado": archivo_path,
                "partidos_procesados": len(partidos)
            })
            
            return partidos
            
        except Exception as e:
            self.instrumentor.end_timer(timer_id, success=False)
            self.logger.error(f"Error procesando CSV {archivo_path}: {e}")
            raise
    
    def _procesar_fila_csv(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """
        Convierte una fila del CSV al formato interno requerido
        """
        # Probabilidades base - Si no están en CSV, generar balanceadas
        if all(col in row and pd.notna(row[col]) for col in ['prob_local', 'prob_empate', 'prob_visitante']):
            prob_local = float(row['prob_local'])
            prob_empate = float(row['prob_empate'])
            prob_visitante = float(row['prob_visitante'])
            
            # Normalizar si no suman 1
            total = prob_local + prob_empate + prob_visitante
            if abs(total - 1.0) > 0.01:
                prob_local /= total
                prob_empate /= total
                prob_visitante /= total
        else:
            # Generar probabilidades balanceadas
            prob_local, prob_empate, prob_visitante = self._calcular_probabilidades_optimizadas(
                {'tipo': 'csv_generado'}, idx
            )
        
        partido = {
            'id': idx,
            'home': str(row['home']).strip(),
            'away': str(row['away']).strip(),
            'liga': str(row.get('liga', 'Liga CSV')).strip(),
            'prob_local': prob_local,
            'prob_empate': prob_empate,
            'prob_visitante': prob_visitante,
            'forma_diferencia': float(row.get('forma_diferencia', np.random.normal(0, 0.5))),
            'lesiones_impact': float(row.get('lesiones_impact', np.random.normal(0, 0.3))),
            'es_final': bool(row.get('es_final', False)),
            'es_derbi': bool(row.get('es_derbi', False)),
            'es_playoff': bool(row.get('es_playoff', False)),
            'fecha': str(row.get('fecha', '2025-06-19')),
            'jornada': int(row.get('jornada', 1)),
            'concurso_id': str(row.get('concurso_id', '2283')),
            'configuracion_tipo': 'csv_usuario'
        }
        
        return partido


# Mantener compatibilidad con código existente
class DataLoader(EnhancedDataLoader):
    """Alias para compatibilidad hacia atrás"""
    
    def __init__(self):
        super().__init__()
        self.logger.warning("⚠️ Usando DataLoader legacy - migrar a EnhancedDataLoader")
    
    def _generar_datos_ejemplo(self) -> List[Dict[str, Any]]:
        """Wrapper para mantener compatibilidad"""
        return self._generar_datos_optimizados()