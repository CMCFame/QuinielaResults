# progol_optimizer/main.py
"""
Orquestador Principal - Integra todos los componentes del pipeline
Basado en la metodología definitiva del documento técnico
"""

import logging
from typing import Dict, List, Any, Optional

class ProgolOptimizer:
    """
    Clase principal que orquesta todo el pipeline de optimización
    """
    
    def __init__(self, debug: bool = False):
        # Configurar logging
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Inicializar componentes
        self._inicializar_componentes()
        
        self.logger.info("ProgolOptimizer inicializado exitosamente")
    
    def _inicializar_componentes(self):
        """Inicializa todos los componentes del sistema"""
        try:
            from .data.loader import DataLoader
            from .data.validator import DataValidator
            from .models.classifier import PartidoClassifier
            from .models.calibrator import BayesianCalibrator
            from .portfolio.core_generator import CoreGenerator
            from .portfolio.satellite_generator import SatelliteGenerator
            from .portfolio.optimizer import GRASPAnnealing
            from .validation.portfolio_validator import PortfolioValidator
            from .export.exporter import PortfolioExporter
            
            self.data_loader = DataLoader()
            self.data_validator = DataValidator()
            self.classifier = PartidoClassifier()
            self.calibrator = BayesianCalibrator()
            self.core_generator = CoreGenerator()
            self.satellite_generator = SatelliteGenerator()
            self.optimizer = GRASPAnnealing()
            self.portfolio_validator = PortfolioValidator()
            self.exporter = PortfolioExporter()
            
            self.logger.debug("Todos los componentes inicializados correctamente")
            
        except ImportError as e:
            self.logger.error(f"Error importando componentes: {e}")
            raise
    
    def procesar_concurso(self, archivo_datos: str = None, concurso_id: str = "2283") -> Dict[str, Any]:
        """
        Procesa un concurso completo siguiendo el pipeline exacto del documento
        
        Args:
            archivo_datos: Ruta al archivo CSV con datos (opcional)
            concurso_id: ID del concurso
            
        Returns:
            Dict: Resultado completo con portafolio, validación y archivos
        """
        self.logger.info(f"=== INICIANDO PROCESAMIENTO CONCURSO {concurso_id} ===")
        
        try:
            # PASO 1: Cargar y validar datos
            self.logger.info("Paso 1: Cargando datos...")
            if archivo_datos:
                partidos = self.data_loader.cargar_datos(archivo_datos)
            else:
                partidos = self.data_loader._generar_datos_ejemplo()
            
            es_valido, errores = self.data_validator.validar_estructura(partidos)
            if not es_valido:
                raise ValueError(f"Datos inválidos: {errores}")
            
            self.logger.info(f"✅ Datos cargados y validados: {len(partidos)} partidos")
            
            # PASO 2: Clasificación y calibración
            self.logger.info("Paso 2: Clasificando y calibrando partidos...")
            partidos_procesados = []
            
            for i, partido in enumerate(partidos):
                # Calibración bayesiana
                partido_calibrado = self.calibrator.aplicar_calibracion_bayesiana(partido)
                
                # Clasificación según taxonomía
                clasificacion = self.classifier.clasificar_partido(partido_calibrado)
                
                partido_final = {
                    **partido_calibrado,
                    "id": i,
                    "clasificacion": clasificacion
                }
                partidos_procesados.append(partido_final)
            
            self.logger.info("✅ Clasificación y calibración completada")
            
            # PASO 3: Generación de Core
            self.logger.info("Paso 3: Generando quinielas Core...")
            quinielas_core = self.core_generator.generar_quinielas_core(partidos_procesados)
            self.logger.info(f"✅ Generadas {len(quinielas_core)} quinielas Core")
            
            # PASO 4: Generación de Satélites
            self.logger.info("Paso 4: Generando satélites anticorrelados...")
            quinielas_satelites = self.satellite_generator.generar_pares_satelites(
                partidos_procesados, 26
            )
            self.logger.info(f"✅ Generados {len(quinielas_satelites)} satélites en pares")
            
            # PASO 5: Optimización GRASP-Annealing
            self.logger.info("Paso 5: Ejecutando optimización GRASP-Annealing...")
            portafolio_inicial = quinielas_core + quinielas_satelites
            
            if len(portafolio_inicial) != 30:
                raise ValueError(f"Portafolio debe tener 30 quinielas, tiene {len(portafolio_inicial)}")
            
            portafolio_optimizado = self.optimizer.optimizar_portafolio_grasp_annealing(
                portafolio_inicial, partidos_procesados
            )
            self.logger.info("✅ Optimización completada")
            
            # PASO 6: Validación completa
            self.logger.info("Paso 6: Validando portafolio final...")
            resultado_validacion = self.portfolio_validator.validar_portafolio_completo(
                portafolio_optimizado
            )
            
            if resultado_validacion["es_valido"]:
                self.logger.info("✅ Portafolio válido - cumple todas las reglas")
            else:
                self.logger.warning("⚠️ Portafolio con advertencias - revisar validación")
            
            # PASO 7: Exportación
            self.logger.info("Paso 7: Exportando resultados...")
            archivos_exportados = self.exporter.exportar_portafolio_completo(
                portafolio_optimizado,
                partidos_procesados,
                resultado_validacion["metricas"],
                concurso_id
            )
            self.logger.info(f"✅ Exportados {len(archivos_exportados)} archivos")
            
            # Resultado final
            resultado = {
                "portafolio": portafolio_optimizado,
                "partidos": partidos_procesados,
                "validacion": resultado_validacion,
                "metricas": resultado_validacion["metricas"],
                "archivos_exportados": archivos_exportados,
                "concurso_id": concurso_id,
                "resumen": {
                    "total_quinielas": len(portafolio_optimizado),
                    "cores": len([q for q in portafolio_optimizado if q["tipo"] == "Core"]),
                    "satelites": len([q for q in portafolio_optimizado if q["tipo"] == "Satelite"]),
                    "es_valido": resultado_validacion["es_valido"]
                }
            }
            
            self.logger.info("=== PROCESAMIENTO COMPLETADO EXITOSAMENTE ===")
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento: {e}")
            raise
    
    def procesar_datos_directos(self, partidos: List[Dict[str, Any]], concurso_id: str = "2283") -> Dict[str, Any]:
        """
        Procesa datos que ya están en memoria (para interfaz Streamlit)
        
        Args:
            partidos: Lista de partidos ya cargados
            concurso_id: ID del concurso
            
        Returns:
            Dict: Resultado completo del procesamiento
        """
        # Reutilizar la lógica del procesamiento principal pero sin cargar datos
        self.logger.info(f"Procesando datos directos para concurso {concurso_id}")
        
        # Validar datos
        es_valido, errores = self.data_validator.validar_estructura(partidos)
        if not es_valido:
            raise ValueError(f"Datos inválidos: {errores}")
        
        # Continuar con el resto del pipeline como en procesar_concurso
        # pero sin la carga inicial de datos
        partidos_procesados = []
        
        for i, partido in enumerate(partidos):
            partido_calibrado = self.calibrator.aplicar_calibracion_bayesiana(partido)
            clasificacion = self.classifier.clasificar_partido(partido_calibrado)
            
            partido_final = {
                **partido_calibrado,
                "id": i,
                "clasificacion": clasificacion
            }
            partidos_procesados.append(partido_final)
        
        # Continuar con generación, optimización, validación y exportación
        quinielas_core = self.core_generator.generar_quinielas_core(partidos_procesados)
        quinielas_satelites = self.satellite_generator.generar_pares_satelites(partidos_procesados, 26)
        
        portafolio_inicial = quinielas_core + quinielas_satelites
        portafolio_optimizado = self.optimizer.optimizar_portafolio_grasp_annealing(
            portafolio_inicial, partidos_procesados
        )
        
        resultado_validacion = self.portfolio_validator.validar_portafolio_completo(portafolio_optimizado)
        
        archivos_exportados = self.exporter.exportar_portafolio_completo(
            portafolio_optimizado,
            partidos_procesados,
            resultado_validacion["metricas"],
            concurso_id
        )
        
        return {
            "portafolio": portafolio_optimizado,
            "partidos": partidos_procesados,
            "validacion": resultado_validacion,
            "metricas": resultado_validacion["metricas"],
            "archivos_exportados": archivos_exportados,
            "concurso_id": concurso_id,
            "resumen": {
                "total_quinielas": len(portafolio_optimizado),
                "cores": len([q for q in portafolio_optimizado if q["tipo"] == "Core"]),
                "satelites": len([q for q in portafolio_optimizado if q["tipo"] == "Satelite"]),
                "es_valido": resultado_validacion["es_valido"]
            }
        }