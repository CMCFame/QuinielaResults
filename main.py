# progol_optimizer/main.py
"""
Orquestador Principal - Ejecuta el pipeline completo de optimización
Integra todos los componentes según la metodología del documento
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ProgolOptimizer:
    """
    Clase principal que orquesta todo el pipeline de optimización
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar componentes
        try:
            from progol_optimizer.data.loader import DataLoader
            from progol_optimizer.data.validator import DataValidator
            from progol_optimizer.models.classifier import PartidoClassifier
            from progol_optimizer.models.calibrator import BayesianCalibrator
            from progol_optimizer.portfolio.core_generator import CoreGenerator
            from progol_optimizer.portfolio.satellite_generator import SatelliteGenerator
            from progol_optimizer.portfolio.optimizer import GRASPAnnealing
            from progol_optimizer.validation.portfolio_validator import PortfolioValidator
            from progol_optimizer.export.exporter import PortfolioExporter
        except ImportError as e:
            self.logger.error(f"Error importando componentes: {e}")
            # Fallback imports usando paths relativos
            sys.path.append(str(Path(__file__).parent))
            
            from data.loader import DataLoader
            from data.validator import DataValidator
            from models.classifier import PartidoClassifier
            from models.calibrator import BayesianCalibrator
            from portfolio.core_generator import CoreGenerator
            from portfolio.satellite_generator import SatelliteGenerator
            from portfolio.optimizer import GRASPAnnealing
            from validation.portfolio_validator import PortfolioValidator
            from export.exporter import PortfolioExporter
        
        # Inicializar componentes
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.classifier = PartidoClassifier()
        self.calibrator = BayesianCalibrator()
        self.core_generator = CoreGenerator()
        self.satellite_generator = SatelliteGenerator()
        self.optimizer = GRASPAnnealing()
        self.portfolio_validator = PortfolioValidator()
        self.exporter = PortfolioExporter()
        
        self.logger.info("✅ ProgolOptimizer inicializado correctamente")
    
    def procesar_concurso(self, archivo_datos: str = None, concurso_id: str = "2283") -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo para un concurso
        
        Args:
            archivo_datos: Ruta al archivo CSV (opcional)
            concurso_id: ID del concurso
            
        Returns:
            Dict con todos los resultados
        """
        self.logger.info(f"=== PROCESANDO CONCURSO {concurso_id} ===")
        
        try:
            # PASO 1: Cargar datos
            self.logger.info("PASO 1: Cargando datos...")
            if archivo_datos:
                partidos = self.data_loader.cargar_datos(archivo_datos)
            else:
                partidos = self.data_loader._generar_datos_ejemplo()
            
            # PASO 2: Validar datos
            self.logger.info("PASO 2: Validando estructura de datos...")
            es_valido, errores = self.data_validator.validar_estructura(partidos)
            if not es_valido:
                raise ValueError(f"Datos inválidos: {errores}")
            
            # PASO 3: Clasificar y calibrar partidos
            self.logger.info("PASO 3: Clasificando y calibrando partidos...")
            partidos_procesados = []
            for i, partido in enumerate(partidos):
                # Calibración bayesiana
                partido_calibrado = self.calibrator.aplicar_calibracion_bayesiana(partido)
                
                # Clasificación
                clasificacion = self.classifier.clasificar_partido(partido_calibrado)
                
                # Agregar metadatos
                partido_final = {
                    **partido_calibrado,
                    "id": i,
                    "clasificacion": clasificacion
                }
                partidos_procesados.append(partido_final)
            
            # PASO 4: Generar quinielas Core
            self.logger.info("PASO 4: Generando 4 quinielas Core...")
            quinielas_core = self.core_generator.generar_quinielas_core(partidos_procesados)
            
            # PASO 5: Generar satélites
            self.logger.info("PASO 5: Generando 26 satélites en pares...")
            quinielas_satelites = self.satellite_generator.generar_pares_satelites(
                partidos_procesados, 26
            )
            
            # PASO 6: Optimizar portafolio
            self.logger.info("PASO 6: Ejecutando optimización GRASP-Annealing...")
            portafolio_inicial = quinielas_core + quinielas_satelites
            portafolio_optimizado = self.optimizer.optimizar_portafolio_grasp_annealing(
                portafolio_inicial, partidos_procesados
            )
            
            # PASO 7: Validar portafolio final
            self.logger.info("PASO 7: Validando portafolio final...")
            resultado_validacion = self.portfolio_validator.validar_portafolio_completo(
                portafolio_optimizado
            )
            
            # PASO 8: Exportar resultados
            self.logger.info("PASO 8: Exportando resultados...")
            archivos_exportados = self.exporter.exportar_portafolio_completo(
                portafolio_optimizado,
                partidos_procesados,
                resultado_validacion["metricas"],
                concurso_id
            )
            
            # Resultado final
            resultado = {
                "success": True,
                "portafolio": portafolio_optimizado,
                "partidos": partidos_procesados,
                "validacion": resultado_validacion,
                "metricas": resultado_validacion["metricas"],
                "archivos_exportados": archivos_exportados,
                "concurso_id": concurso_id
            }
            
            self.logger.info(f"✅ CONCURSO {concurso_id} PROCESADO EXITOSAMENTE")
            self.logger.info(f"   → {len(portafolio_optimizado)} quinielas generadas")
            self.logger.info(f"   → Validación: {'✅ VÁLIDO' if resultado_validacion['es_valido'] else '❌ INVÁLIDO'}")
            self.logger.info(f"   → {len(archivos_exportados)} archivos exportados")
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando concurso: {e}")
            return {
                "success": False,
                "error": str(e),
                "concurso_id": concurso_id
            }


def main():
    """Función principal para uso por línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progol Optimizer - Metodología Definitiva")
    parser.add_argument("--archivo", "-f", help="Archivo CSV con datos de partidos")
    parser.add_argument("--concurso", "-c", default="2283", help="ID del concurso")
    parser.add_argument("--debug", "-d", action="store_true", help="Modo debug")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ejecutar optimización
    optimizer = ProgolOptimizer()
    resultado = optimizer.procesar_concurso(args.archivo, args.concurso)
    
    if resultado["success"]:
        print(f"✅ Optimización exitosa para concurso {args.concurso}")
        print(f"   Archivos generados en: outputs/")
    else:
        print(f"❌ Error: {resultado['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()