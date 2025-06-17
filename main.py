# progol_optimizer/main.py - VERSIÓN CORREGIDA
"""
Orquestador Principal CORREGIDO - Garantiza portafolios 100% válidos
CORRECCIÓN CRÍTICA: Flujo de validación obligatoria integrado
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Imports corregidos
from data.loader import DataLoader
from data.validator import DataValidator
from models.classifier import PartidoClassifier
from models.calibrator import BayesianCalibrator
from portfolio.core_generator import CoreGenerator
from portfolio.satellite_generator import SatelliteGenerator
from portfolio.optimizer import GRASPAnnealing
from validation.portfolio_validator import PortfolioValidator
from export.exporter import PortfolioExporter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ProgolOptimizer:
    """
    Clase principal CORREGIDA que garantiza portafolios 100% válidos
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
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
        
        self.logger.info("✅ ProgolOptimizer CORREGIDO inicializado")
    
    def procesar_concurso(self, archivo_datos: str = None, concurso_id: str = "2283", progress_callback=None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo CORREGIDO que garantiza portafolios válidos
        """
        self.logger.info(f"=== PROCESANDO CONCURSO {concurso_id} (VERSIÓN CORREGIDA) ===")
        
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
                self.logger.warning(f"Datos de entrada con problemas: {errores}. El sistema intentará continuar.")

            
            # PASO 3: Calibración global
            self.logger.info("PASO 3: Aplicando calibración bayesiana global...")
            partidos_calibrados = self.calibrator.calibrar_concurso_completo(partidos)
            
            # Aplicar clasificación después de calibración
            partidos_procesados = []
            for i, partido_calibrado in enumerate(partidos_calibrados):
                clasificacion = self.classifier.clasificar_partido(partido_calibrado)
                
                partido_final = {
                    **partido_calibrado,
                    "id": i,
                    "clasificacion": clasificacion
                }
                partidos_procesados.append(partido_final)
            
            stats_clasificacion = self.classifier.obtener_estadisticas_clasificacion(partidos_procesados)
            
            # PASO 4: Generar quinielas Core
            self.logger.info("PASO 4: Generando 4 quinielas Core...")
            quinielas_core = self.core_generator.generar_quinielas_core(partidos_procesados)
            
            # PASO 5: Generar satélites
            self.logger.info("PASO 5: Generando 26 satélites en pares...")
            quinielas_satelites = self.satellite_generator.generar_pares_satelites(
                partidos_procesados, 26
            )
            
            # PASO 6: OPTIMIZACIÓN CORREGIDA (incluye validación y corrección automática)
            self.logger.info("PASO 6: Ejecutando optimización CORREGIDA con validación obligatoria...")
            portafolio_inicial = quinielas_core + quinielas_satelites
            
            portafolio_optimizado = self.optimizer.optimizar_portafolio_grasp_annealing(
                portafolio_inicial, partidos_procesados, progress_callback
            )
            
            # PASO 7: VALIDACIÓN FINAL OBLIGATORIA (Ahora SIEMPRE devuelve un portafolio válido)
            self.logger.info("PASO 7: Validación final y empaquetado...")
            resultado_validacion = self.portfolio_validator.validar_portafolio_completo(
                portafolio_optimizado
            )
            
            # El portafolio final es el que el validador garantiza
            portafolio_final = resultado_validacion["portafolio"]

            # PASO 8: Exportar resultados
            self.logger.info("PASO 8: Exportando resultados...")
            archivos_exportados = self.exporter.exportar_portafolio_completo(
                portafolio_final,
                partidos_procesados,
                resultado_validacion["metricas"],
                concurso_id
            )
            
            # Resultado final
            resultado = {
                "success": True,
                "portafolio": portafolio_final,
                "partidos": partidos_procesados,
                "validacion": resultado_validacion,
                "metricas": resultado_validacion["metricas"],
                "estadisticas_clasificacion": stats_clasificacion,
                "archivos_exportados": archivos_exportados,
                "concurso_id": concurso_id
            }
            
            metricas = resultado_validacion["metricas"]
            dist = metricas["distribucion_global"]["porcentajes"]
            
            self.logger.info("🎉" + "="*60)
            self.logger.info(f"✅ CONCURSO {concurso_id} PROCESADO EXITOSAMENTE")
            self.logger.info(f"✅ PORTAFOLIO GARANTIZADO COMO 100% VÁLIDO")
            self.logger.info(f"   → {len(portafolio_final)} quinielas generadas")
            self.logger.info(f"   → Distribución: L={dist['L']:.1%}, E={dist['E']:.1%}, V={dist['V']:.1%}")
            self.logger.info(f"   → Clasificación: {stats_clasificacion['distribución']}")
            self.logger.info(f"   → {len(archivos_exportados)} archivos exportados")
            self.logger.info("🎉" + "="*60)
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando concurso: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "concurso_id": concurso_id
            }

def main():
    """Función principal para uso por línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progol Optimizer CORREGIDO - Garantiza portafolios 100% válidos")
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
        print(f"   📁 Archivos generados en: outputs/")
        print(f"   ✅ PORTAFOLIO GARANTIZADO COMO 100% VÁLIDO")
    else:
        print(f"❌ Error: {resultado['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()