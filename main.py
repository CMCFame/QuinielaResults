# progol_optimizer/main.py - VERSIÓN COMPLETA con todas las mejoras
"""
Orquestador Principal - Ejecuta el pipeline completo de optimización
VERSIÓN COMPLETA con todas las mejoras opcionales + PASO 3 actualizado
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Imports originales (sin cambios)
from data.loader import DataLoader
from data.validator import DataValidator
from models.classifier import PartidoClassifier
from models.calibrator import BayesianCalibrator
from portfolio.core_generator import CoreGenerator
from portfolio.satellite_generator import SatelliteGenerator
from portfolio.optimizer import GRASPAnnealing
from validation.portfolio_validator import PortfolioValidator
from export.exporter import PortfolioExporter

# MEJORAS OPCIONALES: Verificar dependencias opcionales
try:
    from data.pdf_parser import PreviasParser
    PDF_PARSER_AVAILABLE = True
except ImportError:
    PDF_PARSER_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Configurar logging mejorado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ProgolOptimizer:
    """
    Clase principal que orquesta todo el pipeline de optimización
    VERSIÓN COMPLETA con optimizaciones de velocidad y fidelidad
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # MEJORA OPCIONAL: Log de dependencias disponibles
        self._verificar_dependencias()
        
        # Inicializar componentes (sin cambios)
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.classifier = PartidoClassifier()
        self.calibrator = BayesianCalibrator()  # Ahora con regularización global
        self.core_generator = CoreGenerator()
        self.satellite_generator = SatelliteGenerator()  # Ahora más robusto
        self.optimizer = GRASPAnnealing()  # Ahora 5x más rápido
        self.portfolio_validator = PortfolioValidator()
        self.exporter = PortfolioExporter()
        
        # MEJORA OPCIONAL: Parser PDF opcional
        if PDF_PARSER_AVAILABLE:
            self.pdf_parser = PreviasParser()
            self.logger.info("✅ Parser PDF de previas disponible")
        
        self.logger.info("✅ ProgolOptimizer v2.0 inicializado correctamente")
    
    def _verificar_dependencias(self):
        """
        MEJORA OPCIONAL: Verifica dependencias opcionales y reporta estado
        """
        self.logger.info("🔍 Verificando dependencias...")
        
        if NUMBA_AVAILABLE:
            self.logger.info("✅ Numba JIT disponible - Optimización de velocidad activada")
        else:
            self.logger.warning("⚠️ Numba no disponible - Velocidad reducida")
        
        if PDF_PARSER_AVAILABLE:
            self.logger.info("✅ Parser PDF disponible - Puede procesar previas")
        else:
            self.logger.info("ℹ️ Parser PDF no disponible - Usando datos simulados")
        
        # Verificar scipy.stats.poisson_binomial
        try:
            from scipy.stats import poisson_binomial
            self.logger.info("✅ Poisson-Binomial exacto disponible")
        except ImportError:
            self.logger.warning("⚠️ Fallback a Monte Carlo para probabilidades")
    
    def procesar_concurso(self, archivo_datos: str = None, concurso_id: str = "2283", 
                         archivo_previas_pdf: str = None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo para un concurso
        MEJORA OPCIONAL: Soporte opcional para previas PDF
        
        Args:
            archivo_datos: Ruta al archivo CSV (opcional)
            concurso_id: ID del concurso
            archivo_previas_pdf: MEJORA OPCIONAL - Ruta al PDF de previas (opcional)
            
        Returns:
            Dict con todos los resultados
        """
        self.logger.info(f"=== PROCESANDO CONCURSO {concurso_id} (v2.0) ===")
        
        try:
            # PASO 1: Cargar datos (sin cambios)
            self.logger.info("PASO 1: Cargando datos...")
            if archivo_datos:
                partidos = self.data_loader.cargar_datos(archivo_datos)
            else:
                partidos = self.data_loader._generar_datos_ejemplo()
            
            # PASO 1.5: MEJORA OPCIONAL - Enriquecer con PDF de previas si está disponible
            if archivo_previas_pdf and PDF_PARSER_AVAILABLE:
                self.logger.info("PASO 1.5: Procesando previas PDF...")
                try:
                    datos_previas = self.pdf_parser.parse_pdf_previas(archivo_previas_pdf)
                    partidos = self._enriquecer_con_previas(partidos, datos_previas)
                    self.logger.info(f"✅ Enriquecidos {len(partidos)} partidos con datos PDF")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error procesando PDF previas: {e}")
            
            # PASO 2: Validar datos
            self.logger.info("PASO 2: Validando estructura de datos...")
            es_valido, errores = self.data_validator.validar_estructura(partidos)
            if not es_valido:
                raise ValueError(f"Datos inválidos: {errores}")
            
            # ====================================================================
            # PASO 3: ACTUALIZADO - Clasificar y calibrar partidos CON REGULARIZACIÓN GLOBAL
            # ====================================================================
            self.logger.info("PASO 3: Clasificando y calibrando partidos CON REGULARIZACIÓN GLOBAL...")
            
            # NUEVO: Calibración global en lugar de individual
            # Esto aplica regularización automática para balancear la distribución
            partidos_calibrados = self.calibrator.calibrar_concurso_completo(partidos)
            
            # Aplicar clasificación después de calibración
            partidos_procesados = []
            for i, partido_calibrado in enumerate(partidos_calibrados):
                # Clasificación
                clasificacion = self.classifier.clasificar_partido(partido_calibrado)
                
                # Agregar metadatos
                partido_final = {
                    **partido_calibrado,
                    "id": i,
                    "clasificacion": clasificacion
                }
                partidos_procesados.append(partido_final)
            
            self.logger.info(f"✅ PASO 3 completado: {len(partidos_procesados)} partidos calibrados y clasificados")
            # ====================================================================
            # FIN PASO 3 ACTUALIZADO
            # ====================================================================
            
            # PASO 4: Generar quinielas Core
            self.logger.info("PASO 4: Generando 4 quinielas Core...")
            quinielas_core = self.core_generator.generar_quinielas_core(partidos_procesados)
            
            # PASO 5: Generar satélites ROBUSTOS
            self.logger.info("PASO 5: Generando 26 satélites robustos en pares...")
            quinielas_satelites = self.satellite_generator.generar_pares_satelites(
                partidos_procesados, 26
            )
            
            # PASO 6: Optimizar portafolio RÁPIDO
            self.logger.info("PASO 6: Ejecutando optimización GRASP-Annealing OPTIMIZADA...")
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
            
            # Resultado final con mejoras opcionales
            resultado = {
                "success": True,
                "portafolio": portafolio_optimizado,
                "partidos": partidos_procesados,
                "validacion": resultado_validacion,
                "metricas": resultado_validacion["metricas"],
                "archivos_exportados": archivos_exportados,
                "concurso_id": concurso_id,
                "version": "2.0",  # MEJORA OPCIONAL
                "mejoras_aplicadas": {  # MEJORA OPCIONAL
                    "bivariate_poisson": self.calibrator.usar_bivariate_poisson,
                    "numba_optimizado": NUMBA_AVAILABLE,
                    "pdf_procesado": archivo_previas_pdf is not None and PDF_PARSER_AVAILABLE,
                    "regularizacion_global": True  # Nueva característica
                }
            }
            
            self.logger.info(f"✅ CONCURSO {concurso_id} PROCESADO EXITOSAMENTE (v2.0)")
            self.logger.info(f"   → {len(portafolio_optimizado)} quinielas generadas")
            self.logger.info(f"   → Validación: {'✅ VÁLIDO' if resultado_validacion['es_valido'] else '❌ INVÁLIDO'}")
            self.logger.info(f"   → {len(archivos_exportados)} archivos exportados")
            self.logger.info(f"   → Mejoras v2.0: Bivariate-Poisson={self.calibrator.usar_bivariate_poisson}, Numba={NUMBA_AVAILABLE}")
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando concurso: {e}")
            return {
                "success": False,
                "error": str(e),
                "concurso_id": concurso_id,
                "version": "2.0"
            }
    
    def _enriquecer_con_previas(self, partidos: List[Dict[str, Any]], 
                               datos_previas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        MEJORA OPCIONAL: Enriquece partidos básicos con datos extraídos de previas PDF
        """
        if not datos_previas:
            return partidos
        
        # Mapear datos de previas a partidos (por índice o nombre)
        partidos_enriquecidos = []
        
        for i, partido in enumerate(partidos):
            partido_enriquecido = partido.copy()
            
            # Si hay datos de previas para este partido
            if i < len(datos_previas):
                previa = datos_previas[i]
                
                # Sobrescribir con datos más ricos de PDF
                partido_enriquecido.update({
                    "forma_diferencia": previa.get("forma_diferencia", partido.get("forma_diferencia", 0)),
                    "lesiones_impact": previa.get("lesiones_impact", partido.get("lesiones_impact", 0)),
                    "es_final": previa.get("es_final", partido.get("es_final", False)),
                    "es_derbi": previa.get("es_derbi", partido.get("es_derbi", False)),
                    "es_playoff": previa.get("es_playoff", partido.get("es_playoff", False)),
                    "h2h_ratio": previa.get("h2h_ratio", 0),
                    "confidence_previas": previa.get("confidence_score", 0)
                })
                
                self.logger.debug(f"Partido {i}: enriquecido con previas (confidence={previa.get('confidence_score', 0):.2f})")
            
            partidos_enriquecidos.append(partido_enriquecido)
        
        return partidos_enriquecidos


def main():
    """Función principal para uso por línea de comandos MEJORADA"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progol Optimizer v2.0 - Metodología Definitiva")
    parser.add_argument("--archivo", "-f", help="Archivo CSV con datos de partidos")
    parser.add_argument("--concurso", "-c", default="2283", help="ID del concurso")
    parser.add_argument("--previas", "-p", help="MEJORA OPCIONAL: Archivo PDF con previas")
    parser.add_argument("--debug", "-d", action="store_true", help="Modo debug")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ejecutar optimización con previas opcionales
    optimizer = ProgolOptimizer()
    resultado = optimizer.procesar_concurso(
        archivo_datos=args.archivo, 
        concurso_id=args.concurso,
        archivo_previas_pdf=args.previas  # MEJORA OPCIONAL
    )
    
    if resultado["success"]:
        print(f"✅ Optimización v2.0 exitosa para concurso {args.concurso}")
        print(f"   Archivos generados en: outputs/")
        if "mejoras_aplicadas" in resultado:
            mejoras = resultado["mejoras_aplicadas"]
            print(f"   Bivariate-Poisson: {'✅' if mejoras['bivariate_poisson'] else '❌'}")
            print(f"   Optimización Numba: {'✅' if mejoras['numba_optimizado'] else '❌'}")
            print(f"   PDF procesado: {'✅' if mejoras['pdf_procesado'] else '❌'}")
            print(f"   Regularización global: {'✅' if mejoras['regularizacion_global'] else '❌'}")
    else:
        print(f"❌ Error: {resultado['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()