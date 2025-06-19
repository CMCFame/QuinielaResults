# progol_optimizer/main.py
"""
Orquestador Principal - VERSIÓN CON AI AGRESIVA
AI interviene automáticamente cuando detecta problemas
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Imports corregidos
from data.loader import DataLoader
from data.validator import DataValidator
from models.classifier import PartidoClassifier
from models.calibrator import BayesianCalibrator
from models.ai_assistant import ProgolAIAssistant
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
    Clase principal con AI agresiva que corrige automáticamente
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurar API key si está en secrets
        try:
            import streamlit as st
            if "OPENAI_API_KEY" in st.secrets:
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        except:
            pass
        
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
        
        # Inicializar asistente AI
        self.ai_assistant = ProgolAIAssistant()
        
        self.logger.info("✅ ProgolOptimizer inicializado correctamente")
    
    def procesar_concurso(self, archivo_datos: str = None, concurso_id: str = "2283", 
                         forzar_ai: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo con AI agresiva
        Args:
            forzar_ai: Si True, usa AI incluso si el portafolio inicial es válido
        """
        self.logger.info(f"=== PROCESANDO CONCURSO {concurso_id} ===")
        
        try:
            # PASO 1-3: Preparación de datos (sin cambios)
            self.logger.info("PASO 1: Cargando datos...")
            if archivo_datos:
                partidos = self.data_loader.cargar_datos(archivo_datos)
            else:
                partidos = self.data_loader._generar_datos_ejemplo()
            
            self.logger.info("PASO 2: Validando estructura de datos...")
            es_valido, errores = self.data_validator.validar_estructura(partidos)
            if not es_valido:
                raise ValueError(f"Datos inválidos: {errores}")
            
            self.logger.info("PASO 3: Aplicando calibración bayesiana global...")
            partidos_calibrados = self.calibrator.calibrar_concurso_completo(partidos)
            
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
            
            # PASO 4-6: Generación y optimización inicial
            self.logger.info("PASO 4: Generando 4 quinielas Core...")
            quinielas_core = self.core_generator.generar_quinielas_core(partidos_procesados)
            
            self.logger.info("PASO 5: Generando 26 satélites en pares...")
            quinielas_satelites = self.satellite_generator.generar_pares_satelites(
                partidos_procesados, 26
            )
            
            self.logger.info("PASO 6: Ejecutando optimización GRASP-Annealing...")
            portafolio_inicial = quinielas_core + quinielas_satelites
            portafolio_optimizado = self.optimizer.optimizar_portafolio_grasp_annealing(
                portafolio_inicial, partidos_procesados
            )
            
            # PASO 7: CORRECCIÓN AGRESIVA CON AI
            validacion_inicial = self.portfolio_validator.validar_portafolio_completo(portafolio_optimizado)
            
            if (not validacion_inicial["es_valido"] or forzar_ai) and self.ai_assistant.enabled:
                self.logger.info("🤖 ACTIVANDO CORRECCIÓN AGRESIVA CON AI...")
                
                # Contar intentos para evitar loops infinitos
                max_intentos_ai = 3
                intento_actual = 0
                
                while intento_actual < max_intentos_ai:
                    intento_actual += 1
                    self.logger.info(f"🔄 Intento AI #{intento_actual}")
                    
                    # PASO 1: Corregir quinielas individuales problemáticas
                    portafolio_corregido = self._corregir_quinielas_con_ai(
                        portafolio_optimizado, partidos_procesados
                    )
                    
                    # PASO 2: Optimización global con AI
                    if self._necesita_optimizacion_global(portafolio_corregido):
                        self.logger.info("🌐 Aplicando optimización global con AI...")
                        portafolio_corregido = self.ai_assistant.optimizar_distribucion_global(
                            portafolio_corregido, partidos_procesados
                        )
                    
                    # Validar resultado
                    validacion_ai = self.portfolio_validator.validar_portafolio_completo(portafolio_corregido)
                    
                    if validacion_ai["es_valido"]:
                        self.logger.info(f"✅ AI logró corregir el portafolio en intento #{intento_actual}")
                        portafolio_optimizado = portafolio_corregido
                        resultado_validacion = validacion_ai
                        break
                    else:
                        self.logger.warning(f"⚠️ Intento #{intento_actual} no resolvió todos los problemas")
                        portafolio_optimizado = portafolio_corregido  # Usar la versión mejorada
                        
                        if intento_actual == max_intentos_ai:
                            self.logger.error("❌ AI no pudo resolver todos los problemas después de 3 intentos")
                            resultado_validacion = validacion_ai
                            
            else:
                resultado_validacion = validacion_inicial
                if not self.ai_assistant.enabled:
                    self.logger.warning("⚠️ AI no disponible para correcciones")
            
            # PASO 8: Exportar resultados
            self.logger.info("PASO 8: Exportando resultados...")
            archivos_exportados = self.exporter.exportar_portafolio_completo(
                portafolio_optimizado,
                partidos_procesados,
                resultado_validacion["metricas"],
                concurso_id
            )
            
            # Resultado final
            es_valido_final = resultado_validacion["es_valido"]
            self.logger.info(f"📊 RESULTADO FINAL: {'✅ VÁLIDO' if es_valido_final else '❌ INVÁLIDO'}")
            
            resultado = {
                "success": True,
                "portafolio": portafolio_optimizado,
                "partidos": partidos_procesados,
                "validacion": resultado_validacion,
                "metricas": resultado_validacion["metricas"],
                "estadisticas_clasificacion": stats_clasificacion,
                "archivos_exportados": archivos_exportados,
                "concurso_id": concurso_id,
                "ai_utilizada": self.ai_assistant.enabled and (not validacion_inicial["es_valido"] or forzar_ai)
            }
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando concurso: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "concurso_id": concurso_id
            }
    
    def _corregir_quinielas_con_ai(self, portafolio: List[Dict[str, Any]], 
                                   partidos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Corrige agresivamente las quinielas problemáticas con AI
        """
        self.logger.info("🔧 Corrigiendo quinielas individuales con AI...")
        
        portafolio_corregido = []
        quinielas_corregidas = 0
        
        for quiniela in portafolio:
            problemas = self._detectar_problemas_quiniela(quiniela)
            
            # Solo corregir si hay problemas Y no es Core (o si tiene problemas graves)
            if problemas and (quiniela["tipo"] != "Core" or len(problemas) > 1):
                self.logger.debug(f"Corrigiendo {quiniela['id']}: {problemas}")
                
                quiniela_corregida = self.ai_assistant.corregir_quiniela_invalida(
                    quiniela, partidos, problemas
                )
                
                if quiniela_corregida:
                    # Verificar que la corrección es mejor
                    problemas_despues = self._detectar_problemas_quiniela(quiniela_corregida)
                    if len(problemas_despues) < len(problemas):
                        portafolio_corregido.append(quiniela_corregida)
                        quinielas_corregidas += 1
                    else:
                        self.logger.warning(f"Corrección de {quiniela['id']} no mejoró, manteniendo original")
                        portafolio_corregido.append(quiniela)
                else:
                    portafolio_corregido.append(quiniela)
            else:
                portafolio_corregido.append(quiniela)
        
        self.logger.info(f"✅ AI corrigió {quinielas_corregidas} quinielas problemáticas")
        return portafolio_corregido
    
    def _detectar_problemas_quiniela(self, quiniela: Dict[str, Any]) -> List[str]:
        """
        Detecta todos los problemas de una quiniela
        """
        problemas = []
        
        # Empates
        if not (4 <= quiniela["empates"] <= 6):
            problemas.append(f"empates={quiniela['empates']} (debe ser 4-6)")
        
        # Concentración general
        max_conc = max(quiniela["distribución"].values()) / 14
        if max_conc > 0.70:
            problemas.append(f"concentración general {max_conc:.1%} > 70%")
        
        # Concentración inicial
        primeros_3 = quiniela["resultados"][:3]
        max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3
        if max_conc_inicial > 0.60:
            problemas.append(f"concentración inicial {max_conc_inicial:.1%} > 60%")
        
        return problemas
    
    def _necesita_optimizacion_global(self, portafolio: List[Dict[str, Any]]) -> bool:
        """
        Determina si el portafolio necesita optimización global
        """
        # Calcular distribución global
        total_L = sum(q["distribución"]["L"] for q in portafolio)
        total_E = sum(q["distribución"]["E"] for q in portafolio)
        total_V = sum(q["distribución"]["V"] for q in portafolio)
        total = total_L + total_E + total_V
        
        porc_L = total_L / total if total > 0 else 0
        porc_E = total_E / total if total > 0 else 0
        porc_V = total_V / total if total > 0 else 0
        
        # Verificar si está fuera de rangos
        fuera_L = not (0.35 <= porc_L <= 0.41)
        fuera_E = not (0.25 <= porc_E <= 0.33)
        fuera_V = not (0.30 <= porc_V <= 0.36)
        
        return fuera_L or fuera_E or fuera_V

def main():
    """Función principal para uso por línea de comandos"""
    import argparse
    parser = argparse.ArgumentParser(description="Progol Optimizer - Metodología Definitiva")
    parser.add_argument("--archivo", "-f", help="Archivo CSV con datos de partidos")
    parser.add_argument("--concurso", "-c", default="2283", help="ID del concurso")
    parser.add_argument("--debug", "-d", action="store_true", help="Modo debug")
    parser.add_argument("--api-key", "-k", help="OpenAI API key (opcional)")
    parser.add_argument("--forzar-ai", "-ai", action="store_true", help="Forzar uso de AI")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    optimizer = ProgolOptimizer()
    resultado = optimizer.procesar_concurso(args.archivo, args.concurso, args.forzar_ai)
    
    if resultado["success"]:
        print(f"✅ Optimización exitosa para concurso {args.concurso}")
        print(f"   AI utilizada: {'Sí' if resultado.get('ai_utilizada') else 'No'}")
        print(f"   Portafolio válido: {'Sí' if resultado['validacion']['es_valido'] else 'No'}")
        print(f"   Archivos generados en: outputs/")
    else:
        print(f"❌ Error: {resultado['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()