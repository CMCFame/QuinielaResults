# progol_optimizer/main.py
"""
Orquestador Principal - VERSIÓN CON AI AGRESIVA Y MÚLTIPLES MÉTODOS
Permite seleccionar entre el optimizador Híbrido y el Heredado.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Imports corregidos
from data.loader import DataLoader
from data.validator import DataValidator
from models.classifier import PartidoClassifier
from models.calibrator import BayesianCalibrator
from models.ai_assistant import ProgolAIAssistant
from portfolio.core_generator import CoreGenerator
from portfolio.satellite_generator import SatelliteGenerator
from portfolio.optimizer import GRASPAnnealing
# NUEVA IMPORTACIÓN: El motor Híbrido
from portfolio.hybrid_optimizer import HybridOptimizer
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
        self.portfolio_validator = PortfolioValidator()
        self.exporter = PortfolioExporter()
        self.ai_assistant = ProgolAIAssistant()
        
        # El optimizador heredado se instancia bajo demanda
        self.legacy_optimizer = GRASPAnnealing()
        
        self.logger.info("✅ ProgolOptimizer inicializado correctamente")
    
    def procesar_concurso(self, archivo_datos: str = None, concurso_id: str = "2283", 
                         forzar_ai: bool = False, method: str = "hybrid") -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo con AI agresiva
        Args:
            forzar_ai: Si True, usa AI incluso si el portafolio inicial es válido.
            method: 'hybrid' (recomendado) o 'legacy' para el método de optimización.
        """
        self.logger.info(f"=== PROCESANDO CONCURSO {concurso_id} CON MÉTODO: {method.upper()} ===")
        
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
            
            # --- SELECCIÓN DE MÉTODO DE OPTIMIZACIÓN ---
            if method == "hybrid":
                self.logger.info("PASO 4-6: Ejecutando optimización Híbrida (IP + Annealing)...")
                try:
                    # CORRECCIÓN: Importación condicional para evitar errores
                    from portfolio.hybrid_optimizer import HybridOptimizer
                    
                    partidos_df = pd.DataFrame(partidos_procesados)
                    hybrid_opt = HybridOptimizer(partidos_df)
                    portafolio_optimizado = hybrid_opt.generate_portfolio()
                    
                    if not portafolio_optimizado:
                        self.logger.warning("El optimizador Híbrido falló, usando método heredado como fallback")
                        method = "legacy"
                        
                except ImportError as e:
                    self.logger.error(f"Error importando HybridOptimizer: {e}")
                    self.logger.info("Usando método heredado como fallback")
                    method = "legacy"
                except Exception as e:
                    self.logger.error(f"Error en optimizador híbrido: {e}")
                    self.logger.info("Usando método heredado como fallback") 
                    method = "legacy"
            
            # Método 'legacy' o fallback
            if method == "legacy" or not portafolio_optimizado:
                self.logger.info("PASO 4: Generando 4 quinielas Core (Método Heredado)...")
                
                try:
                    core_generator = CoreGenerator()
                    quinielas_core = core_generator.generar_quinielas_core(partidos_procesados)
                    
                    self.logger.info("PASO 5: Generando 26 satélites en pares (Método Heredado)...")
                    satellite_generator = SatelliteGenerator()
                    quinielas_satelites = satellite_generator.generar_pares_satelites(
                        partidos_procesados, 26
                    )
                    
                    self.logger.info("PASO 6: Ejecutando optimización GRASP-Annealing (Heredado)...")
                    portafolio_inicial = quinielas_core + quinielas_satelites
                    portafolio_optimizado = self.legacy_optimizer.optimizar_portafolio_grasp_annealing(
                        portafolio_inicial, partidos_procesados
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error en método heredado: {e}")
                    raise RuntimeError(f"Ambos métodos de optimización fallaron: {e}")

            # Verificar que tenemos un portafolio válido
            if not portafolio_optimizado:
                raise RuntimeError("No se pudo generar un portafolio válido con ningún método")
            
            # PASO 7: CORRECCIÓN AGRESIVA CON AI (resto del código igual...)
            validacion_inicial = self.portfolio_validator.validar_portafolio_completo(portafolio_optimizado)
            
            ai_fue_utilizada = False
            if (not validacion_inicial["es_valido"] or forzar_ai) and self.ai_assistant.enabled:
                self.logger.info("🤖 ACTIVANDO CORRECCIÓN AGRESIVA CON AI...")
                ai_fue_utilizada = True
                
                max_intentos_ai = 3
                intento_actual = 0
                
                while intento_actual < max_intentos_ai:
                    intento_actual += 1
                    self.logger.info(f"🔄 Intento AI #{intento_actual}")
                    
                    portafolio_corregido_paso1 = self._corregir_quinielas_con_ai(
                        portafolio_optimizado, partidos_procesados
                    )
                    
                    if self._necesita_optimizacion_global(portafolio_corregido_paso1):
                        self.logger.info("🌐 Aplicando optimización global con AI...")
                        portafolio_corregido_paso2 = self.ai_assistant.optimizar_distribucion_global(
                            portafolio_corregido_paso1, partidos_procesados
                        )
                    else:
                        portafolio_corregido_paso2 = portafolio_corregido_paso1
                    
                    validacion_ai = self.portfolio_validator.validar_portafolio_completo(portafolio_corregido_paso2)
                    
                    if validacion_ai["es_valido"]:
                        self.logger.info(f"✅ AI logró corregir el portafolio en intento #{intento_actual}")
                        portafolio_optimizado = portafolio_corregido_paso2
                        break
                    else:
                        self.logger.warning(f"⚠️ Intento #{intento_actual} no resolvió todos los problemas")
                        portafolio_optimizado = portafolio_corregido_paso2
            
            resultado_validacion = self.portfolio_validator.validar_portafolio_completo(portafolio_optimizado)
            
            # PASO 8: Exportar resultados
            self.logger.info("PASO 8: Exportando resultados...")
            archivos_exportados = self.exporter.exportar_portafolio_completo(
                portafolio_optimizado,
                partidos_procesados,
                resultado_validacion["metricas"],
                concurso_id
            )
            
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
                "ai_utilizada": ai_fue_utilizada,
                "metodo_usado": method  # NUEVO: Indicar qué método se usó finalmente
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
            
            if problemas and (quiniela["tipo"] != "Core" or len(problemas) > 1):
                self.logger.debug(f"Corrigiendo {quiniela['id']}: {problemas}")
                
                quiniela_corregida = self.ai_assistant.corregir_quiniela_invalida(
                    quiniela, partidos, problemas
                )
                
                if quiniela_corregida:
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
        if not (4 <= quiniela.get("empates", 0) <= 6):
            problemas.append(f"empates={quiniela.get('empates', 0)} (debe ser 4-6)")
        
        # Concentración general
        if "distribucion" in quiniela:
            max_conc = max(quiniela["distribucion"].values()) / 14
            if max_conc > 0.70:
                problemas.append(f"concentración general {max_conc:.1%} > 70%")
        
        # Concentración inicial
        if "resultados" in quiniela:
            primeros_3 = quiniela["resultados"][:3]
            max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3
            if max_conc_inicial > 0.60:
                problemas.append(f"concentración inicial {max_conc_inicial:.1%} > 60%")
        
        return problemas
    
    def _necesita_optimizacion_global(self, portafolio: List[Dict[str, Any]]) -> bool:
        """
        Determina si el portafolio necesita optimización global
        """
        if not all("distribucion" in q for q in portafolio): return True

        total_L = sum(q["distribucion"]["L"] for q in portafolio)
        total_E = sum(q["distribucion"]["E"] for q in portafolio)
        total_V = sum(q["distribucion"]["V"] for q in portafolio)
        total = total_L + total_E + total_V
        
        if total == 0: return True

        porc_L = total_L / total
        porc_E = total_E / total
        porc_V = total_V / total
        
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
    # Nuevo argumento para seleccionar método
    parser.add_argument("--method", "-m", default="hybrid", choices=['hybrid', 'legacy'], help="Método de optimización a usar")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    optimizer = ProgolOptimizer()
    resultado = optimizer.procesar_concurso(args.archivo, args.concurso, args.forzar_ai, args.method)
    
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