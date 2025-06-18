# progol_optimizer/export/exporter.py
"""
Exportador de Portafolio - Genera archivos CSV, JSON, TXT con los resultados
CORRECCIÓN: Maneja correctamente el formato de 'resultados' como string.
"""

import logging
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class PortfolioExporter:
    """
    Exporta el portafolio optimizado en múltiples formatos
    """
    
    def __init__(self, directorio_salida: str = "outputs"):
        self.logger = logging.getLogger(__name__)
        self.directorio_salida = Path(directorio_salida)
        self.directorio_salida.mkdir(exist_ok=True)
        
        self.logger.debug(f"Exportador configurado: directorio={self.directorio_salida}")
    
    def exportar_portafolio_completo(self, portafolio: List[Dict[str, Any]], 
                                   partidos: List[Dict[str, Any]],
                                   metricas: Dict[str, Any],
                                   concurso_id: str = None) -> Dict[str, str]:
        """
        Exporta el portafolio completo en todos los formatos disponibles
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        concurso_str = f"_{concurso_id}" if concurso_id else ""
        
        self.logger.info("Iniciando exportación completa...")
        
        archivos_generados = {}
        
        try:
            # 1. CSV con quinielas
            archivo_csv = self._exportar_csv_quinielas(portafolio, timestamp, concurso_str)
            archivos_generados["csv_quinielas"] = archivo_csv
            
            # 2. JSON completo
            archivo_json = self._exportar_json_completo(portafolio, partidos, metricas, timestamp, concurso_str)
            archivos_generados["json_completo"] = archivo_json
            
            # 3. Reporte de texto
            archivo_reporte = self._exportar_reporte_texto(portafolio, partidos, metricas, timestamp, concurso_str)
            archivos_generados["reporte_texto"] = archivo_reporte
            
            # 4. CSV resumen de partidos
            archivo_partidos = self._exportar_csv_partidos(partidos, timestamp, concurso_str)
            archivos_generados["csv_partidos"] = archivo_partidos
            
            # 5. Archivo de configuración usado
            archivo_config = self._exportar_configuracion(timestamp, concurso_str)
            archivos_generados["configuracion"] = archivo_config
            
            self.logger.info(f"✅ Exportación completada: {len(archivos_generados)} archivos generados")
            
            return archivos_generados
            
        except Exception as e:
            self.logger.error(f"Error en exportación: {e}", exc_info=True)
            raise
    
    def _exportar_csv_quinielas(self, portafolio: List[Dict[str, Any]], timestamp: str, concurso_str: str) -> str:
        """
        Exporta las 30 quinielas en formato CSV para impresión
        """
        archivo = self.directorio_salida / f"quinielas_progol{concurso_str}_{timestamp}.csv"
        
        with open(archivo, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            header = ['ID', 'Tipo', 'Par_ID'] + [f'P{i+1}' for i in range(14)] + ['Empates', 'L', 'E', 'V']
            writer.writerow(header)
            
            for quiniela in portafolio:
                # CORRECCIÓN CLAVE: Convertir la cadena de resultados a una lista
                resultados_lista = list(quiniela['resultados'])

                row = [
                    quiniela.get('id', ''),
                    quiniela.get('tipo', ''),
                    quiniela.get('par_id', ''),
                ] + resultados_lista + [
                    quiniela.get('empates', ''),
                    quiniela.get('distribución', {}).get('L', ''),
                    quiniela.get('distribución', {}).get('E', ''),
                    quiniela.get('distribución', {}).get('V', '')
                ]
                writer.writerow(row)
        
        self.logger.debug(f"CSV quinielas exportado: {archivo}")
        return str(archivo)
    
    def _exportar_json_completo(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]], 
                               metricas: Dict[str, Any], timestamp: str, concurso_str: str) -> str:
        """
        Exporta todo en formato JSON estructurado
        """
        archivo = self.directorio_salida / f"progol_completo{concurso_str}_{timestamp}.json"
        
        datos_json = {
            "metadata": {
                "timestamp": timestamp,
                "concurso_id": concurso_str.replace("_", ""),
                "version": "1.0.0",
                "metodologia": "Definitiva Progol - Implementación Robusta"
            },
            "portafolio": portafolio,
            "partidos": partidos,
            "metricas": metricas,
        }
        
        with open(archivo, 'w', encoding='utf-8') as f:
            json.dump(datos_json, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.debug(f"JSON completo exportado: {archivo}")
        return str(archivo)
    
    def _exportar_reporte_texto(self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]], 
                               metricas: Dict[str, Any], timestamp: str, concurso_str: str) -> str:
        """
        Genera reporte legible en texto plano
        """
        archivo = self.directorio_salida / f"reporte_progol{concurso_str}_{timestamp}.txt"
        
        with open(archivo, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PROGOL OPTIMIZER - REPORTE DE OPTIMIZACIÓN\n")
            f.write("Metodología Definitiva - Implementación Robusta\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Concurso ID: {concurso_str.replace('_', '') or 'N/A'}\n\n")
            
            if "distribucion_global" in metricas:
                dist = metricas["distribucion_global"]["porcentajes"]
                f.write("DISTRIBUCIÓN GLOBAL DEL PORTAFOLIO:\n")
                f.write(f"  Locales (L): {dist.get('L', 0):.1%}\n")
                f.write(f"  Empates (E): {dist.get('E', 0):.1%}\n")
                f.write(f"  Visitantes (V): {dist.get('V', 0):.1%}\n\n")

            f.write("QUINIELAS GENERADAS:\n")
            f.write("-" * 40 + "\n")
            
            for q in portafolio:
                f.write(f"{q.get('id', 'N/A'):<15} ({q.get('tipo', 'N/A'):<8}): ")
                f.write("".join(q.get('resultados', '')))
                f.write(f" [Empates: {q.get('empates', 'N/A')}]\n")
        
        self.logger.debug(f"Reporte de texto exportado: {archivo}")
        return str(archivo)
    
    def _exportar_csv_partidos(self, partidos: List[Dict[str, Any]], timestamp: str, concurso_str: str) -> str:
        """
        Exporta datos de partidos en CSV
        """
        archivo = self.directorio_salida / f"partidos{concurso_str}_{timestamp}.csv"
        
        with open(archivo, 'w', newline='', encoding='utf-8') as f:
            if not partidos: return str(archivo)
            
            fieldnames = list(partidos[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for partido in partidos:
                row = {k: str(v) for k, v in partido.items()}
                writer.writerow(row)
        
        self.logger.debug(f"CSV partidos exportado: {archivo}")
        return str(archivo)
    
    def _exportar_configuracion(self, timestamp: str, concurso_str: str) -> str:
        """
        Exporta la configuración utilizada
        """
        archivo = self.directorio_salida / f"configuracion{concurso_str}_{timestamp}.json"
        
        try:
            from config.constants import PROGOL_CONFIG
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(PROGOL_CONFIG, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Configuración exportada: {archivo}")
        except Exception as e:
            self.logger.warning(f"No se pudo exportar configuración: {e}")
            return ""
        
        return str(archivo)