# progol_optimizer/export/exporter.py
"""
Exportador de Portafolio - Genera archivos CSV, JSON, PDF con los resultados
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
        
        Args:
            portafolio: 30 quinielas optimizadas
            partidos: Datos de partidos con clasificación
            metricas: Métricas de validación
            concurso_id: ID del concurso
            
        Returns:
            Dict[str, str]: Rutas de archivos generados
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
            self.logger.error(f"Error en exportación: {e}")
            raise
    
    def _exportar_csv_quinielas(self, portafolio: List[Dict[str, Any]], timestamp: str, concurso_str: str) -> str:
        """
        Exporta las 30 quinielas en formato CSV para impresión
        """
        archivo = self.directorio_salida / f"quinielas_progol{concurso_str}_{timestamp}.csv"
        
        with open(archivo, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Encabezado
            header = ['ID', 'Tipo', 'Par_ID'] + [f'P{i+1}' for i in range(14)] + ['Empates', 'L', 'E', 'V']
            writer.writerow(header)
            
            # Datos de cada quiniela
            for quiniela in portafolio:
                row = [
                    quiniela['id'],
                    quiniela['tipo'],
                    quiniela.get('par_id', ''),
                ] + quiniela['resultados'] + [
                    quiniela['empates'],
                    quiniela['distribución']['L'],
                    quiniela['distribución']['E'],
                    quiniela['distribución']['V']
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
        
        # Preparar datos serializables
        datos_json = {
            "metadata": {
                "timestamp": timestamp,
                "concurso_id": concurso_str.replace("_", ""),
                "version": "1.0.0",
                "metodologia": "Definitiva Progol - Implementación Exacta"
            },
            "portafolio": portafolio,
            "partidos": partidos,
            "metricas": metricas,
            "resumen": {
                "total_quinielas": len(portafolio),
                "cores": len([q for q in portafolio if q["tipo"] == "Core"]),
                "satelites": len([q for q in portafolio if q["tipo"] == "Satelite"]),
                "total_partidos": len(partidos)
            }
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
            f.write("Metodología Definitiva - Implementación Exacta\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Concurso ID: {concurso_str.replace('_', '') or 'No especificado'}\n\n")
            
            # Resumen de partidos
            f.write("RESUMEN DE PARTIDOS:\n")
            f.write("-" * 40 + "\n")
            
            clasificaciones = {}
            for partido in partidos:
                clase = partido.get("clasificacion", "Sin clasificar")
                clasificaciones[clase] = clasificaciones.get(clase, 0) + 1
            
            for clase, count in clasificaciones.items():
                f.write(f"  {clase}: {count} partidos\n")
            
            f.write(f"\nTotal partidos: {len(partidos)}\n\n")
            
            # Resumen del portafolio
            f.write("RESUMEN DEL PORTAFOLIO:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total quinielas: {len(portafolio)}\n")
            f.write(f"Quinielas Core: {len([q for q in portafolio if q['tipo'] == 'Core'])}\n")
            f.write(f"Quinielas Satélite: {len([q for q in portafolio if q['tipo'] == 'Satelite'])}\n\n")
            
            # Distribución global
            if "distribucion_global" in metricas:
                dist = metricas["distribucion_global"]["porcentajes"]
                f.write("DISTRIBUCIÓN GLOBAL:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Locales (L): {dist['L']:.1%}\n")
                f.write(f"  Empates (E): {dist['E']:.1%}\n")
                f.write(f"  Visitantes (V): {dist['V']:.1%}\n\n")
            
            # Lista de partidos
            f.write("DETALLE DE PARTIDOS:\n")
            f.write("-" * 40 + "\n")
            for i, partido in enumerate(partidos, 1):
                f.write(f"{i:2d}. {partido['home']} vs {partido['away']}\n")
                f.write(f"     Liga: {partido['liga']}\n")
                f.write(f"     Probabilidades: L={partido['prob_local']:.3f}, "
                       f"E={partido['prob_empate']:.3f}, V={partido['prob_visitante']:.3f}\n")
                f.write(f"     Clasificación: {partido.get('clasificacion', 'N/A')}\n\n")
            
            # Lista de quinielas
            f.write("QUINIELAS GENERADAS:\n")
            f.write("-" * 40 + "\n")
            
            for quiniela in portafolio:
                f.write(f"{quiniela['id']} ({quiniela['tipo']}): ")
                f.write("".join(quiniela['resultados']))
                f.write(f" [Empates: {quiniela['empates']}]\n")
        
        self.logger.debug(f"Reporte de texto exportado: {archivo}")
        return str(archivo)
    
    def _exportar_csv_partidos(self, partidos: List[Dict[str, Any]], timestamp: str, concurso_str: str) -> str:
        """
        Exporta datos de partidos en CSV
        """
        archivo = self.directorio_salida / f"partidos{concurso_str}_{timestamp}.csv"
        
        with open(archivo, 'w', newline='', encoding='utf-8') as f:
            if not partidos:
                return str(archivo)
            
            # Usar todas las claves del primer partido como header
            fieldnames = partidos[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for partido in partidos:
                # Convertir valores no serializables
                row = {}
                for key, value in partido.items():
                    if isinstance(value, (int, float, str, bool)):
                        row[key] = value
                    else:
                        row[key] = str(value)
                writer.writerow(row)
        
        self.logger.debug(f"CSV partidos exportado: {archivo}")
        return str(archivo)
    
    def _exportar_configuracion(self, timestamp: str, concurso_str: str) -> str:
        """
        Exporta la configuración utilizada
        """
        archivo = self.directorio_salida / f"configuracion{concurso_str}_{timestamp}.json"
        
        try:
            from progol_optimizer.config.constants import PROGOL_CONFIG
            
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(PROGOL_CONFIG, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Configuración exportada: {archivo}")
            
        except Exception as e:
            self.logger.warning(f"No se pudo exportar configuración: {e}")
            return ""
        
        return str(archivo)
    
    def exportar_quinielas_simples(self, portafolio: List[Dict[str, Any]], archivo_nombre: str = None) -> str:
        """
        Exporta solo las quinielas en formato simple para impresión rápida
        """
        if archivo_nombre is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archivo_nombre = f"quinielas_simples_{timestamp}.txt"
        
        archivo = self.directorio_salida / archivo_nombre
        
        with open(archivo, 'w', encoding='utf-8') as f:
            f.write("QUINIELAS PROGOL - FORMATO SIMPLE\n")
            f.write("=" * 50 + "\n\n")
            
            for quiniela in portafolio:
                f.write(f"{quiniela['id']:8s}: ")
                f.write("".join(quiniela['resultados']))
                f.write(f" (E:{quiniela['empates']})\n")
        
        self.logger.info(f"Quinielas simples exportadas: {archivo}")
        return str(archivo)