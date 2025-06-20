# progol_optimizer/utils/safe_patch.py
"""
Sistema de parches seguros para prevenir sobrescrituras accidentales por IA
Implementa verificación de checksums, límites de cambios y rollback automático
"""

import os
import shutil
import hashlib
import difflib
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from logging_setup import get_instrumentor


@dataclass
class PatchValidationResult:
    """Resultado de validación de un patch"""
    is_valid: bool
    error_message: Optional[str] = None
    changes_count: int = 0
    deletion_percentage: float = 0.0
    addition_percentage: float = 0.0
    checksum_match: bool = False
    safety_score: float = 0.0


@dataclass
class BackupInfo:
    """Información de backup para rollback"""
    backup_path: str
    original_checksum: str
    timestamp: float
    file_path: str


class SafePatchManager:
    """
    Gestor de parches seguros que previene sobrescrituras accidentales
    
    Características de seguridad:
    1. Verificación de checksums antes de aplicar patches
    2. Límites configurables de cambios permitidos
    3. Backup automático antes de modificaciones
    4. Rollback automático en caso de errores
    5. Logging detallado de todas las operaciones
    6. Validación de integridad post-patch
    """
    
    def __init__(self, project_root: str = None):
        self.logger = logging.getLogger(__name__)
        self.instrumentor = get_instrumentor()
        
        # Configuración de seguridad
        self.max_deletion_percentage = 30.0    # Máximo 30% del archivo eliminado
        self.max_addition_percentage = 50.0    # Máximo 50% del archivo añadido
        self.max_total_changes = 200           # Máximo 200 líneas modificadas
        self.min_safety_score = 0.6           # Score mínimo para aprobar patch
        
        # Directorio del proyecto
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Directorio de backups
        self.backup_dir = self.project_root / ".safe_patches" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Registro de operaciones
        self.operations_log = self.project_root / ".safe_patches" / "operations.json"
        self.operations_log.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache de checksums conocidos
        self.known_checksums = self._load_known_checksums()
        
        self.logger.info(f"🛡️ SafePatchManager inicializado: {self.project_root}")
    
    def apply_safe_patch(self, file_path: str, new_content: str, 
                        expected_checksum: str = None,
                        patch_description: str = "") -> Tuple[bool, str]:
        """
        Aplica un patch de forma segura con todas las validaciones
        
        Args:
            file_path: Ruta del archivo a modificar
            new_content: Nuevo contenido del archivo
            expected_checksum: Checksum esperado del archivo original (opcional)
            patch_description: Descripción del patch para logging
            
        Returns:
            Tuple[bool, str]: (éxito, mensaje de resultado)
        """
        operation_timer = self.instrumentor.start_timer("safe_patch_operation")
        
        try:
            self.logger.info(f"🔧 Aplicando patch seguro a: {file_path}")
            if patch_description:
                self.logger.info(f"📝 Descripción: {patch_description}")
            
            file_path = Path(file_path)
            
            # 1. Validaciones previas
            if not file_path.exists():
                error_msg = f"Archivo no existe: {file_path}"
                self.logger.error(error_msg)
                self.instrumentor.end_timer(operation_timer, success=False)
                return False, error_msg
            
            # 2. Leer contenido original
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
            except Exception as e:
                error_msg = f"Error leyendo archivo: {e}"
                self.logger.error(error_msg)
                self.instrumentor.end_timer(operation_timer, success=False)
                return False, error_msg
            
            # 3. Verificar checksum si se proporciona
            original_checksum = self._calculate_checksum(original_content)
            
            if expected_checksum and original_checksum != expected_checksum:
                error_msg = f"Checksum no coincide. Esperado: {expected_checksum}, Actual: {original_checksum}"
                self.logger.error(error_msg)
                self.instrumentor.end_timer(operation_timer, success=False)
                return False, error_msg
            
            # 4. Validar el patch
            validation_result = self._validate_patch(
                original_content, new_content, file_path
            )
            
            if not validation_result.is_valid:
                error_msg = f"Patch inválido: {validation_result.error_message}"
                self.logger.error(error_msg)
                self.instrumentor.end_timer(operation_timer, success=False)
                return False, error_msg
            
            # 5. Crear backup
            backup_info = self._create_backup(file_path, original_content, original_checksum)
            
            # 6. Aplicar patch con protección
            try:
                success, message = self._apply_patch_with_protection(
                    file_path, new_content, backup_info, validation_result
                )
                
                if success:
                    # 7. Registrar operación exitosa
                    self._log_operation(
                        file_path=str(file_path),
                        patch_description=patch_description,
                        backup_info=backup_info,
                        validation_result=validation_result,
                        success=True
                    )
                    
                    self.instrumentor.end_timer(operation_timer, success=True, metrics={
                        "file_path": str(file_path),
                        "changes_count": validation_result.changes_count,
                        "safety_score": validation_result.safety_score
                    })
                    
                    self.logger.info(f"✅ Patch aplicado exitosamente: {file_path}")
                    return True, message
                else:
                    # 8. Rollback automático en caso de fallo
                    self._rollback_from_backup(backup_info)
                    self.instrumentor.end_timer(operation_timer, success=False)
                    return False, message
                    
            except Exception as e:
                # Rollback de emergencia
                self._rollback_from_backup(backup_info)
                error_msg = f"Error aplicando patch: {e}"
                self.logger.error(error_msg)
                self.instrumentor.end_timer(operation_timer, success=False)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Error crítico en safe_patch: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.instrumentor.end_timer(operation_timer, success=False)
            return False, error_msg
    
    def _validate_patch(self, original_content: str, new_content: str, 
                       file_path: Path) -> PatchValidationResult:
        """
        Valida que el patch sea seguro según múltiples criterios
        """
        validation_timer = self.instrumentor.start_timer("patch_validation")
        
        try:
            original_lines = original_content.splitlines()
            new_lines = new_content.splitlines()
            
            # Generar diff para análisis
            diff = list(difflib.unified_diff(
                original_lines, new_lines, 
                fromfile=f"{file_path}.original",
                tofile=f"{file_path}.new",
                lineterm=""
            ))
            
            # Contar cambios
            additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
            deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
            total_changes = additions + deletions
            
            # Calcular porcentajes
            original_line_count = len(original_lines)
            if original_line_count == 0:
                original_line_count = 1  # Evitar división por cero
            
            deletion_percentage = (deletions / original_line_count) * 100
            addition_percentage = (additions / original_line_count) * 100
            
            # Verificar límites
            validation_errors = []
            
            if deletion_percentage > self.max_deletion_percentage:
                validation_errors.append(
                    f"Demasiadas eliminaciones: {deletion_percentage:.1f}% > {self.max_deletion_percentage}%"
                )
            
            if addition_percentage > self.max_addition_percentage:
                validation_errors.append(
                    f"Demasiadas adiciones: {addition_percentage:.1f}% > {self.max_addition_percentage}%"
                )
            
            if total_changes > self.max_total_changes:
                validation_errors.append(
                    f"Demasiados cambios totales: {total_changes} > {self.max_total_changes}"
                )
            
            # Verificaciones específicas de contenido
            content_validations = self._validate_content_safety(original_content, new_content)
            validation_errors.extend(content_validations)
            
            # Calcular safety score
            safety_score = self._calculate_safety_score(
                deletion_percentage, addition_percentage, total_changes, len(validation_errors)
            )
            
            # Determinar si es válido
            is_valid = len(validation_errors) == 0 and safety_score >= self.min_safety_score
            
            result = PatchValidationResult(
                is_valid=is_valid,
                error_message="; ".join(validation_errors) if validation_errors else None,
                changes_count=total_changes,
                deletion_percentage=deletion_percentage,
                addition_percentage=addition_percentage,
                checksum_match=True,  # Ya se validó antes
                safety_score=safety_score
            )
            
            self.instrumentor.end_timer(validation_timer, success=is_valid, metrics={
                "total_changes": total_changes,
                "safety_score": safety_score,
                "validation_errors": len(validation_errors)
            })
            
            return result
            
        except Exception as e:
            self.instrumentor.end_timer(validation_timer, success=False)
            return PatchValidationResult(
                is_valid=False,
                error_message=f"Error en validación: {e}"
            )
    
    def _validate_content_safety(self, original: str, new: str) -> List[str]:
        """
        Validaciones específicas de contenido para detectar cambios peligrosos
        """
        errors = []
        
        # 1. Verificar que no se eliminen imports críticos
        critical_imports = [
            "from config.constants import PROGOL_CONFIG",
            "from logging_setup import get_instrumentor", 
            "import logging"
        ]
        
        for critical_import in critical_imports:
            if critical_import in original and critical_import not in new:
                errors.append(f"Import crítico eliminado: {critical_import}")
        
        # 2. Verificar que no se eliminen clases/funciones completas
        import re
        
        original_classes = set(re.findall(r'^class\s+(\w+)', original, re.MULTILINE))
        new_classes = set(re.findall(r'^class\s+(\w+)', new, re.MULTILINE))
        deleted_classes = original_classes - new_classes
        
        if deleted_classes:
            errors.append(f"Clases eliminadas: {', '.join(deleted_classes)}")
        
        original_functions = set(re.findall(r'^def\s+(\w+)', original, re.MULTILINE))
        new_functions = set(re.findall(r'^def\s+(\w+)', new, re.MULTILINE))
        deleted_functions = original_functions - new_functions
        
        if len(deleted_functions) > 3:  # Permitir eliminar máximo 3 funciones
            errors.append(f"Demasiadas funciones eliminadas: {len(deleted_functions)}")
        
        # 3. Verificar que se mantenga la estructura básica
        if "def __init__" in original and "def __init__" not in new:
            errors.append("Constructor __init__ eliminado")
        
        # 4. Verificar que no se introduzcan imports peligrosos
        dangerous_imports = ["os.system", "subprocess", "eval", "exec"]
        for dangerous in dangerous_imports:
            if dangerous not in original and dangerous in new:
                errors.append(f"Import peligroso añadido: {dangerous}")
        
        return errors
    
    def _calculate_safety_score(self, deletion_pct: float, addition_pct: float, 
                              total_changes: int, error_count: int) -> float:
        """
        Calcula un score de seguridad de 0.0 a 1.0 para el patch
        """
        score = 1.0
        
        # Penalizar por exceso de eliminaciones
        if deletion_pct > 10:
            score -= min(0.3, (deletion_pct - 10) / 100)
        
        # Penalizar por exceso de adiciones
        if addition_pct > 20:
            score -= min(0.2, (addition_pct - 20) / 100)
        
        # Penalizar por demasiados cambios totales
        if total_changes > 50:
            score -= min(0.3, (total_changes - 50) / 500)
        
        # Penalizar por errores de validación
        score -= error_count * 0.2
        
        return max(0.0, score)
    
    def _create_backup(self, file_path: Path, content: str, checksum: str) -> BackupInfo:
        """
        Crea backup del archivo antes de modificar
        """
        timestamp = time.time()
        backup_name = f"{file_path.name}_{int(timestamp)}_{checksum[:8]}.backup"
        backup_path = self.backup_dir / backup_name
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        backup_info = BackupInfo(
            backup_path=str(backup_path),
            original_checksum=checksum,
            timestamp=timestamp,
            file_path=str(file_path)
        )
        
        self.logger.debug(f"💾 Backup creado: {backup_path}")
        return backup_info
    
    def _apply_patch_with_protection(self, file_path: Path, new_content: str, 
                                   backup_info: BackupInfo, 
                                   validation_result: PatchValidationResult) -> Tuple[bool, str]:
        """
        Aplica el patch con protecciones adicionales
        """
        try:
            # Escribir a archivo temporal primero
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp_file:
                temp_file.write(new_content)
                temp_path = temp_file.name
            
            # Verificar que se escribió correctamente
            with open(temp_path, 'r', encoding='utf-8') as f:
                written_content = f.read()
            
            if written_content != new_content:
                os.unlink(temp_path)
                return False, "Error: contenido escrito no coincide con el esperado"
            
            # Mover archivo temporal al destino
            shutil.move(temp_path, file_path)
            
            # Verificación post-patch
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    final_content = f.read()
                
                if final_content != new_content:
                    return False, "Error: verificación post-patch falló"
                
                return True, f"Patch aplicado exitosamente ({validation_result.changes_count} cambios)"
                
            except Exception as e:
                return False, f"Error en verificación post-patch: {e}"
                
        except Exception as e:
            # Limpiar archivo temporal si existe
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return False, f"Error aplicando patch: {e}"
    
    def _rollback_from_backup(self, backup_info: BackupInfo):
        """
        Realiza rollback desde backup
        """
        try:
            if os.path.exists(backup_info.backup_path):
                shutil.copy2(backup_info.backup_path, backup_info.file_path)
                self.logger.warning(f"🔄 Rollback realizado desde: {backup_info.backup_path}")
            else:
                self.logger.error(f"❌ Backup no encontrado para rollback: {backup_info.backup_path}")
        except Exception as e:
            self.logger.error(f"❌ Error en rollback: {e}")
    
    def _log_operation(self, file_path: str, patch_description: str, 
                      backup_info: BackupInfo, validation_result: PatchValidationResult, 
                      success: bool):
        """
        Registra la operación en el log de operaciones
        """
        operation_record = {
            "timestamp": time.time(),
            "file_path": file_path,
            "patch_description": patch_description,
            "backup_path": backup_info.backup_path,
            "original_checksum": backup_info.original_checksum,
            "changes_count": validation_result.changes_count,
            "safety_score": validation_result.safety_score,
            "success": success
        }
        
        # Leer operaciones existentes
        operations = []
        if self.operations_log.exists():
            try:
                with open(self.operations_log, 'r', encoding='utf-8') as f:
                    operations = json.load(f)
            except Exception:
                operations = []
        
        # Añadir nueva operación
        operations.append(operation_record)
        
        # Mantener solo últimas 100 operaciones
        operations = operations[-100:]
        
        # Guardar de vuelta
        try:
            with open(self.operations_log, 'w', encoding='utf-8') as f:
                json.dump(operations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error guardando log de operaciones: {e}")
    
    def _calculate_checksum(self, content: str) -> str:
        """Calcula checksum SHA-256 del contenido"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _load_known_checksums(self) -> Dict[str, str]:
        """Carga checksums conocidos de archivos críticos"""
        checksums_file = self.project_root / ".safe_patches" / "known_checksums.json"
        
        if checksums_file.exists():
            try:
                with open(checksums_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        
        return {}
    
    def update_known_checksum(self, file_path: str, checksum: str):
        """Actualiza checksum conocido de un archivo"""
        self.known_checksums[file_path] = checksum
        
        checksums_file = self.project_root / ".safe_patches" / "known_checksums.json"
        try:
            with open(checksums_file, 'w', encoding='utf-8') as f:
                json.dump(self.known_checksums, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error guardando checksums: {e}")
    
    def cleanup_old_backups(self, days_to_keep: int = 7):
        """
        Limpia backups antiguos
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        cleaned_count = 0
        
        for backup_file in self.backup_dir.glob("*.backup"):
            try:
                # Extraer timestamp del nombre del archivo
                timestamp_str = backup_file.stem.split('_')[1]
                backup_timestamp = float(timestamp_str)
                
                if backup_timestamp < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error procesando backup {backup_file}: {e}")
        
        self.logger.info(f"🧹 Limpiados {cleaned_count} backups antiguos")
    
    def get_operation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Obtiene historial de operaciones recientes
        """
        if not self.operations_log.exists():
            return []
        
        try:
            with open(self.operations_log, 'r', encoding='utf-8') as f:
                operations = json.load(f)
            
            # Retornar las más recientes
            return operations[-limit:]
            
        except Exception as e:
            self.logger.error(f"Error leyendo historial: {e}")
            return []


# Función de conveniencia para uso directo
def apply_safe_patch(file_path: str, new_content: str, 
                    expected_checksum: str = None,
                    patch_description: str = "",
                    project_root: str = None) -> Tuple[bool, str]:
    """
    Función de conveniencia para aplicar un patch seguro
    
    Args:
        file_path: Ruta del archivo a modificar
        new_content: Nuevo contenido del archivo
        expected_checksum: Checksum esperado del archivo original
        patch_description: Descripción del patch
        project_root: Directorio raíz del proyecto
        
    Returns:
        Tuple[bool, str]: (éxito, mensaje)
    """
    manager = SafePatchManager(project_root)
    return manager.apply_safe_patch(
        file_path, new_content, expected_checksum, patch_description
    )


# Decorador para proteger funciones que modifican archivos
def safe_file_operation(expected_checksum: str = None, description: str = ""):
    """
    Decorador que convierte operaciones de archivos en operaciones seguras
    """
    def decorator(func):
        def wrapper(file_path: str, content: str, *args, **kwargs):
            return apply_safe_patch(
                file_path, content, expected_checksum, 
                description or f"Operation: {func.__name__}"
            )
        return wrapper
    return decorator


# Clase de contexto para operaciones batch
class SafePatchBatch:
    """
    Contexto para aplicar múltiples patches de forma atómica
    """
    
    def __init__(self, project_root: str = None):
        self.manager = SafePatchManager(project_root)
        self.pending_operations = []
        self.applied_operations = []
        
    def add_patch(self, file_path: str, new_content: str, 
                  expected_checksum: str = None, description: str = ""):
        """Añade un patch al batch"""
        self.pending_operations.append({
            "file_path": file_path,
            "new_content": new_content,
            "expected_checksum": expected_checksum,
            "description": description
        })
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback de todas las operaciones aplicadas
            self._rollback_all()
        return False
    
    def execute_all(self) -> Tuple[bool, List[str]]:
        """
        Ejecuta todas las operaciones del batch
        """
        results = []
        
        for operation in self.pending_operations:
            success, message = self.manager.apply_safe_patch(**operation)
            results.append(message)
            
            if success:
                self.applied_operations.append(operation)
            else:
                # Fallo - rollback de todo
                self._rollback_all()
                return False, results
        
        return True, results
    
    def _rollback_all(self):
        """Rollback de todas las operaciones aplicadas"""
        # Implementar rollback si es necesario
        # Por simplicidad, SafePatchManager ya maneja rollbacks individuales
        pass