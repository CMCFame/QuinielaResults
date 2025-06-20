# progol_optimizer/logging_setup.py
"""
Sistema de logging estructurado con rotación y métricas JSON
Implementa instrumentación completa para debug y trazabilidad
"""

import logging
import json
import time
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class StructuredJSONFormatter(logging.Formatter):
    """
    Formatter que convierte logs a JSON estructurado para análisis
    """
    
    def format(self, record):
        """Convierte LogRecord a JSON estructurado"""
        
        # Timestamp ISO con milisegundos
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Log base estructurado
        log_entry = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Agregar contexto específico de Progol si existe
        if hasattr(record, 'progol_context'):
            log_entry["progol"] = record.progol_context
            
        # Agregar métricas si existen
        if hasattr(record, 'metrics'):
            log_entry["metrics"] = record.metrics
            
        # Agregar trazabilidad
        if hasattr(record, 'session_id'):
            log_entry["session_id"] = record.session_id
            
        return json.dumps(log_entry, ensure_ascii=False)


class ProgolInstrumentor:
    """
    Instrumentador específico para Progol Optimizer
    Captura métricas, tiempos y estados críticos
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.logger = logging.getLogger(f"progol.instrumentor.{self.session_id}")
        self.timers = {}
        self.counters = {}
        self.state_history = []
        
        # Inyectar session_id en todos los logs
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.session_id = self.session_id
            return record
        logging.setLogRecordFactory(record_factory)
        
    def start_timer(self, operation: str) -> str:
        """Inicia timer para una operación"""
        timer_id = f"{operation}_{int(time.time()*1000)}"
        self.timers[timer_id] = {
            "operation": operation,
            "start_time": time.time(),
            "status": "running"
        }
        
        self.logger.info(f"⏱️ Iniciando: {operation}", 
                        extra={"progol_context": {
                            "operation": "timer_start",
                            "timer_id": timer_id,
                            "operation_name": operation
                        }})
        
        return timer_id
    
    def end_timer(self, timer_id: str, success: bool = True, 
                  metrics: Optional[Dict[str, Any]] = None) -> float:
        """Termina timer y registra duración"""
        if timer_id not in self.timers:
            self.logger.warning(f"Timer {timer_id} no encontrado")
            return 0.0
            
        timer_info = self.timers[timer_id]
        duration = time.time() - timer_info["start_time"]
        
        timer_info.update({
            "end_time": time.time(),
            "duration_seconds": duration,
            "status": "success" if success else "failed"
        })
        
        log_context = {
            "operation": "timer_end",
            "timer_id": timer_id,
            "operation_name": timer_info["operation"],
            "duration_seconds": duration,
            "success": success
        }
        
        if metrics:
            log_context["operation_metrics"] = metrics
            
        self.logger.info(f"⏱️ Completado: {timer_info['operation']} ({duration:.2f}s)", 
                        extra={"progol_context": log_context})
        
        return duration
    
    def log_state_change(self, component: str, old_state: Any, new_state: Any, 
                        context: Optional[Dict[str, Any]] = None):
        """Registra cambio de estado en componente"""
        state_change = {
            "component": component,
            "old_state": self._serialize_state(old_state),
            "new_state": self._serialize_state(new_state),
            "timestamp": time.time(),
            "context": context or {}
        }
        
        self.state_history.append(state_change)
        
        self.logger.debug(f"🔄 Estado cambió en {component}", 
                         extra={"progol_context": {
                             "operation": "state_change",
                             "component": component,
                             "change_id": len(self.state_history),
                             **state_change
                         }})
    
    def log_optimization_iteration(self, iteration: int, score: float, 
                                  temperature: float = None, 
                                  accepted: bool = None,
                                  portfolio_hash: str = None):
        """Registra iteración específica de optimización"""
        iteration_data = {
            "iteration": iteration,
            "score": score,
            "temperature": temperature,
            "accepted": accepted,
            "portfolio_hash": portfolio_hash
        }
        
        self.logger.debug(f"🎯 Iteración {iteration}: score={score:.6f}", 
                         extra={"progol_context": {
                             "operation": "optimization_iteration",
                             **iteration_data
                         }})
    
    def log_validation_result(self, component: str, rules_passed: Dict[str, bool], 
                             metrics: Dict[str, Any] = None):
        """Registra resultado de validación detallado"""
        total_rules = len(rules_passed)
        passed_rules = sum(rules_passed.values())
        
        validation_data = {
            "component": component,
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "success_rate": passed_rules / total_rules if total_rules > 0 else 0,
            "rules_detail": rules_passed,
            "metrics": metrics or {}
        }
        
        level = logging.INFO if passed_rules == total_rules else logging.WARNING
        self.logger.log(level, f"✅ Validación {component}: {passed_rules}/{total_rules} reglas", 
                       extra={"progol_context": {
                           "operation": "validation",
                           **validation_data
                       }})
    
    def log_ai_interaction(self, operation: str, prompt_hash: str, 
                          model: str, temperature: float, 
                          success: bool, response_hash: str = None, 
                          parsing_success: bool = None):
        """Registra interacción con IA para auditoría"""
        ai_data = {
            "operation": operation,
            "prompt_hash": prompt_hash,
            "model": model,
            "temperature": temperature,
            "success": success,
            "response_hash": response_hash,
            "parsing_success": parsing_success
        }
        
        self.logger.info(f"🤖 IA {operation}: {'✅' if success else '❌'}", 
                        extra={"progol_context": {
                            "operation": "ai_interaction",
                            **ai_data
                        }})
    
    def increment_counter(self, counter_name: str, increment: int = 1):
        """Incrementa contador específico"""
        self.counters[counter_name] = self.counters.get(counter_name, 0) + increment
        
        self.logger.debug(f"📊 {counter_name}: {self.counters[counter_name]}", 
                         extra={"progol_context": {
                             "operation": "counter_increment",
                             "counter_name": counter_name,
                             "new_value": self.counters[counter_name],
                             "increment": increment
                         }})
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Obtiene resumen completo de la sesión"""
        return {
            "session_id": self.session_id,
            "timers": self.timers,
            "counters": self.counters,
            "state_changes": len(self.state_history),
            "summary_timestamp": datetime.now().isoformat()
        }
    
    def _serialize_state(self, state: Any) -> str:
        """Serializa estado para logging seguro"""
        try:
            if isinstance(state, (dict, list)):
                return json.dumps(state, default=str, ensure_ascii=False)
            elif hasattr(state, '__dict__'):
                return json.dumps(state.__dict__, default=str, ensure_ascii=False)
            else:
                return str(state)
        except Exception:
            return f"<no serializable: {type(state).__name__}>"


def setup_progol_logging(log_level: str = "INFO", 
                        enable_file_logging: bool = True,
                        enable_json_format: bool = True,
                        debug_ai: bool = False) -> ProgolInstrumentor:
    """
    Configura sistema de logging completo para Progol Optimizer
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        enable_file_logging: Si activar logs a archivo
        enable_json_format: Si usar formato JSON estructurado
        debug_ai: Si activar debug específico de IA
        
    Returns:
        ProgolInstrumentor: Instrumentador configurado para la sesión
    """
    
    # Convertir nivel de string a constante
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Limpiar handlers existentes
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Crear directorio de logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Formatter base
    if enable_json_format:
        formatter = StructuredJSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Handler consola (siempre activo)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Handler archivo rotativo (opcional)
    file_handler = None
    if enable_file_logging:
        log_file = logs_dir / f"progol_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG if debug_ai else numeric_level)
        file_handler.setFormatter(formatter)
    
    # Configurar logger raíz
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Logger específico para IA con nivel DEBUG si está habilitado
    if debug_ai:
        ai_logger = logging.getLogger("progol.ai")
        ai_logger.setLevel(logging.DEBUG)
        
        ai_file = logs_dir / f"progol_ai_{datetime.now().strftime('%Y%m%d')}.log"
        ai_handler = RotatingFileHandler(
            ai_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        ai_handler.setFormatter(formatter)
        ai_logger.addHandler(ai_handler)
    
    # Crear instrumentador
    instrumentor = ProgolInstrumentor()
    
    # Log inicial de configuración
    logger = logging.getLogger("progol.setup")
    logger.info("🚀 Sistema de logging configurado", 
                extra={"progol_context": {
                    "operation": "logging_setup",
                    "session_id": instrumentor.session_id,
                    "log_level": log_level,
                    "file_logging": enable_file_logging,
                    "json_format": enable_json_format,
                    "debug_ai": debug_ai,
                    "logs_directory": str(logs_dir.absolute())
                }})
    
    return instrumentor


def create_hash(data: Any) -> str:
    """Crea hash SHA-256 de datos para trazabilidad"""
    if isinstance(data, str):
        content = data
    else:
        content = json.dumps(data, sort_keys=True, default=str)
    
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


# Singleton global para fácil acceso
_global_instrumentor = None

def get_instrumentor() -> ProgolInstrumentor:
    """Obtiene el instrumentador global (lazy initialization)"""
    global _global_instrumentor
    if _global_instrumentor is None:
        _global_instrumentor = setup_progol_logging()
    return _global_instrumentor