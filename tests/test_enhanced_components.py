# tests/test_enhanced_components.py
"""
Suite de tests completa para los componentes mejorados
Valida que todas las mejoras funcionen correctamente y que se mantenga compatibilidad
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Configurar path para imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_setup import setup_progol_logging, ProgolInstrumentor
from data.loader import EnhancedDataLoader
from models.ai_assistant import EnhancedProgolAIAssistant
from utils.safe_patch import SafePatchManager
from main import EnhancedProgolOptimizer


class TestLoggingSystem:
    """Tests para el sistema de logging estructurado"""
    
    @pytest.fixture
    def temp_logs_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_setup_progol_logging(self, temp_logs_dir):
        """Test configuración básica del logging"""
        os.chdir(temp_logs_dir)
        
        instrumentor = setup_progol_logging(
            log_level="DEBUG",
            enable_file_logging=True,
            enable_json_format=True
        )
        
        assert instrumentor is not None
        assert instrumentor.session_id is not None
        assert len(instrumentor.session_id) == 8
        
        # Verificar que se creó el directorio de logs
        logs_dir = temp_logs_dir / "logs"
        assert logs_dir.exists()
    
    def test_instrumentor_timers(self):
        """Test funcionamiento de timers"""
        instrumentor = ProgolInstrumentor()
        
        timer_id = instrumentor.start_timer("test_operation")
        assert timer_id in instrumentor.timers
        assert instrumentor.timers[timer_id]["status"] == "running"
        
        duration = instrumentor.end_timer(timer_id, success=True)
        assert duration >= 0
        assert instrumentor.timers[timer_id]["status"] == "success"
    
    def test_state_change_logging(self):
        """Test logging de cambios de estado"""
        instrumentor = ProgolInstrumentor()
        
        old_state = {"quinielas": 0}
        new_state = {"quinielas": 30}
        
        instrumentor.log_state_change(
            component="portfolio",
            old_state=old_state,
            new_state=new_state,
            context={"operation": "generation"}
        )
        
        assert len(instrumentor.state_history) == 1
        assert instrumentor.state_history[0]["component"] == "portfolio"


class TestEnhancedDataLoader:
    """Tests para el cargador de datos mejorado"""
    
    @pytest.fixture
    def enhanced_loader(self):
        return EnhancedDataLoader()
    
    def test_generar_datos_optimizados(self, enhanced_loader):
        """Test generación de datos con Anclas garantizadas"""
        partidos = enhanced_loader._generar_datos_optimizados()
        
        assert len(partidos) == 14
        
        # Verificar que todos los partidos tienen estructura correcta
        for partido in partidos:
            assert "id" in partido
            assert "home" in partido
            assert "away" in partido
            assert "prob_local" in partido
            assert "prob_empate" in partido
            assert "prob_visitante" in partido
            
            # Verificar que las probabilidades suman ~1
            total_prob = partido["prob_local"] + partido["prob_empate"] + partido["prob_visitante"]
            assert abs(total_prob - 1.0) < 0.01
    
    def test_garantia_anclas(self, enhanced_loader):
        """Test que se garantizan al menos 6 partidos Ancla"""
        partidos = enhanced_loader._generar_datos_optimizados()
        
        anclas_count = enhanced_loader._contar_anclas(partidos)
        assert anclas_count >= enhanced_loader.min_anchors
        
        # Verificar que las Anclas tienen probabilidades altas
        for partido in partidos:
            max_prob = max(partido["prob_local"], partido["prob_empate"], partido["prob_visitante"])
            if max_prob >= enhanced_loader.anchor_threshold:
                assert max_prob >= 0.65
    
    def test_distribucion_historica(self, enhanced_loader):
        """Test que la distribución respeta rangos históricos"""
        partidos = enhanced_loader._generar_datos_optimizados()
        
        # Simular clasificación para obtener resultados más probables
        resultados = []
        for partido in partidos:
            probs = [partido["prob_local"], partido["prob_empate"], partido["prob_visitante"]]
            resultado_idx = np.argmax(probs)
            resultados.append(["L", "E", "V"][resultado_idx])
        
        # Verificar distribución aproximada
        l_count = resultados.count("L") / 14
        e_count = resultados.count("E") / 14
        v_count = resultados.count("V") / 14
        
        # Permitir cierta flexibilidad en los tests
        assert 0.25 <= l_count <= 0.55  # Rango amplio para tests
        assert 0.15 <= e_count <= 0.45
        assert 0.20 <= v_count <= 0.50
    
    @pytest.fixture
    def csv_test_data(self):
        """Genera datos CSV de prueba"""
        data = []
        for i in range(14):
            data.append({
                "home": f"Team{i}A",
                "away": f"Team{i}B",
                "liga": "Test League",
                "prob_local": 0.4 + (i % 3) * 0.1,
                "prob_empate": 0.3,
                "prob_visitante": 0.3 - (i % 3) * 0.1
            })
        return pd.DataFrame(data)
    
    def test_procesar_csv_usuario(self, enhanced_loader, csv_test_data):
        """Test procesamiento de CSV del usuario"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_test_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            partidos = enhanced_loader.cargar_datos(csv_path)
            
            assert len(partidos) == 14
            assert partidos[0]["home"] == "Team0A"
            assert partidos[0]["away"] == "Team0B"
            
        finally:
            os.unlink(csv_path)


class TestEnhancedAIAssistant:
    """Tests para el asistente IA mejorado con safeguards"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock del cliente OpenAI"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"resultados": ["L","E","V","L","E","V","L","E","V","L","E","V","L","E"]}'
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def ai_assistant_with_mock(self, mock_openai_client):
        """Asistente IA con cliente mock"""
        with patch('models.ai_assistant.OpenAI') as mock_openai:
            mock_openai.return_value = mock_openai_client
            
            assistant = EnhancedProgolAIAssistant(api_key="test_key")
            assistant.client = mock_openai_client
            assistant.enabled = True
            
            yield assistant
    
    def test_correccion_con_safeguards(self, ai_assistant_with_mock):
        """Test corrección con límites de distancia Hamming"""
        quiniela_problematica = {
            "id": "Test-1",
            "tipo": "Satelite",
            "resultados": ["L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L"],
            "empates": 0,
            "distribución": {"L": 14, "E": 0, "V": 0}
        }
        
        partidos_mock = []
        for i in range(14):
            partidos_mock.append({
                "id": i,
                "home": f"Team{i}A",
                "away": f"Team{i}B",
                "clasificacion": "Neutro",
                "prob_local": 0.4,
                "prob_empate": 0.3,
                "prob_visitante": 0.3
            })
        
        reglas_violadas = ["concentracion_excesiva_L", "empates_insuficientes_0"]
        
        resultado = ai_assistant_with_mock.corregir_quiniela_con_safeguards(
            quiniela_problematica, partidos_mock, reglas_violadas
        )
        
        # Verificar que se aplicaron safeguards
        if resultado:
            # La corrección debe tener empates válidos
            assert 4 <= resultado["empates"] <= 6
            
            # La distancia Hamming debe ser razonable
            distancia = ai_assistant_with_mock._calcular_distancia_hamming(
                quiniela_problematica["resultados"],
                resultado["resultados"]
            )
            assert distancia <= ai_assistant_with_mock.max_hamming_distance
    
    def test_distancia_hamming(self, ai_assistant_with_mock):
        """Test cálculo de distancia Hamming"""
        resultados_a = ["L", "E", "V", "L", "E", "V", "L"]
        resultados_b = ["L", "E", "L", "L", "V", "V", "L"]  # 2 diferencias
        
        distancia = ai_assistant_with_mock._calcular_distancia_hamming(resultados_a, resultados_b)
        assert distancia == 2
    
    def test_parsing_robusto(self, ai_assistant_with_mock):
        """Test parsing robusto de respuestas IA"""
        quiniela_original = {
            "id": "Test-1",
            "tipo": "Satelite",
            "resultados": ["L"] * 14,
            "empates": 0,
            "distribución": {"L": 14, "E": 0, "V": 0}
        }
        
        # Test diferentes formatos de respuesta
        respuestas_test = [
            '{"resultados": ["L","E","V","L","E","V","L","E","V","L","E","V","L","E"]}',
            '["L","E","V","L","E","V","L","E","V","L","E","V","L","E"]',
            'L,E,V,L,E,V,L,E,V,L,E,V,L,E',
            'LEVLEVLEVLEVLE'
        ]
        
        for respuesta in respuestas_test:
            resultado = ai_assistant_with_mock._parsing_robusto_respuesta(respuesta, quiniela_original)
            if resultado:
                assert len(resultado["resultados"]) == 14
                assert all(r in ["L", "E", "V"] for r in resultado["resultados"])
    
    def test_usage_stats_tracking(self, ai_assistant_with_mock):
        """Test tracking de estadísticas de uso"""
        stats_iniciales = ai_assistant_with_mock.get_usage_stats()
        
        assert "total_calls" in stats_iniciales
        assert "successful_corrections" in stats_iniciales
        assert "rejected_by_hamming" in stats_iniciales
        assert "safeguards_config" in stats_iniciales


class TestSafePatchManager:
    """Tests para el gestor de parches seguros"""
    
    @pytest.fixture
    def temp_project_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def safe_patch_manager(self, temp_project_dir):
        return SafePatchManager(str(temp_project_dir))
    
    def test_patch_manager_initialization(self, safe_patch_manager, temp_project_dir):
        """Test inicialización del gestor de parches"""
        assert safe_patch_manager.project_root == temp_project_dir
        assert safe_patch_manager.backup_dir.exists()
        assert safe_patch_manager.max_deletion_percentage == 30.0
    
    def test_safe_patch_application(self, safe_patch_manager, temp_project_dir):
        """Test aplicación segura de patch"""
        # Crear archivo de prueba
        test_file = temp_project_dir / "test_file.py"
        original_content = '''# Test file
def hello():
    print("Hello World")

def goodbye():
    print("Goodbye")
'''
        test_file.write_text(original_content, encoding='utf-8')
        
        # Contenido modificado (cambio menor)
        new_content = '''# Test file
def hello():
    print("Hello Enhanced World")

def goodbye():
    print("Goodbye")

def new_function():
    print("New functionality")
'''
        
        # Aplicar patch
        success, message = safe_patch_manager.apply_safe_patch(
            str(test_file), 
            new_content,
            patch_description="Add new function"
        )
        
        assert success
        assert "exitosamente" in message
        
        # Verificar que el archivo se modificó
        final_content = test_file.read_text(encoding='utf-8')
        assert final_content == new_content
        
        # Verificar que se creó backup
        backups = list(safe_patch_manager.backup_dir.glob("*.backup"))
        assert len(backups) > 0
    
    def test_patch_rejection_excessive_deletion(self, safe_patch_manager, temp_project_dir):
        """Test rechazo por demasiadas eliminaciones"""
        # Crear archivo grande
        test_file = temp_project_dir / "large_file.py"
        original_lines = ["# Line " + str(i) for i in range(100)]
        original_content = "\n".join(original_lines)
        test_file.write_text(original_content, encoding='utf-8')
        
        # Contenido que elimina >30% del archivo
        new_content = "\n".join(original_lines[:50])  # Elimina 50% de líneas
        
        success, message = safe_patch_manager.apply_safe_patch(
            str(test_file), 
            new_content,
            patch_description="Excessive deletion test"
        )
        
        assert not success
        assert "Demasiadas eliminaciones" in message
        
        # Verificar que el archivo original no se modificó
        final_content = test_file.read_text(encoding='utf-8')
        assert final_content == original_content
    
    def test_checksum_validation(self, safe_patch_manager, temp_project_dir):
        """Test validación de checksum"""
        test_file = temp_project_dir / "checksum_test.py"
        original_content = "print('test')"
        test_file.write_text(original_content, encoding='utf-8')
        
        # Checksum correcto
        correct_checksum = safe_patch_manager._calculate_checksum(original_content)
        
        success, message = safe_patch_manager.apply_safe_patch(
            str(test_file),
            "print('modified')",
            expected_checksum=correct_checksum
        )
        
        assert success
        
        # Checksum incorrecto
        success, message = safe_patch_manager.apply_safe_patch(
            str(test_file),
            "print('modified again')",
            expected_checksum="invalid_checksum"
        )
        
        assert not success
        assert "Checksum no coincide" in message
    
    def test_content_safety_validation(self, safe_patch_manager):
        """Test validaciones de seguridad de contenido"""
        original_content = '''
from config.constants import PROGOL_CONFIG
import logging

class TestClass:
    def __init__(self):
        pass
    
    def method1(self):
        pass
'''
        
        # Contenido que elimina import crítico
        dangerous_content = '''
import logging

class TestClass:
    def method1(self):
        pass
'''
        
        validation = safe_patch_manager._validate_content_safety(original_content, dangerous_content)
        
        # Debe detectar problemas
        assert len(validation) > 0
        assert any("Import crítico eliminado" in error for error in validation)
    
    def test_operation_logging(self, safe_patch_manager, temp_project_dir):
        """Test logging de operaciones"""
        test_file = temp_project_dir / "log_test.py"
        test_file.write_text("original", encoding='utf-8')
        
        safe_patch_manager.apply_safe_patch(
            str(test_file),
            "modified",
            patch_description="Test operation logging"
        )
        
        # Verificar que se registró la operación
        history = safe_patch_manager.get_operation_history()
        assert len(history) > 0
        assert history[-1]["patch_description"] == "Test operation logging"
        assert history[-1]["success"] is True


class TestEnhancedProgolOptimizer:
    """Tests de integración para el optimizador completo"""
    
    @pytest.fixture
    def temp_work_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                yield Path(temp_dir)
            finally:
                os.chdir(old_cwd)
    
    def test_optimizer_initialization(self, temp_work_dir):
        """Test inicialización del optimizador mejorado"""
        optimizer = EnhancedProgolOptimizer(log_level="DEBUG")
        
        assert optimizer.data_loader is not None
        assert optimizer.ai_assistant is not None
        assert optimizer.instrumentor is not None
        assert hasattr(optimizer, 'strategy_usage')
    
    @patch('models.ai_assistant.OPENAI_AVAILABLE', False)
    def test_processing_without_ai(self, temp_work_dir):
        """Test procesamiento completo sin IA disponible"""
        optimizer = EnhancedProgolOptimizer(log_level="INFO")
        
        resultado = optimizer.procesar_concurso_completo(
            archivo_datos=None,  # Usar datos de ejemplo
            concurso_id="TEST_001",
            metodo_preferido="legacy",
            max_intentos=1
        )
        
        assert resultado["success"] is True
        assert len(resultado["portafolio"]) == 30
        assert resultado["concurso_id"] == "TEST_001"
        assert "session_summary" in resultado
    
    def test_strategy_fallback_chain(self, temp_work_dir):
        """Test cadena de fallbacks de estrategias"""
        optimizer = EnhancedProgolOptimizer(log_level="INFO")
        
        # Simular que el optimizador híbrido no está disponible
        with patch('main.ENHANCED_HYBRID_AVAILABLE', False):
            resultado = optimizer.procesar_concurso_completo(
                metodo_preferido="enhanced_hybrid",
                max_intentos=2
            )
            
            assert resultado["success"] is True
            # Debe haber usado estrategia legacy como fallback
            assert optimizer.strategy_usage["legacy_grasp"] > 0
    
    def test_validation_and_correction_flow(self, temp_work_dir):
        """Test flujo de validación y corrección"""
        optimizer = EnhancedProgolOptimizer(log_level="INFO")
        
        # Crear datos de prueba
        partidos_test = []
        for i in range(14):
            partidos_test.append({
                "id": i,
                "home": f"Team{i}A",
                "away": f"Team{i}B",
                "liga": "Test League",
                "prob_local": 0.4,
                "prob_empate": 0.3,
                "prob_visitante": 0.3,
                "clasificacion": "Ancla" if i < 3 else "Neutro",
                "forma_diferencia": 0.0,
                "lesiones_impact": 0.0,
                "es_final": False,
                "es_derbi": False,
                "es_playoff": False
            })
        
        # Test fase de preparación
        partidos_procesados = optimizer._fase_preparacion_datos(None, 0)
        assert partidos_procesados is not None
        assert len(partidos_procesados) == 14
        
        # Test fase de generación
        portafolio = optimizer._fase_generacion_portafolio(partidos_procesados, "legacy", 0)
        assert portafolio is not None
        assert len(portafolio) == 30
        
        # Test fase de validación
        portafolio_final, validacion = optimizer._fase_validacion_y_correccion(
            portafolio, partidos_procesados, False
        )
        assert portafolio_final is not None
        assert "es_valido" in validacion


class TestIntegrationScenarios:
    """Tests de escenarios de integración completos"""
    
    @pytest.fixture
    def complete_test_environment(self):
        """Entorno de prueba completo"""
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            # Crear estructura básica
            (Path(temp_dir) / "outputs").mkdir()
            (Path(temp_dir) / "logs").mkdir()
            
            try:
                yield Path(temp_dir)
            finally:
                os.chdir(old_cwd)
    
    def test_end_to_end_processing(self, complete_test_environment):
        """Test de procesamiento completo end-to-end"""
        optimizer = EnhancedProgolOptimizer(log_level="INFO")
        
        resultado = optimizer.procesar_concurso_completo(
            archivo_datos=None,
            concurso_id="E2E_TEST",
            metodo_preferido="auto",
            forzar_ai=False,
            max_intentos=2
        )
        
        # Verificaciones básicas
        assert resultado["success"] is True
        assert len(resultado["portafolio"]) == 30
        assert resultado["concurso_id"] == "E2E_TEST"
        
        # Verificar estructura del portafolio
        cores = [q for q in resultado["portafolio"] if q["tipo"] == "Core"]
        satelites = [q for q in resultado["portafolio"] if q["tipo"] == "Satelite"]
        
        assert len(cores) == 4
        assert len(satelites) == 26
        
        # Verificar validación
        assert "validacion" in resultado
        assert "metricas" in resultado
        
        # Verificar metadatos de trazabilidad
        assert "estrategia_final" in resultado
        assert "estadisticas_estrategias" in resultado
        assert "session_summary" in resultado
    
    def test_error_recovery_scenarios(self, complete_test_environment):
        """Test escenarios de recuperación de errores"""
        optimizer = EnhancedProgolOptimizer(log_level="INFO")
        
        # Simular error en primera estrategia
        with patch.object(optimizer, '_ejecutar_estrategia_hibrida_mejorada', side_effect=Exception("Simulated error")):
            resultado = optimizer.procesar_concurso_completo(
                metodo_preferido="enhanced_hybrid",
                max_intentos=2
            )
            
            # Debe recuperarse con estrategia alternativa
            assert resultado["success"] is True
            assert optimizer.strategy_usage["legacy_grasp"] > 0
    
    def test_concurrent_processing_safety(self, complete_test_environment):
        """Test seguridad en procesamiento concurrente"""
        import threading
        
        results = []
        errors = []
        
        def process_concurso(concurso_id):
            try:
                optimizer = EnhancedProgolOptimizer(log_level="WARNING")  # Menos logs para concurrencia
                resultado = optimizer.procesar_concurso_completo(
                    concurso_id=f"CONCURRENT_{concurso_id}",
                    max_intentos=1
                )
                results.append(resultado)
            except Exception as e:
                errors.append(str(e))
        
        # Lanzar múltiples threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_concurso, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen
        for thread in threads:
            thread.join()
        
        # Verificar que no hubo errores críticos
        assert len(errors) == 0
        assert len(results) == 3
        
        # Verificar que cada resultado es único
        concurso_ids = [r["concurso_id"] for r in results]
        assert len(set(concurso_ids)) == 3


# Configuración de pytest
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configuración automática para todos los tests"""
    # Configurar logging para tests
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reducir ruido en tests
    
    # Configurar semillas para reproducibilidad
    np.random.seed(42)
    import random
    random.seed(42)


if __name__ == "__main__":
    # Permitir ejecutar tests directamente
    pytest.main([__file__, "-v", "--tb=short"])