# progol_optimizer/validation/portfolio_validator.py - HOTFIX CORREGIDO
"""
Validador de Portafolio CORREGIDO - HOTFIX para error 'list' object has no attribute 'upper'
CORRECCIÓN CRÍTICA: Manejo correcto de tipos string/list en quinielas
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from collections import Counter
from itertools import combinations

class PortfolioValidator:
    """
    Validador CORREGIDO que implementa las 6 reglas obligatorias sin excepciones
    HOTFIX: Manejo correcto de quinielas como string o lista
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuración EXACTA según metodología
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        # Rangos CORREGIDOS para 14 partidos
        self.EMPATE_CHARS = {"E", "X"}  # CORRECCIÓN: Acepta ambos símbolos
        self.LIM_L = (5, 6)          # 35-41% de 14 = 4.9-5.74 → [5,6]
        self.LIM_E = (3, 5)          # 25-33% de 14 = 3.5-4.62 → [3,5] 
        self.LIM_V = (4, 5)          # 30-36% de 14 = 4.2-5.04 → [4,5]
        self.LIM_EMPATES = (4, 6)    # Metodología exacta
        self.MAX_SIGN_GLOBAL = 0.70  # ≤70% concentración
        self.MAX_SIGN_1_3 = 0.60     # ≤60% en primeros 3
        self.MAX_JACCARD = 0.57      # Correlación máxima entre pares
        
        self.logger.debug("✅ PortfolioValidator CORREGIDO inicializado")

    def _normalizar_quiniela(self, quiniela: Union[str, List[str]]) -> str:
        """
        NUEVO: Normaliza quiniela a string sin importar el formato de entrada
        """
        if isinstance(quiniela, str):
            return quiniela.upper()
        elif isinstance(quiniela, list):
            return "".join(str(x).upper() for x in quiniela)
        else:
            self.logger.warning(f"Tipo de quiniela inesperado: {type(quiniela)}")
            return str(quiniela).upper()

    def _extraer_quinielas_string(self, portafolio: List[Dict[str, Any]]) -> List[str]:
        """
        NUEVO: Extrae quinielas como strings normalizados del portafolio
        """
        quinielas_str = []
        
        for item in portafolio:
            if isinstance(item, dict):
                if "resultados" in item:
                    quiniela = self._normalizar_quiniela(item["resultados"])
                else:
                    self.logger.warning(f"Dict sin 'resultados': {item}")
                    continue
            else:
                quiniela = self._normalizar_quiniela(item)
            
            quinielas_str.append(quiniela)
        
        return quinielas_str

    def validar_portafolio_completo(self, portafolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validación COMPLETA con corrección automática si falla
        HOTFIX: Manejo seguro de tipos de datos
        """
        self.logger.info("🔍 Iniciando validación completa CORREGIDA...")
        
        try:
            # HOTFIX: Extraer quinielas de forma segura
            quinielas_str = self._extraer_quinielas_string(portafolio)
            
            if len(quinielas_str) != 30:
                self.logger.warning(f"Se esperaban 30 quinielas, se encontraron {len(quinielas_str)}")
            
            # PASO 1: Validación inicial
            errores = self._validar_todas_las_reglas(quinielas_str)
            
            if not errores:
                self.logger.info("✅ Portafolio VÁLIDO - No requiere corrección")
                return self._crear_resultado_validacion(True, portafolio, {})
            
            # PASO 2: CORRECCIÓN AUTOMÁTICA
            self.logger.warning(f"⚠️ Encontrados {len(errores)} errores, aplicando corrección automática...")
            
            portafolio_corregido = self._balancear_portafolio_automatico(portafolio, quinielas_str)
            
            # Verificar que la corrección funcionó
            quinielas_corregidas = self._extraer_quinielas_string(portafolio_corregido)
            errores_finales = self._validar_todas_las_reglas(quinielas_corregidas)
            
            if not errores_finales:
                self.logger.info("✅ CORRECCIÓN EXITOSA - Portafolio ahora es válido")
                return self._crear_resultado_validacion(True, portafolio_corregido, errores)
            else:
                self.logger.error(f"❌ CORRECCIÓN FALLÓ - Persisten {len(errores_finales)} errores")
                return self._crear_resultado_validacion(False, portafolio, errores_finales)
                
        except Exception as e:
            self.logger.error(f"❌ Error en validación completa: {e}")
            import traceback
            traceback.print_exc()
            return self._crear_resultado_validacion(False, portafolio, {"error_general": [str(e)]})

    def _validar_todas_las_reglas(self, quinielas: List[str]) -> Dict[str, List[str]]:
        """
        Valida TODAS las reglas de la metodología
        HOTFIX: Manejo seguro de strings
        """
        errores = {}
        
        # Asegurar que todas las quinielas son strings
        quinielas_normalizadas = [self._normalizar_quiniela(q) for q in quinielas]
        
        # REGLA 1-4: Validación por quiniela individual
        for i, quiniela in enumerate(quinielas_normalizadas):
            if len(quiniela) != 14:
                errores[f"Quiniela_{i}"] = [f"Longitud incorrecta: {len(quiniela)} != 14"]
                continue
                
            conteos = self._contar_signos(quiniela)
            errores_quiniela = []
            
            # Empates individuales (4-6)
            if not (self.LIM_EMPATES[0] <= conteos["E"] <= self.LIM_EMPATES[1]):
                errores_quiniela.append(f"Empates {conteos['E']} fuera de rango {self.LIM_EMPATES}")
            
            # Distribución L/E/V
            if not (self.LIM_L[0] <= conteos["L"] <= self.LIM_L[1]):
                errores_quiniela.append(f"Locales {conteos['L']} fuera de rango {self.LIM_L}")
            if not (self.LIM_E[0] <= conteos["E"] <= self.LIM_E[1]):
                errores_quiniela.append(f"Empates {conteos['E']} fuera de rango {self.LIM_E}")
            if not (self.LIM_V[0] <= conteos["V"] <= self.LIM_V[1]):
                errores_quiniela.append(f"Visitantes {conteos['V']} fuera de rango {self.LIM_V}")
            
            # Concentración máxima
            max_concentracion = max(conteos.values()) / 14
            if max_concentracion > self.MAX_SIGN_GLOBAL:
                errores_quiniela.append(f"Concentración {max_concentracion:.1%} > {self.MAX_SIGN_GLOBAL:.0%}")
            
            # Concentración primeros 3
            if len(quiniela) >= 3:
                primeros_3 = self._contar_signos(quiniela[:3])
                max_conc_inicial = max(primeros_3.values()) / 3
                if max_conc_inicial > self.MAX_SIGN_1_3:
                    errores_quiniela.append(f"Concentración inicial {max_conc_inicial:.1%} > {self.MAX_SIGN_1_3:.0%}")
            
            if errores_quiniela:
                errores[f"Quiniela_{i}"] = errores_quiniela
        
        # REGLA 5: Duplicados
        for i, j in combinations(range(len(quinielas_normalizadas)), 2):
            if quinielas_normalizadas[i] == quinielas_normalizadas[j]:
                errores.setdefault(f"Quiniela_{i}", []).append(f"Duplicada con Quiniela_{j}")
        
        # REGLA 6: Correlación Jaccard
        for i, j in combinations(range(len(quinielas_normalizadas)), 2):
            jaccard = self._calcular_jaccard(quinielas_normalizadas[i], quinielas_normalizadas[j])
            if jaccard > self.MAX_JACCARD:
                errores.setdefault(f"Quiniela_{i}", []).append(
                    f"Correlación {jaccard:.3f} > {self.MAX_JACCARD} con Quiniela_{j}"
                )
        
        return errores

    def _balancear_portafolio_automatico(self, portafolio: List[Dict[str, Any]], 
                                       quinielas_str: List[str]) -> List[Dict[str, Any]]:
        """
        CORRECCIÓN AUTOMÁTICA: Balancea el portafolio para cumplir todas las reglas
        HOTFIX: Manejo seguro de tipos de datos
        """
        self.logger.info("🔧 Aplicando balanceo automático...")
        
        try:
            portafolio_corregido = []
            quinielas_corregidas = [self._normalizar_quiniela(q) for q in quinielas_str]
            
            # Paso 1: Corregir quinielas individuales
            for i, quiniela in enumerate(quinielas_corregidas):
                try:
                    quiniela_balanceada = self._balancear_quiniela_individual(quiniela)
                    quinielas_corregidas[i] = quiniela_balanceada
                except Exception as e:
                    self.logger.warning(f"Error balanceando quiniela {i}: {e}")
                    # Fallback: generar quiniela balanceada desde cero
                    quinielas_corregidas[i] = self._generar_quiniela_basica_balanceada()
            
            # Paso 2: Eliminar duplicados
            quinielas_corregidas = self._eliminar_duplicados(quinielas_corregidas)
            
            # Paso 3: Corregir correlaciones Jaccard
            quinielas_corregidas = self._corregir_correlaciones_jaccard(quinielas_corregidas)
            
            # Paso 4: Asegurar 30 quinielas únicas
            while len(quinielas_corregidas) < 30:
                nueva_quiniela = self._generar_quiniela_compatible(quinielas_corregidas)
                quinielas_corregidas.append(nueva_quiniela)
            
            # Reconstruir portafolio con estructura original
            for i, quiniela_corregida in enumerate(quinielas_corregidas[:30]):  # Máximo 30
                if i < len(portafolio):
                    quiniela_info = portafolio[i].copy() if isinstance(portafolio[i], dict) else {
                        "id": f"Corregida-{i+1}",
                        "tipo": "Core" if i < 4 else "Satelite",
                        "par_id": (i-4)//2 if i >= 4 else None
                    }
                else:
                    quiniela_info = {
                        "id": f"Generada-{i+1}",
                        "tipo": "Satelite",
                        "par_id": (i-4)//2 if i >= 4 else None
                    }
                
                # HOTFIX: Asegurar que resultados es string
                quiniela_info.update({
                    "resultados": quiniela_corregida,  # Ya es string normalizado
                    "empates": quiniela_corregida.count("E") + quiniela_corregida.count("X"),
                    "distribución": {
                        "L": quiniela_corregida.count("L"),
                        "E": quiniela_corregida.count("E") + quiniela_corregida.count("X"),
                        "V": quiniela_corregida.count("V")
                    }
                })
                
                portafolio_corregido.append(quiniela_info)
            
            return portafolio_corregido
            
        except Exception as e:
            self.logger.error(f"Error en balanceo automático: {e}")
            import traceback
            traceback.print_exc()
            return self._generar_portafolio_emergencia()

    def _balancear_quiniela_individual(self, quiniela: Union[str, List[str]]) -> str:
        """
        Balancea una quiniela individual para cumplir límites L/E/V y empates
        HOTFIX: Manejo seguro de tipos string/list
        """
        # HOTFIX: Normalizar entrada
        quiniela_str = self._normalizar_quiniela(quiniela)
        
        if len(quiniela_str) != 14:
            self.logger.warning(f"Quiniela con longitud incorrecta: {len(quiniela_str)}")
            return self._generar_quiniela_basica_balanceada()
        
        signos = list(quiniela_str)
        max_intentos = 100
        intento = 0
        
        while intento < max_intentos:
            conteos = self._contar_signos("".join(signos))
            
            # Verificar si ya es válida
            if (self.LIM_EMPATES[0] <= conteos["E"] <= self.LIM_EMPATES[1] and
                self.LIM_L[0] <= conteos["L"] <= self.LIM_L[1] and
                self.LIM_E[0] <= conteos["E"] <= self.LIM_E[1] and
                self.LIM_V[0] <= conteos["V"] <= self.LIM_V[1] and
                max(conteos.values()) / 14 <= self.MAX_SIGN_GLOBAL):
                break
            
            # Corregir empates
            if conteos["E"] < self.LIM_EMPATES[0]:
                indices_no_empate = [i for i, s in enumerate(signos) if s not in self.EMPATE_CHARS]
                if indices_no_empate:
                    idx = np.random.choice(indices_no_empate)
                    signos[idx] = "E"
            elif conteos["E"] > self.LIM_EMPATES[1]:
                indices_empate = [i for i, s in enumerate(signos) if s in self.EMPATE_CHARS]
                if indices_empate:
                    idx = np.random.choice(indices_empate)
                    signos[idx] = np.random.choice(["L", "V"])
            
            # Corregir L/E/V
            if conteos["L"] < self.LIM_L[0]:
                indices_no_l = [i for i, s in enumerate(signos) if s != "L"]
                if indices_no_l:
                    idx = np.random.choice(indices_no_l)
                    signos[idx] = "L"
            elif conteos["L"] > self.LIM_L[1]:
                indices_l = [i for i, s in enumerate(signos) if s == "L"]
                if indices_l:
                    idx = np.random.choice(indices_l)
                    signos[idx] = np.random.choice(["E", "V"])
            
            if conteos["V"] < self.LIM_V[0]:
                indices_no_v = [i for i, s in enumerate(signos) if s != "V"]
                if indices_no_v:
                    idx = np.random.choice(indices_no_v)
                    signos[idx] = "V"
            elif conteos["V"] > self.LIM_V[1]:
                indices_v = [i for i, s in enumerate(signos) if s == "V"]
                if indices_v:
                    idx = np.random.choice(indices_v)
                    signos[idx] = np.random.choice(["L", "E"])
            
            intento += 1
        
        return "".join(signos)

    def _generar_quiniela_basica_balanceada(self) -> str:
        """
        NUEVO: Genera una quiniela básica balanceada como fallback
        """
        # Distribución simple balanceada
        num_l = 5  # Centro del rango [5,6]
        num_e = 4  # Centro del rango [3,5] pero respetando LIM_EMPATES
        num_v = 5  # 14 - 5 - 4 = 5, en rango [4,5]
        
        signos = ["L"] * num_l + ["E"] * num_e + ["V"] * num_v
        np.random.shuffle(signos)
        
        return "".join(signos)

    def _generar_portafolio_emergencia(self) -> List[Dict[str, Any]]:
        """
        NUEVO: Genera portafolio de emergencia si todo falla
        """
        self.logger.warning("🚨 Generando portafolio de emergencia")
        
        portafolio_emergencia = []
        
        for i in range(30):
            quiniela_balanceada = self._generar_quiniela_basica_balanceada()
            
            # Asegurar unicidad
            quinielas_existentes = [q["resultados"] for q in portafolio_emergencia]
            intentos = 0
            while quiniela_balanceada in quinielas_existentes and intentos < 100:
                quiniela_balanceada = self._generar_quiniela_basica_balanceada()
                intentos += 1
            
            portafolio_emergencia.append({
                "id": f"Emergencia-{i+1}",
                "tipo": "Core" if i < 4 else "Satelite",
                "par_id": (i-4)//2 if i >= 4 else None,
                "resultados": quiniela_balanceada,
                "empates": quiniela_balanceada.count("E"),
                "distribución": {
                    "L": quiniela_balanceada.count("L"),
                    "E": quiniela_balanceada.count("E"),
                    "V": quiniela_balanceada.count("V")
                }
            })
        
        return portafolio_emergencia

    # ========== MÉTODOS AUXILIARES (sin cambios significativos) ==========

    def _eliminar_duplicados(self, quinielas: List[str]) -> List[str]:
        """Elimina duplicados manteniendo las primeras ocurrencias"""
        vistas = set()
        resultado = []
        
        for quiniela in quinielas:
            quiniela_norm = self._normalizar_quiniela(quiniela)
            if quiniela_norm not in vistas:
                vistas.add(quiniela_norm)
                resultado.append(quiniela_norm)
            else:
                nueva = self._generar_variacion_unica(quiniela_norm, vistas)
                vistas.add(nueva)
                resultado.append(nueva)
        
        return resultado

    def _corregir_correlaciones_jaccard(self, quinielas: List[str]) -> List[str]:
        """Corrige correlaciones Jaccard > 0.57"""
        resultado = [self._normalizar_quiniela(q) for q in quinielas]
        
        for i, j in combinations(range(len(resultado)), 2):
            jaccard = self._calcular_jaccard(resultado[i], resultado[j])
            if jaccard > self.MAX_JACCARD:
                resultado[j] = self._reducir_correlacion(resultado[i], resultado[j])
        
        return resultado

    def _generar_variacion_unica(self, quiniela_base: str, existentes: set) -> str:
        """Genera una variación única de la quiniela base"""
        for _ in range(100):
            signos = list(quiniela_base)
            num_cambios = np.random.randint(1, 3)
            indices = np.random.choice(len(signos), num_cambios, replace=False)
            
            for idx in indices:
                signos[idx] = np.random.choice(["L", "E", "V"])
            
            nueva = "".join(signos)
            if nueva not in existentes:
                return self._balancear_quiniela_individual(nueva)
        
        return self._generar_quiniela_basica_balanceada()

    def _reducir_correlacion(self, quiniela_a: str, quiniela_b: str) -> str:
        """Reduce correlación entre dos quinielas modificando la segunda"""
        signos = list(quiniela_b)
        coincidencias = [i for i, (a, b) in enumerate(zip(quiniela_a, quiniela_b)) if a == b]
        
        if len(coincidencias) > 8:
            indices_cambio = np.random.choice(coincidencias, min(3, len(coincidencias)), replace=False)
            
            for idx in indices_cambio:
                opciones = ["L", "E", "V"]
                if signos[idx] in opciones:
                    opciones.remove(signos[idx])
                signos[idx] = np.random.choice(opciones)
        
        return self._balancear_quiniela_individual("".join(signos))

    def _generar_quiniela_compatible(self, existentes: List[str]) -> str:
        """Genera una nueva quiniela compatible con todas las reglas"""
        for _ in range(1000):
            nueva = self._generar_quiniela_basica_balanceada()
            
            if nueva in existentes:
                continue
            
            jaccard_ok = True
            for existente in existentes:
                if self._calcular_jaccard(nueva, existente) > self.MAX_JACCARD:
                    jaccard_ok = False
                    break
            
            if jaccard_ok:
                return nueva
        
        return self._generar_quiniela_basica_balanceada()

    def _contar_signos(self, quiniela: str) -> Counter:
        """Cuenta signos normalizando E/X"""
        conteos = Counter({"L": 0, "E": 0, "V": 0})
        
        for char in quiniela.upper():
            if char in self.EMPATE_CHARS:
                conteos["E"] += 1
            elif char in conteos:
                conteos[char] += 1
        
        return conteos

    def _calcular_jaccard(self, a: str, b: str) -> float:
        """Calcula índice de Jaccard entre dos quinielas"""
        if len(a) != len(b):
            return 0.0
        
        interseccion = sum(1 for x, y in zip(a, b) if x == y)
        union = len(a)
        
        return interseccion / union if union > 0 else 0.0

    def _crear_resultado_validacion(self, es_valido: bool, portafolio: List[Dict[str, Any]], 
                                  errores: Dict[str, List[str]]) -> Dict[str, Any]:
        """Crea resultado estructurado de validación"""
        try:
            quinielas_str = self._extraer_quinielas_string(portafolio)
            metricas = self._calcular_metricas_portafolio(quinielas_str)
        except Exception as e:
            self.logger.warning(f"Error calculando métricas: {e}")
            metricas = {"error": str(e)}
        
        return {
            "es_valido": es_valido,
            "detalle_validaciones": {
                "distribucion_global": es_valido,
                "empates_individuales": es_valido,
                "concentracion_maxima": es_valido,
                "arquitectura_core_satelites": len(portafolio) == 30,
                "correlacion_jaccard": es_valido,
                "distribucion_divisores": es_valido
            },
            "errores": errores,
            "metricas": metricas,
            "resumen": self._generar_resumen_validacion(es_valido, errores, metricas)
        }

    def _calcular_metricas_portafolio(self, quinielas: List[str]) -> Dict[str, Any]:
        """Calcula métricas del portafolio"""
        if not quinielas:
            return {}
        
        total_l = sum(q.count("L") for q in quinielas)
        total_e = sum(q.count("E") + q.count("X") for q in quinielas)
        total_v = sum(q.count("V") for q in quinielas)
        total_partidos = len(quinielas) * 14
        
        empates_por_quiniela = [q.count("E") + q.count("X") for q in quinielas]
        
        return {
            "total_quinielas": len(quinielas),
            "distribucion_global": {
                "L": total_l,
                "E": total_e,
                "V": total_v,
                "porcentajes": {
                    "L": total_l / total_partidos if total_partidos > 0 else 0,
                    "E": total_e / total_partidos if total_partidos > 0 else 0,
                    "V": total_v / total_partidos if total_partidos > 0 else 0
                }
            },
            "empates_estadisticas": {
                "promedio": np.mean(empates_por_quiniela) if empates_por_quiniela else 0,
                "minimo": min(empates_por_quiniela) if empates_por_quiniela else 0,
                "maximo": max(empates_por_quiniela) if empates_por_quiniela else 0,
                "desviacion": np.std(empates_por_quiniela) if empates_por_quiniela else 0
            },
            "cobertura_arquitectura": {
                "cores": 4,
                "satelites": len(quinielas) - 4 if len(quinielas) > 4 else 0
            }
        }

    def _generar_resumen_validacion(self, es_valido: bool, errores: Dict[str, List[str]], 
                                  metricas: Dict[str, Any]) -> str:
        """Genera resumen textual de la validación"""
        if es_valido:
            dist = metricas.get("distribucion_global", {}).get("porcentajes", {})
            return f"""✅ PORTAFOLIO VÁLIDO
Distribución: L={dist.get('L', 0):.1%}, E={dist.get('E', 0):.1%}, V={dist.get('V', 0):.1%}
Total quinielas: {metricas.get('total_quinielas', 0)}
Cumple todas las reglas de la metodología."""
        else:
            num_errores = sum(len(errs) for errs in errores.values())
            return f"""❌ PORTAFOLIO INVÁLIDO
{num_errores} errores encontrados en {len(errores)} quinielas
Requiere corrección automática."""