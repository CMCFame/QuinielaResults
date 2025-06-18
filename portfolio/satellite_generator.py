# progol_optimizer/portfolio/satellite_generator.py - VERSIÓN MEJORADA
"""
Generador de Satélites MEJORADO - Mejor diversidad y evita concentraciones excesivas
CORRECCIÓN: Algoritmo inteligente que maximiza diversidad por posición desde la generación
"""

import logging
import random
from typing import List, Dict, Any, Tuple

class SatelliteGenerator:
    """
    Genera pares de satélites con mejor diversidad y balance desde el inicio
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.correlacion_max = self.config["ARQUITECTURA_PORTAFOLIO"]["correlacion_jaccard_max"]
        self.concentracion_max = self.config["CONCENTRACION_MAX_GENERAL"]
        
        self.logger.debug(f"SatelliteGenerator MEJORADO: diversidad inteligente habilitada")
    
    def generar_pares_satelites(self, partidos_clasificados: List[Dict[str, Any]], num_satelites: int) -> List[Dict[str, Any]]:
        """
        Genera satélites con algoritmo MEJORADO que maximiza diversidad desde el inicio
        """
        if num_satelites % 2 != 0:
            raise ValueError(f"Número de satélites debe ser par, recibido: {num_satelites}")
        
        num_pares = num_satelites // 2
        
        self.logger.info(f"🔄 Generando {num_satelites} satélites MEJORADOS con diversidad inteligente...")
        
        satelites = []
        
        # Estrategia de diversidad: alternar tipos de pares para maximizar diversidad global
        estrategias_par = self._planificar_estrategias_diversidad(num_pares, partidos_clasificados)
        
        # Generar cada par con estrategia específica
        for par_id in range(num_pares):
            try:
                estrategia = estrategias_par[par_id]
                quiniela_a, quiniela_b = self._crear_par_con_estrategia(
                    partidos_clasificados, par_id, estrategia
                )
                
                # Crear objetos satélite
                correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
                
                satelite_a = {
                    "id": f"Sat-{par_id+1}A",
                    "tipo": "Satelite",
                    "resultados": quiniela_a,
                    "par_id": par_id,
                    "estrategia": estrategia,
                    "correlacion_jaccard": correlacion,
                    "empates": quiniela_a.count("E"),
                    "distribución": {
                        "L": quiniela_a.count("L"),
                        "E": quiniela_a.count("E"),
                        "V": quiniela_a.count("V")
                    }
                }
                
                satelite_b = {
                    "id": f"Sat-{par_id+1}B", 
                    "tipo": "Satelite",
                    "resultados": quiniela_b,
                    "par_id": par_id,
                    "estrategia": estrategia,
                    "correlacion_jaccard": correlacion,
                    "empates": quiniela_b.count("E"),
                    "distribución": {
                        "L": quiniela_b.count("L"),
                        "E": quiniela_b.count("E"),
                        "V": quiniela_b.count("V")
                    }
                }
                
                satelites.extend([satelite_a, satelite_b])
                
                self.logger.debug(f"✅ Par {par_id+1} ({estrategia}): correlación={correlacion:.3f}, "
                               f"empates=({satelite_a['empates']},{satelite_b['empates']})")
                
            except Exception as e:
                self.logger.error(f"❌ Error generando par {par_id}: {e}, usando par de emergencia.")
                quiniela_a, quiniela_b = self._crear_par_emergencia_mejorado(partidos_clasificados, par_id)
                correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
                satelites.extend([
                    {
                        "id": f"Sat-{par_id+1}A", "tipo": "Satelite", "resultados": quiniela_a,
                        "par_id": par_id, "estrategia": "emergencia", "correlacion_jaccard": correlacion, 
                        "empates": quiniela_a.count("E"),
                        "distribución": {"L": quiniela_a.count("L"), "E": quiniela_a.count("E"), "V": quiniela_a.count("V")}
                    },
                    {
                        "id": f"Sat-{par_id+1}B", "tipo": "Satelite", "resultados": quiniela_b,
                        "par_id": par_id, "estrategia": "emergencia", "correlacion_jaccard": correlacion,
                        "empates": quiniela_b.count("E"),
                        "distribución": {"L": quiniela_b.count("L"), "E": quiniela_b.count("E"), "V": quiniela_b.count("V")}
                    }
                ])
        
        # Validación final
        self._validar_satelites_y_diversidad(satelites)
        
        self.logger.info(f"✅ Generados {len(satelites)} satélites mejorados con diversidad maximizada")
        return satelites

    def _planificar_estrategias_diversidad(self, num_pares: int, partidos: List[Dict[str, Any]]) -> List[str]:
        """
        NUEVA FUNCIÓN: Planifica estrategias para cada par para maximizar diversidad global
        """
        # Identificar tipos de partidos
        anclas = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        divisores = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Divisor"]
        tendencias_empate = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "TendenciaEmpate"]
        neutros = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Neutro"]
        
        estrategias = []
        
        # Estrategia balanceada para diversidad máxima
        tipos_estrategia = ["divisor_flip", "empate_boost", "local_boost", "visitante_boost", "equilibrado"]
        
        for i in range(num_pares):
            # Rotar estrategias para asegurar diversidad
            estrategia = tipos_estrategia[i % len(tipos_estrategia)]
            estrategias.append(estrategia)
            
        self.logger.debug(f"Estrategias planificadas: {estrategias}")
        return estrategias

    def _crear_par_con_estrategia(self, partidos: List[Dict[str, Any]], par_id: int, estrategia: str) -> Tuple[List[str], List[str]]:
        """
        Crea un par de satélites usando la estrategia específica asignada
        """
        max_intentos = 15
        
        for intento in range(max_intentos):
            quiniela_a = [""] * 14
            quiniela_b = [""] * 14

            # Identificar partidos por tipo
            anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
            divisores_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Divisor"]
            otros_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") not in ["Ancla", "Divisor"]]

            # ANCLAS: Siempre idénticas (requisito crítico)
            for i in anclas_indices:
                resultado = self._get_resultado_max_prob(partidos[i])
                quiniela_a[i] = resultado
                quiniela_b[i] = resultado

            # APLICAR ESTRATEGIA ESPECÍFICA
            if estrategia == "divisor_flip":
                self._aplicar_estrategia_divisor_flip(quiniela_a, quiniela_b, divisores_indices, otros_indices, partidos)
            elif estrategia == "empate_boost":
                self._aplicar_estrategia_empate_boost(quiniela_a, quiniela_b, divisores_indices, otros_indices, partidos)
            elif estrategia == "local_boost":
                self._aplicar_estrategia_local_boost(quiniela_a, quiniela_b, divisores_indices, otros_indices, partidos)
            elif estrategia == "visitante_boost":
                self._aplicar_estrategia_visitante_boost(quiniela_a, quiniela_b, divisores_indices, otros_indices, partidos)
            else:  # equilibrado
                self._aplicar_estrategia_equilibrada(quiniela_a, quiniela_b, divisores_indices, otros_indices, partidos)

            # AJUSTE FINAL: Forzar 4-6 empates
            quiniela_a = self._ajustar_empates_inteligente(quiniela_a, partidos, anclas_indices)
            quiniela_b = self._ajustar_empates_inteligente(quiniela_b, partidos, anclas_indices)
            
            # VERIFICAR CONCENTRACIÓN
            if (not self._tiene_concentracion_excesiva(quiniela_a) and 
                not self._tiene_concentracion_excesiva(quiniela_b)):
                
                correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
                if correlacion <= self.correlacion_max:
                    return quiniela_a, quiniela_b

        # Si no se logra después de muchos intentos, usar método de emergencia
        self.logger.warning(f"Par {par_id} con estrategia {estrategia} requirió método de emergencia")
        return self._crear_par_emergencia_mejorado(partidos, par_id)

    def _aplicar_estrategia_divisor_flip(self, quiniela_a: List[str], quiniela_b: List[str], 
                                       divisores: List[int], otros: List[int], partidos: List[Dict[str, Any]]):
        """Estrategia: Flipear divisores, otros con máxima probabilidad"""
        for i in divisores:
            res_a = self._get_resultado_max_prob(partidos[i])
            res_b = self._get_resultado_opuesto_inteligente(res_a, partidos[i])
            quiniela_a[i] = res_a
            quiniela_b[i] = res_b
            
        for i in otros:
            resultado = self._get_resultado_max_prob(partidos[i])
            quiniela_a[i] = resultado
            quiniela_b[i] = resultado

    def _aplicar_estrategia_empate_boost(self, quiniela_a: List[str], quiniela_b: List[str], 
                                       divisores: List[int], otros: List[int], partidos: List[Dict[str, Any]]):
        """Estrategia: Forzar más empates en una quiniela"""
        # Quiniela A: más empates
        for i in divisores + otros:
            probs = [partidos[i]["prob_local"], partidos[i]["prob_empate"], partidos[i]["prob_visitante"]]
            if partidos[i]["prob_empate"] > 0.25:  # Si empate es razonable
                quiniela_a[i] = "E"
                quiniela_b[i] = self._get_resultado_max_prob(partidos[i])
            else:
                resultado = self._get_resultado_max_prob(partidos[i])
                quiniela_a[i] = resultado
                quiniela_b[i] = self._get_resultado_opuesto_inteligente(resultado, partidos[i])

    def _aplicar_estrategia_local_boost(self, quiniela_a: List[str], quiniela_b: List[str], 
                                      divisores: List[int], otros: List[int], partidos: List[Dict[str, Any]]):
        """Estrategia: Una quiniela favorece locales, otra visitantes"""
        for i in divisores + otros:
            # Quiniela A: Favor a locales si es razonable
            if partidos[i]["prob_local"] > 0.30:
                quiniela_a[i] = "L"
            else:
                quiniela_a[i] = self._get_resultado_max_prob(partidos[i])
                
            # Quiniela B: Favor a visitantes si es razonable  
            if partidos[i]["prob_visitante"] > 0.30:
                quiniela_b[i] = "V"
            else:
                quiniela_b[i] = self._get_resultado_max_prob(partidos[i])

    def _aplicar_estrategia_visitante_boost(self, quiniela_a: List[str], quiniela_b: List[str], 
                                          divisores: List[int], otros: List[int], partidos: List[Dict[str, Any]]):
        """Estrategia: Una quiniela favorece visitantes, otra empates"""
        for i in divisores + otros:
            # Quiniela A: Favor a visitantes
            if partidos[i]["prob_visitante"] > 0.30:
                quiniela_a[i] = "V"
            else:
                quiniela_a[i] = self._get_resultado_max_prob(partidos[i])
                
            # Quiniela B: Favor a empates
            if partidos[i]["prob_empate"] > 0.25:
                quiniela_b[i] = "E"
            else:
                quiniela_b[i] = self._get_resultado_max_prob(partidos[i])

    def _aplicar_estrategia_equilibrada(self, quiniela_a: List[str], quiniela_b: List[str], 
                                       divisores: List[int], otros: List[int], partidos: List[Dict[str, Any]]):
        """Estrategia: Distribución equilibrada"""
        for i in divisores + otros:
            resultado = self._get_resultado_max_prob(partidos[i])
            quiniela_a[i] = resultado
            # Alternar en quiniela B para crear diversidad
            if i % 2 == 0:
                quiniela_b[i] = self._get_resultado_opuesto_inteligente(resultado, partidos[i])
            else:
                quiniela_b[i] = resultado

    def _ajustar_empates_inteligente(self, quiniela: List[str], partidos: List[Dict[str, Any]], anclas_indices: List[int]) -> List[str]:
        """
        Ajuste inteligente de empates que evita crear concentración excesiva
        """
        quiniela_ajustada = quiniela.copy()
        modificables_indices = [i for i in range(14) if i not in anclas_indices]

        empates_actuales = quiniela_ajustada.count("E")

        # Necesitamos MÁS empates
        if empates_actuales < self.empates_min:
            necesarios = self.empates_min - empates_actuales
            # Buscar candidatos que NO creen concentración excesiva
            candidatos = []
            for i in modificables_indices:
                if quiniela_ajustada[i] in ["L", "V"]:
                    # Simular el cambio y verificar concentración
                    test_quiniela = quiniela_ajustada.copy()
                    test_quiniela[i] = "E"
                    if not self._tiene_concentracion_excesiva(test_quiniela):
                        candidatos.append((i, partidos[i]["prob_empate"]))
            
            # Ordenar por probabilidad de empate (los más probables primero)
            candidatos.sort(key=lambda x: x[1], reverse=True)
            
            # Convertir los mejores candidatos
            for i in range(min(necesarios, len(candidatos))):
                idx = candidatos[i][0]
                quiniela_ajustada[idx] = "E"

        # Necesitamos MENOS empates
        elif empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            candidatos = []
            for i in modificables_indices:
                if quiniela_ajustada[i] == "E":
                    candidatos.append((i, partidos[i]["prob_empate"]))
            
            # Ordenar por probabilidad de empate (los menos probables primero)
            candidatos.sort(key=lambda x: x[1])
            
            # Convertir los peores candidatos
            for i in range(min(exceso, len(candidatos))):
                idx = candidatos[i][0]
                partido = partidos[idx]
                # Elegir el mejor resultado alternativo
                if partido["prob_local"] > partido["prob_visitante"]:
                    nuevo_resultado = "L"
                else:
                    nuevo_resultado = "V"
                
                # Verificar que no se cree concentración excesiva
                test_quiniela = quiniela_ajustada.copy()
                test_quiniela[idx] = nuevo_resultado
                if not self._tiene_concentracion_excesiva(test_quiniela):
                    quiniela_ajustada[idx] = nuevo_resultado

        return quiniela_ajustada

    def _tiene_concentracion_excesiva(self, quiniela: List[str]) -> bool:
        """Verifica si una quiniela tiene concentración excesiva"""
        # Concentración general
        max_conc_general = max(quiniela.count(s) for s in ["L", "E", "V"]) / 14.0
        if max_conc_general > self.concentracion_max:
            return True
            
        # Concentración inicial
        primeros_3 = quiniela[:3]
        max_conc_inicial = max(primeros_3.count(s) for s in ["L", "E", "V"]) / 3.0
        if max_conc_inicial > self.config["CONCENTRACION_MAX_INICIAL"]:
            return True
            
        return False
        
    def _crear_par_emergencia_mejorado(self, partidos: List[Dict[str, Any]], par_id: int) -> Tuple[List[str], List[str]]:
        """
        Par de emergencia mejorado que evita concentración desde el inicio
        """
        self.logger.warning(f"🚨 Generando par de emergencia MEJORADO {par_id}")
        anclas_indices = [i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        
        # Base: distribución inteligente
        quiniela_a = []
        quiniela_b = []
        
        for i, partido in enumerate(partidos):
            if i in anclas_indices:
                # Anclas idénticas
                resultado = self._get_resultado_max_prob(partido)
                quiniela_a.append(resultado)
                quiniela_b.append(resultado)
            else:
                # Alternar para crear diversidad básica
                if i % 3 == 0:
                    quiniela_a.append("L" if partido["prob_local"] > 0.3 else self._get_resultado_max_prob(partido))
                    quiniela_b.append("V" if partido["prob_visitante"] > 0.3 else self._get_resultado_max_prob(partido))
                elif i % 3 == 1:
                    quiniela_a.append("E" if partido["prob_empate"] > 0.25 else self._get_resultado_max_prob(partido))
                    quiniela_b.append(self._get_resultado_max_prob(partido))
                else:
                    resultado = self._get_resultado_max_prob(partido)
                    quiniela_a.append(resultado)
                    quiniela_b.append(self._get_resultado_opuesto_inteligente(resultado, partido))

        # Ajustar empates sin crear concentración
        quiniela_a = self._ajustar_empates_inteligente(quiniela_a, partidos, anclas_indices)
        quiniela_b = self._ajustar_empates_inteligente(quiniela_b, partidos, anclas_indices)
        
        return quiniela_a, quiniela_b

    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        return max(probs, key=probs.get)

    def _get_resultado_opuesto_inteligente(self, resultado_actual: str, partido: Dict[str, Any]) -> str:
        """Obtiene resultado opuesto de forma inteligente (segunda mayor probabilidad)"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        # Eliminar la opción actual y devolver la mejor de las restantes
        del probs[resultado_actual]
        return max(probs, key=probs.get)

    def _calcular_correlacion_jaccard(self, quiniela_a: List[str], quiniela_b: List[str]) -> float:
        """Calcula correlación de Jaccard entre dos quinielas"""
        if len(quiniela_a) != len(quiniela_b) or len(quiniela_a) == 0: 
            return 0.0
        coincidencias = sum(1 for a, b in zip(quiniela_a, quiniela_b) if a == b)
        return coincidencias / len(quiniela_a)
    
    def _validar_satelites_y_diversidad(self, satelites: List[Dict[str, Any]]):
        """Validación robusta con énfasis en diversidad"""
        self.logger.debug("🔍 Validando satélites y diversidad...")
        errores = []
        
        # Validaciones básicas
        for satelite in satelites:
            empates = satelite["resultados"].count("E")
            if not (self.empates_min <= empates <= self.empates_max):
                errores.append(f"{satelite['id']}: empates {empates} fuera del rango [{self.empates_min}-{self.empates_max}]")
            if len(satelite["resultados"]) != 14:
                errores.append(f"{satelite['id']}: longitud {len(satelite['resultados'])} != 14")
            if self._tiene_concentracion_excesiva(satelite["resultados"]):
                errores.append(f"{satelite['id']}: concentración excesiva")
        
        # Validación de diversidad por posición
        self._validar_diversidad_posiciones(satelites)
        
        if errores:
            for error in errores:
                self.logger.error(f"  - {error}")
            raise ValueError(f"Validación de satélites falló: {', '.join(errores)}")
        
        self.logger.debug("✅ Todos los satélites y diversidad son válidos")

    def _validar_diversidad_posiciones(self, satelites: List[Dict[str, Any]]):
        """Valida que hay buena diversidad en cada posición"""
        total_satelites = len(satelites)
        if total_satelites == 0:
            return
            
        for posicion in range(14):
            conteos = {"L": 0, "E": 0, "V": 0}
            for satelite in satelites:
                resultado = satelite["resultados"][posicion]
                conteos[resultado] += 1
            
            # Verificar que ningún resultado domine excesivamente
            max_apariciones = total_satelites * 0.70  # 70% máximo
            for resultado, count in conteos.items():
                if count > max_apariciones:
                    self.logger.warning(f"⚠️ Posición {posicion+1}: {resultado} aparece {count}/{total_satelites} veces (>{max_apariciones:.1f})")
        
        self.logger.debug("✅ Diversidad por posiciones verificada")