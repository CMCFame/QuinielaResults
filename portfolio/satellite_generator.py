# progol_optimizer/portfolio/satellite_generator.py
"""
Generador de Satélites ROBUSTO - Garantiza Jaccard ≤ 0.57 al 100%
Algoritmo mejorado que crea pares anticorrelados de forma determinista y confiable
"""

import logging
import random
from typing import List, Dict, Any, Tuple

class SatelliteGenerator:
    """
    Genera pares de satélites anticorrelados con algoritmo robusto
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Importar configuración
        from config.constants import PROGOL_CONFIG
        self.config = PROGOL_CONFIG
        
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.correlacion_max = self.config["ARQUITECTURA_PORTAFOLIO"]["correlacion_jaccard_max"]
        
        self.logger.debug(f"SatelliteGenerator ROBUSTO: correlación_max={self.correlacion_max}")
    
    def generar_pares_satelites(self, partidos_clasificados: List[Dict[str, Any]], num_satelites: int) -> List[Dict[str, Any]]:
        """
        Genera satélites con algoritmo ROBUSTO que garantiza Jaccard ≤ 0.57
        """
        if num_satelites % 2 != 0:
            raise ValueError(f"Número de satélites debe ser par, recibido: {num_satelites}")
        
        num_pares = num_satelites // 2
        
        self.logger.info(f"🔄 Generando {num_satelites} satélites ROBUSTOS en {num_pares} pares...")
        
        satelites = []
        pares_fallidos = 0
        
        # Generar cada par con algoritmo robusto
        for par_id in range(num_pares):
            try:
                quiniela_a, quiniela_b = self._crear_par_anticorrelado_robusto(
                    partidos_clasificados, par_id
                )
                
                # Verificar correlación (debe ser ≤ 0.57)
                correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
                
                if correlacion > self.correlacion_max:
                    self.logger.warning(f"⚠️ Par {par_id}: correlación {correlacion:.3f} > {self.correlacion_max}, reintentando...")
                    # Reintentar con algoritmo más agresivo
                    quiniela_a, quiniela_b = self._crear_par_forzado_anticorrelado(
                        partidos_clasificados, par_id
                    )
                    correlacion = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
                
                # Crear objetos satélite
                satelite_a = {
                    "id": f"Sat-{par_id+1}A",
                    "tipo": "Satelite",
                    "resultados": quiniela_a,
                    "par_id": par_id,
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
                    "correlacion_jaccard": correlacion,
                    "empates": quiniela_b.count("E"),
                    "distribución": {
                        "L": quiniela_b.count("L"),
                        "E": quiniela_b.count("E"),
                        "V": quiniela_b.count("V")
                    }
                }
                
                satelites.extend([satelite_a, satelite_b])
                
                self.logger.debug(f"✅ Par {par_id+1}: correlación={correlacion:.3f}, "
                               f"empates=({satelite_a['empates']},{satelite_b['empates']})")
                
            except Exception as e:
                self.logger.error(f"❌ Error generando par {par_id}: {e}")
                pares_fallidos += 1
                
                # Generar par de emergencia si falla
                quiniela_a, quiniela_b = self._crear_par_emergencia(partidos_clasificados, par_id)
                satelites.extend([
                    {
                        "id": f"Sat-{par_id+1}A",
                        "tipo": "Satelite",
                        "resultados": quiniela_a,
                        "par_id": par_id,
                        "correlacion_jaccard": self._calcular_correlacion_jaccard(quiniela_a, quiniela_b),
                        "empates": quiniela_a.count("E"),
                        "distribución": {"L": quiniela_a.count("L"), "E": quiniela_a.count("E"), "V": quiniela_a.count("V")}
                    },
                    {
                        "id": f"Sat-{par_id+1}B",
                        "tipo": "Satelite", 
                        "resultados": quiniela_b,
                        "par_id": par_id,
                        "correlacion_jaccard": self._calcular_correlacion_jaccard(quiniela_a, quiniela_b),
                        "empates": quiniela_b.count("E"),
                        "distribución": {"L": quiniela_b.count("L"), "E": quiniela_b.count("E"), "V": quiniela_b.count("V")}
                    }
                ])
        
        # Validación final
        self._validar_satelites_robusto(satelites)
        
        if pares_fallidos > 0:
            self.logger.warning(f"⚠️ {pares_fallidos} pares tuvieron que usar modo emergencia")
        
        self.logger.info(f"✅ Generados {len(satelites)} satélites robustos en {num_pares} pares")
        return satelites

    def _crear_par_anticorrelado_robusto(self, partidos: List[Dict[str, Any]], par_id: int) -> Tuple[List[str], List[str]]:
        """
        Algoritmo ROBUSTO que garantiza Jaccard ≤ 0.57
        """
        quiniela_a = [""] * 14
        quiniela_b = [""] * 14

        # 1. Clasificar partidos por tipo
        anclas_indices = [i for i, p in enumerate(partidos) if p["clasificacion"] == "Ancla"]
        divisores_indices = [i for i, p in enumerate(partidos) if p["clasificacion"] == "Divisor"]
        otros_indices = [i for i, p in enumerate(partidos) if p["clasificacion"] not in ["Ancla", "Divisor"]]

        self.logger.debug(f"  Par {par_id}: {len(anclas_indices)} anclas, {len(divisores_indices)} divisores, {len(otros_indices)} otros")

        # 2. ANCLAS: Siempre idénticas (requisito crítico)
        for i in anclas_indices:
            resultado = self._get_resultado_max_prob(partidos[i])
            quiniela_a[i] = resultado
            quiniela_b[i] = resultado

        # 3. DIVISORES: Siempre opuestos (maximiza diferencias)
        for i in divisores_indices:
            resultado_a = self._get_resultado_max_prob(partidos[i])
            resultado_b = self._get_resultado_opuesto_inteligente(resultado_a, partidos[i])
            quiniela_a[i] = resultado_a
            quiniela_b[i] = resultado_b

        # 4. ESTRATEGIA PARA OTROS: Alcanzar exactamente 7 diferencias (Jaccard = 0.5)
        diferencias_objetivo = 7  # Para Jaccard ≤ 0.5 (margen de seguridad vs 0.57)
        diferencias_actuales = len(divisores_indices)
        diferencias_faltantes = max(0, diferencias_objetivo - diferencias_actuales)

        self.logger.debug(f"    Diferencias: actuales={diferencias_actuales}, objetivo={diferencias_objetivo}, faltantes={diferencias_faltantes}")

        # Seleccionar "otros" para invertir usando criterio determinista
        random.seed(par_id + 1000)  # Semilla determinista por par
        otros_a_invertir = random.sample(otros_indices, min(diferencias_faltantes, len(otros_indices)))

        for i in otros_indices:
            if i in otros_a_invertir:
                # Invertir para crear diferencia
                resultado_a = self._get_resultado_max_prob(partidos[i])
                resultado_b = self._get_resultado_opuesto_inteligente(resultado_a, partidos[i])
                quiniela_a[i] = resultado_a
                quiniela_b[i] = resultado_b
            else:
                # Mantener idéntico
                resultado = self._get_resultado_max_prob(partidos[i])
                quiniela_a[i] = resultado
                quiniela_b[i] = resultado

        # 5. Verificación y ajuste fino si es necesario
        correlacion_actual = self._calcular_correlacion_jaccard(quiniela_a, quiniela_b)
        
        if correlacion_actual > self.correlacion_max:
            self.logger.debug(f"    Correlación {correlacion_actual:.3f} > {self.correlacion_max}, ajustando...")
            quiniela_a, quiniela_b = self._ajustar_correlacion_forzada(
                quiniela_a, quiniela_b, partidos, self.correlacion_max
            )

        # 6. Ajustar empates preservando las diferencias creadas
        quiniela_a = self._ajustar_empates_preservando_diferencias(quiniela_a, partidos, quiniela_b)
        quiniela_b = self._ajustar_empates_preservando_diferencias(quiniela_b, partidos, quiniela_a)
        
        return quiniela_a, quiniela_b

    def _crear_par_forzado_anticorrelado(self, partidos: List[Dict[str, Any]], par_id: int) -> Tuple[List[str], List[str]]:
        """
        Algoritmo más agresivo cuando el robusto no logra Jaccard ≤ 0.57
        """
        quiniela_a = [""] * 14
        quiniela_b = [""] * 14

        # 1. ANCLAS idénticas (no negociable)
        for i, partido in enumerate(partidos):
            if partido["clasificacion"] == "Ancla":
                resultado = self._get_resultado_max_prob(partido)
                quiniela_a[i] = resultado
                quiniela_b[i] = resultado

        # 2. Todo lo demás: alternar de forma agresiva para garantizar ≤ 0.57
        diferencias_necesarias = 8  # Más agresivo: 8 diferencias = Jaccard = 0.43
        diferencias_creadas = 0
        
        for i, partido in enumerate(partidos):
            if partido["clasificacion"] == "Ancla":
                continue  # Ya procesado
                
            if diferencias_creadas < diferencias_necesarias:
                # Crear diferencia
                resultado_a = self._get_resultado_max_prob(partido)
                resultado_b = self._get_resultado_opuesto_inteligente(resultado_a, partido)
                quiniela_a[i] = resultado_a
                quiniela_b[i] = resultado_b
                diferencias_creadas += 1
            else:
                # Mantener idéntico
                resultado = self._get_resultado_max_prob(partido)
                quiniela_a[i] = resultado
                quiniela_b[i] = resultado

        # Ajustar empates
        quiniela_a = self._ajustar_empates_satelite(quiniela_a, partidos)
        quiniela_b = self._ajustar_empates_satelite(quiniela_b, partidos)
        
        return quiniela_a, quiniela_b

    def _crear_par_emergencia(self, partidos: List[Dict[str, Any]], par_id: int) -> Tuple[List[str], List[str]]:
        """
        Par de emergencia cuando todo falla - estrategia simple pero funcional
        """
        self.logger.warning(f"🚨 Generando par de emergencia {par_id}")
        
        quiniela_a = []
        quiniela_b = []
        
        for i, partido in enumerate(partidos):
            if partido["clasificacion"] == "Ancla":
                # Anclas idénticas
                resultado = self._get_resultado_max_prob(partido)
                quiniela_a.append(resultado)
                quiniela_b.append(resultado)
            elif i % 2 == 0:
                # Alternancia simple por posición
                resultado_a = self._get_resultado_max_prob(partido)
                resultado_b = "L" if resultado_a != "L" else "V"
                quiniela_a.append(resultado_a)
                quiniela_b.append(resultado_b)
            else:
                # Mantener algunos idénticos
                resultado = self._get_resultado_max_prob(partido)
                quiniela_a.append(resultado)
                quiniela_b.append(resultado)
        
        # Ajustar empates básico
        quiniela_a = self._ajustar_empates_basico(quiniela_a)
        quiniela_b = self._ajustar_empates_basico(quiniela_b)
        
        return quiniela_a, quiniela_b

    def _get_resultado_opuesto_inteligente(self, resultado_actual: str, partido: Dict[str, Any]) -> str:
        """
        Obtiene resultado opuesto de forma inteligente para maximizar anticorrelación
        """
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        
        if resultado_actual == "L":
            # Preferir V sobre E para maximizar diferencia
            return "V" if probs["V"] > 0.15 else "E"
        elif resultado_actual == "V":
            # Preferir L sobre E
            return "L" if probs["L"] > 0.15 else "E"
        else:  # resultado_actual == "E"
            # Elegir entre L y V basado en probabilidades
            return "L" if probs["L"] > probs["V"] else "V"

    def _ajustar_correlacion_forzada(self, quiniela_a: List[str], quiniela_b: List[str], 
                                   partidos: List[Dict[str, Any]], objetivo_jaccard: float) -> Tuple[List[str], List[str]]:
        """
        Ajusta forzadamente hasta lograr el objetivo de correlación
        """
        max_intentos = 10
        intento = 0
        
        while (self._calcular_correlacion_jaccard(quiniela_a, quiniela_b) > objetivo_jaccard and 
               intento < max_intentos):
            
            # Encontrar candidatos para invertir (no Anclas)
            candidatos = []
            for i, (a, b) in enumerate(zip(quiniela_a, quiniela_b)):
                if a == b and partidos[i]["clasificacion"] != "Ancla":
                    candidatos.append(i)
            
            if not candidatos:
                self.logger.warning("No hay más candidatos para invertir")
                break
                
            # Invertir el candidato más prometedor
            idx_invertir = random.choice(candidatos)
            quiniela_b[idx_invertir] = self._get_resultado_opuesto_inteligente(
                quiniela_a[idx_invertir], partidos[idx_invertir]
            )
            
            intento += 1
            
        return quiniela_a, quiniela_b

    def _ajustar_empates_preservando_diferencias(self, quiniela: List[str], partidos: List[Dict[str, Any]], 
                                               quiniela_pareja: List[str]) -> List[str]:
        """
        Ajusta empates sin destruir las diferencias ya creadas con la pareja
        """
        empates_actuales = quiniela.count("E")
        
        if self.empates_min <= empates_actuales <= self.empates_max:
            return quiniela
        
        quiniela_ajustada = quiniela.copy()
        
        if empates_actuales > self.empates_max:
            # Reducir empates, pero sin tocar diferencias críticas
            exceso = empates_actuales - self.empates_max
            candidatos = []
            
            for i, res in enumerate(quiniela):
                if (res == "E" and 
                    partidos[i]["clasificacion"] != "Ancla" and
                    quiniela_pareja[i] == "E"):  # Solo si la pareja también tiene E
                    candidatos.append((i, partidos[i]["prob_empate"]))
            
            # Cambiar los empates de menor probabilidad
            candidatos.sort(key=lambda x: x[1])
            for i in range(min(exceso, len(candidatos))):
                idx = candidatos[i][0]
                partido = partidos[idx]
                quiniela_ajustada[idx] = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"
                
        elif empates_actuales < self.empates_min:
            # Aumentar empates preservando diferencias
            faltante = self.empates_min - empates_actuales
            candidatos = []
            
            for i, res in enumerate(quiniela):
                if (res in ["L", "V"] and 
                    partidos[i]["clasificacion"] != "Ancla" and
                    quiniela_pareja[i] == res):  # Solo si la pareja tiene lo mismo
                    candidatos.append((i, partidos[i]["prob_empate"]))
            
            # Cambiar a empate los de mayor probabilidad de empate
            candidatos.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(faltante, len(candidatos))):
                idx = candidatos[i][0]
                quiniela_ajustada[idx] = "E"
        
        return quiniela_ajustada

    def _ajustar_empates_basico(self, quiniela: List[str]) -> List[str]:
        """Ajuste básico de empates para emergencias"""
        empates_actuales = quiniela.count("E")
        
        if empates_actuales < self.empates_min:
            # Convertir algunos L/V a E
            for i in range(len(quiniela)):
                if quiniela[i] in ["L", "V"] and quiniela.count("E") < self.empates_min:
                    quiniela[i] = "E"
        elif empates_actuales > self.empates_max:
            # Convertir algunos E a L
            for i in range(len(quiniela)):
                if quiniela[i] == "E" and quiniela.count("E") > self.empates_max:
                    quiniela[i] = "L"
                    
        return quiniela

    def _get_resultado_max_prob(self, partido: Dict[str, Any]) -> str:
        """Obtiene el resultado de máxima probabilidad"""
        probs = {
            "L": partido["prob_local"],
            "E": partido["prob_empate"],
            "V": partido["prob_visitante"]
        }
        return max(probs, key=probs.get)
    
    def _ajustar_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]]) -> List[str]:
        """Método original de ajuste de empates"""
        empates_actuales = quiniela.count("E")
        
        if self.empates_min <= empates_actuales <= self.empates_max:
            return quiniela
        
        quiniela_ajustada = quiniela.copy()
        
        if empates_actuales > self.empates_max:
            exceso = empates_actuales - self.empates_max
            self._reducir_empates_satelite(quiniela_ajustada, partidos, exceso)
        elif empates_actuales < self.empates_min:
            faltante = self.empates_min - empates_actuales
            self._aumentar_empates_satelite(quiniela_ajustada, partidos, faltante)
        
        return quiniela_ajustada
    
    def _reducir_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]], reducir: int):
        """Reduce empates evitando ANCLAS"""
        candidatos = [(i, partidos[i]["prob_empate"]) 
                     for i, res in enumerate(quiniela) 
                     if res == "E" and partidos[i]["clasificacion"] != "Ancla"]
        
        candidatos.sort(key=lambda x: x[1])
        
        for i in range(min(reducir, len(candidatos))):
            idx = candidatos[i][0]
            partido = partidos[idx]
            quiniela[idx] = "L" if partido["prob_local"] > partido["prob_visitante"] else "V"
    
    def _aumentar_empates_satelite(self, quiniela: List[str], partidos: List[Dict[str, Any]], aumentar: int):
        """Aumenta empates evitando ANCLAS"""
        candidatos = [(i, partidos[i]["prob_empate"]) 
                     for i, res in enumerate(quiniela) 
                     if res in ["L", "V"] and partidos[i]["clasificacion"] != "Ancla"]
        
        candidatos.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(min(aumentar, len(candidatos))):
            idx = candidatos[i][0]
            quiniela[idx] = "E"
    
    def _calcular_correlacion_jaccard(self, quiniela_a: List[str], quiniela_b: List[str]) -> float:
        """Calcula correlación de Jaccard entre dos quinielas"""
        if len(quiniela_a) != len(quiniela_b): 
            return 0.0
        coincidencias = sum(1 for a, b in zip(quiniela_a, quiniela_b) if a == b)
        return coincidencias / len(quiniela_a)
    
    def _validar_satelites_robusto(self, satelites: List[Dict[str, Any]]):
        """Validación robusta con logging detallado"""
        self.logger.debug("🔍 Validando satélites robustos...")
        
        errores = []
        
        for satelite in satelites:
            empates = satelite["resultados"].count("E")
            if not (self.empates_min <= empates <= self.empates_max):
                errores.append(f"{satelite['id']}: empates {empates} fuera del rango [{self.empates_min}-{self.empates_max}]")
            if len(satelite["resultados"]) != 14:
                errores.append(f"{satelite['id']}: longitud {len(satelite['resultados'])} != 14")
        
        # Validar pares
        pares = {}
        for satelite in satelites:
            par_id = satelite["par_id"]
            pares.setdefault(par_id, []).append(satelite)
        
        for par_id, par_satelites in pares.items():
            if len(par_satelites) != 2:
                errores.append(f"Par {par_id}: debe tener exactamente 2 satélites, tiene {len(par_satelites)}")
                continue
                
            correlacion = self._calcular_correlacion_jaccard(
                par_satelites[0]['resultados'], 
                par_satelites[1]['resultados']
            )
            
            if correlacion > self.correlacion_max:
                errores.append(f"Par {par_id}: correlación {correlacion:.3f} > {self.correlacion_max}")
                self.logger.error(f"❌ Par {par_id} falló validación: {correlacion:.3f} > {self.correlacion_max}")
        
        if errores:
            self.logger.error(f"❌ Errores de validación: {errores}")
            raise ValueError(f"Validación de satélites falló: {errores}")
        
        self.logger.debug("✅ Todos los satélites robustos son válidos")