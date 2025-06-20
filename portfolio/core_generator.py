# -*- coding: utf-8 -*-
"""
progol_optimizer/portfolio/core_generator.py
Generador de Quinielas Core – Versión 2025-06-20 b
•  Garantiza 4 quinielas distintas con ≥ 2 diferencias entre sí
•  Cumple rango de empates y distribución L/E/V definidos en PROGOL_CONFIG
•  Registra en DEBUG todos los cambios aplicados
"""

import logging
import random
import copy
from typing import List, Dict, Any, Tuple

class CoreGenerator:
    MAX_REINTENTOS = 5           # para evitar loops infinitos
    DIFERENCIAS_MINIMAS = 2      # entre cada par de Core

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from config.constants import PROGOL_CONFIG          # import diferido
        self.config = PROGOL_CONFIG
        self.empates_min = self.config["EMPATES_MIN"]
        self.empates_max = self.config["EMPATES_MAX"]
        self.rangos_hist = self.config["RANGOS_HISTORICOS"]   # {'L':(min,max),...}

    # ---------- API pública ----------
    def generar_quinielas_core(self,
                               partidos: List[Dict[str, Any]]
                               ) -> List[Dict[str, Any]]:
        """
        Genera 4 quinielas Core distintas que respetan:
        • Rango de empates
        • Distribución L/E/V histórica
        • ≥ DIFERENCIAS_MINIMAS entre cada par
        """
        anclas = [i for i,p in enumerate(partidos) if p.get("clasificacion") == "Ancla"]
        self.logger.info("Anclas en posiciones %s", [i+1 for i in anclas])

        base = self._generar_quiniela_base(partidos, anclas)
        cores: List[Dict[str,Any]] = []

        for core_idx in range(4):
            for intento in range(self.MAX_REINTENTOS):
                if core_idx == 0 and intento == 0:
                    q = base.copy()
                else:
                    q = self._variar_quiniela(base,
                                              partidos,
                                              anclas,
                                              core_idx,
                                              intento)
                q = self._normalizar_empates(q, partidos, anclas)
                if self._valida_quiniela(q):
                    cores.append(self._info(core_idx, q))
                    break
            else:
                raise RuntimeError(f"No pude construir Core {core_idx+1} tras "
                                   f"{self.MAX_REINTENTOS} intentos")
        self._verificar_diferencias(cores)
        return cores

    # ---------- Generación base ----------
    def _generar_quiniela_base(self,
                               partidos: List[Dict[str,Any]],
                               anclas: List[int]) -> List[str]:
        q = [""]*14
        for idx in range(14):
            proba = self._ordenar_prob(partidos[idx])
            if idx in anclas:
                q[idx] = proba[0][0]                     # siempre la máxima
            else:
                for res,_ in proba:
                    tent = q.copy()
                    tent[idx] = res
                    if self._respeta_rangos(tent):
                        q[idx] = res
                        break
        return self._normalizar_empates(q, partidos, anclas)

    # ---------- Variación mínima pero garantizada ----------
    def _variar_quiniela(self,
                         base: List[str],
                         partidos: List[Dict[str,Any]],
                         anclas: List[int],
                         core_idx: int,
                         intento: int) -> List[str]:
        q = base.copy()
        modificables = [i for i in range(14) if i not in anclas]
        mods_orden = sorted(modificables,
                            key=lambda i: self._ordenar_prob(partidos[i])[0][1] -
                                          self._ordenar_prob(partidos[i])[1][1])
        random.shuffle(mods_orden)
        cambios_realizados = 0
        objetivo = min(3, core_idx + 1)   # 1,2,3,3 cambios para Core 1-4

        for pos in mods_orden:
            mejor, segundo = self._ordenar_prob(partidos[pos])[:2]
            objetivo_res = segundo if segundo[1] >= 0.8*mejor[1] else mejor
            if q[pos] != objetivo_res[0]:
                q[pos] = objetivo_res[0]
                cambios_realizados += 1
            if cambios_realizados >= objetivo:
                break
        self.logger.debug("Core %d intento %d → %s",
                          core_idx+1, intento+1, "".join(q))
        return q

    # ---------- Ajustes y verificaciones ----------
    def _normalizar_empates(self,
                            q: List[str],
                            partidos: List[Dict[str,Any]],
                            anclas: List[int]) -> List[str]:
        q = q.copy()
        empates = q.count("E")
        if empates < self.empates_min:
            candidatos = [i for i in range(14)
                          if i not in anclas and q[i] != "E"]
            candidatos.sort(key=lambda i: partidos[i]["prob_empate"], reverse=True)
            for idx in candidatos:
                q[idx] = "E"
                empates += 1
                if empates >= self.empates_min:
                    break
        elif empates > self.empates_max:
            candidatos = [i for i in range(14)
                          if i not in anclas and q[i] == "E"]
            candidatos.sort(key=lambda i: partidos[i]["prob_empate"])
            for idx in candidatos:
                q[idx] = self._ordenar_prob(partidos[idx])[0][0]
                empates -= 1
                if empates <= self.empates_max:
                    break
        return q

    def _respeta_rangos(self, q: List[str]) -> bool:
        if "" in q:
            return True
        for res in ("L","E","V"):
            minimo, maximo = self.rangos_hist[res]
            if not (minimo <= q.count(res) <= maximo):
                return False
        return True

    def _valida_quiniela(self, q: List[str]) -> bool:
        return (self.empates_min <= q.count("E") <= self.empates_max
                and self._respeta_rangos(q))

    def _verificar_diferencias(self, cores: List[Dict[str,Any]]):
        for i in range(len(cores)):
            for j in range(i+1, len(cores)):
                diff = sum(1 for a,b in zip(cores[i]["resultados"],
                                            cores[j]["resultados"]) if a!=b)
                if diff < self.DIFERENCIAS_MINIMAS:
                    raise ValueError(f"{cores[i]['id']} y {cores[j]['id']} "
                                     f"tienen sólo {diff} diferencias")

    # ---------- Utilidades ----------
    @staticmethod
    def _ordenar_prob(partido: Dict[str,Any]) -> List[Tuple[str,float]]:
        return sorted([("L", partido["prob_local"]),
                       ("E", partido["prob_empate"]),
                       ("V", partido["prob_visitante"])],
                      key=lambda t: t[1],
                      reverse=True)

    @staticmethod
    def _info(idx: int, q: List[str]) -> Dict[str,Any]:
        return {
            "id": f"Core-{idx+1}",
            "tipo": "Core",
            "resultados": q,
            "empates": q.count("E"),
            "distribucion": {
                "L": q.count("L"),
                "E": q.count("E"),
                "V": q.count("V")
            }
        }
