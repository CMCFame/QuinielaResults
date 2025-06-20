# -*- coding: utf-8 -*-
# progol_optimizer/portfolio/optimizer.py 每?CORRECCI車N DEFINITIVA
"""
Optimizador GRASP?Annealing 每?versi車n final
------------------------------------------------
? B迆squeda GRASP menos restrictiva durante la exploraci車n.
? Enfriamiento simulado agresivo en la fase de ajuste final.
? Correcciones de codificaci車n: acentos y emojis restaurados para evitar errores UTF?8.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# CLASE PRINCIPAL
# ---------------------------------------------------------------------------


class GRASPAnnealing:
    """Optimizador h赤brido GRASP?+ Enfriamiento Simulado."""

    # ---------------------------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------------------------

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Configuraci車n global (inyectada desde config.constants)
        from config.constants import PROGOL_CONFIG

        self.config = PROGOL_CONFIG
        self.opt_cfg = self.config["OPTIMIZACION"]

        self.max_iter = self.opt_cfg["max_iteraciones"]
        self.temp_inicial = self.opt_cfg["temperatura_inicial"]
        self.factor_enfriamiento = self.opt_cfg["factor_enfriamiento"]
        self.alpha_grasp = self.opt_cfg["alpha_grasp"]
        self.max_sin_mejora = self.opt_cfg["iteraciones_sin_mejora"]

        # Cach谷 para acelerar evaluaciones repetidas
        self._cache: Dict[int, float] = {}
        self._hits = 0
        self._miss = 0

        # Validador externo
        from validation.portfolio_validator import PortfolioValidator

        self.validator = PortfolioValidator()

        # (Opcional) Asistente IA experimental
        try:
            from models.ai_assistant import ProgolAIAssistant

            self.ai_assistant = ProgolAIAssistant()
        except Exception:  # pragma: no cover 每 may no existir en prod
            self.ai_assistant = None
            self.logger.debug("Asistente IA no disponible; se continuar芍 sin 谷l.")

    # ---------------------------------------------------------------------
    # M谷TODOS UTILITARIOS
    # ---------------------------------------------------------------------

    @staticmethod
    def _resultado_a_clave(resultado: str) -> str:
        """Convierte s赤mbolo (L/E/V) en nombre de campo de probabilidad."""

        return {"L": "local", "E": "empate", "V": "visitante"}.get(resultado, "local")

    @staticmethod
    def _crear_cache_key(portafolio: List[Dict[str, Any]]) -> int:
        """Clave hashable para el portafolio completo."""

        return hash(tuple("".join(q["resultados"]) for q in portafolio))

    # ---------------------------------------------------------------------
    # L車GICA DE OPTIMIZACI車N
    # ---------------------------------------------------------------------

    def optimizar_portafolio_grasp_annealing(
        self,
        quinielas_iniciales: List[Dict[str, Any]],
        partidos: List[Dict[str, Any]],
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Punto de entrada principal al optimizador."""

        self.logger.info("?? Iniciando optimizaci車n GRASP?Annealing definitiva＃")
        self._precalcular_prob_matriz(partidos)

        # Estado inicial
        port_actual = [q.copy() for q in quinielas_iniciales]
        score_actual = self._objetivo_f(port_actual, partidos)

        mejor_port = port_actual
        mejor_score = score_actual

        temp = self.temp_inicial
        sin_mejora = 0
        self.logger.info("Score inicial: F=%.6f", score_actual)

        for it in range(self.max_iter):
            nuevo_port = self._movimiento_grasp(port_actual, partidos)
            if nuevo_port is None:
                continue

            nuevo_score = self._objetivo_f(nuevo_port, partidos)
            delta = nuevo_score - score_actual

            # Regla de aceptaci車n
            acepta = delta > 0 or (
                temp > 0 and random.random() < math.exp(delta / temp)
            )
            if acepta:
                port_actual = nuevo_port
                score_actual = nuevo_score

                if nuevo_score > mejor_score:
                    mejor_port = port_actual
                    mejor_score = nuevo_score
                    sin_mejora = 0
                    self.logger.debug("Iter %d: nuevo mejor ↙ %.6f", it, mejor_score)
                else:
                    sin_mejora += 1
            else:
                sin_mejora += 1

            # Callback de progreso (opcional)
            if progress_callback and it % 10 == 0:
                progress_callback(it / self.max_iter, f"Iter {it}/{self.max_iter} | F={mejor_score:.5f}")

            # Enfriar
            temp *= self.factor_enfriamiento

            # Parada temprana
            if sin_mejora >= self.max_sin_mejora:
                self.logger.info("?? Parada temprana en iter %d (sin mejora)", it)
                break

        # Ajuste final exhaustivo
        self.logger.info("?? Ajuste final agresivo de concentraci車n y distribuci車n＃")
        mejor_port = self._ajuste_final(mejor_port, partidos)
        score_final = self._objetivo_f(mejor_port, partidos)
        self.logger.info("? Optimizaci車n completada: F=%.6f", score_final)
        return mejor_port

    # ------------------------------------------------------------------
    # GENERACI車N DE MOVIMIENTOS (GRASP)
    # ------------------------------------------------------------------

    def _movimiento_grasp(
        self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]] | None:
        """Crea un vecino v芍lido con criterios permisivos."""

        for _ in range(50):  # m芍ximo intentos
            nuevo = [q.copy() for q in portafolio]
            idx_q = random.randrange(len(nuevo))

            # Prioridad a sat谷lites
            if random.random() < 0.8:
                sat谷lites = [i for i, q in enumerate(nuevo) if q["tipo"] == "Satelite"]
                if sat谷lites:
                    idx_q = random.choice(sat谷lites)

            q_orig = nuevo[idx_q]
            res = q_orig["resultados"].copy()

            modificables = [i for i, p in enumerate(partidos) if p.get("clasificacion") != "Ancla"]
            if not modificables:
                continue

            n_cambios = random.choice([1, 1, 2])
            for i_c in random.sample(modificables, n_cambios):
                opciones = ["L", "E", "V"]
                opciones.remove(res[i_c])
                res[i_c] = random.choice(opciones)

            if self._valido_permisivo(portafolio, idx_q, res):
                q_mod = q_orig.copy()
                q_mod["resultados"] = res
                q_mod["empates"] = res.count("E")
                q_mod["distribuci車n"] = {
                    "L": res.count("L"),
                    "E": res.count("E"),
                    "V": res.count("V"),
                }
                nuevo[idx_q] = q_mod
                return nuevo
        return None

    def _valido_permisivo(
        self, portafolio: List[Dict[str, Any]], idx_q: int, nuevos: List[str]
    ) -> bool:
        """Criterios menos estrictos durante la exploraci車n."""

        # 1) Rango de empates
        emp = nuevos.count("E")
        if not (self.config["EMPATES_MIN"] <= emp <= self.config["EMPATES_MAX"]):
            return False

        # 2) Concentraci車n global ≒?85?%
        if max(nuevos.count(s) for s in "LEV") / 14.0 > 0.85:
            return False

        # 3) Primeros tres partidos ≒?85?%
        if max(nuevos[:3].count(s) for s in "LEV") / 3.0 > 0.85:
            return False

        # 4) Correlaci車n Jaccard para sat谷lites
        q = portafolio[idx_q]
        if q["tipo"] == "Satelite":
            pid = q.get("par_id")
            par = next(
                (p for i, p in enumerate(portafolio) if p.get("par_id") == pid and i != idx_q),
                None,
            )
            if par:
                jac = sum(a == b for a, b in zip(nuevos, par["resultados"])) / 14.0
                if jac > self.config["ARQUITECTURA_PORTAFOLIO"]["correlacion_jaccard_max"]:
                    return False
        return True

    # ------------------------------------------------------------------
    # AJUSTE FINAL AGRESIVO
    # ------------------------------------------------------------------

    def _ajuste_final(
        self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ajuste exhaustivo para cumplir TODAS las validaciones."""

        port = [q.copy() for q in portafolio]

        # Paso?1: concentraci車n individual ≒?70?% global y ≒?60?% primeros?3
        for _ in range(3):
            cambios = 0
            for i, q in enumerate(port):
                if q["tipo"] == "Core":
                    continue
                res_ok = self._forzar_concentracion(q["resultados"], partidos)
                if res_ok != q["resultados"]:
                    q["resultados"] = res_ok
                    q["empates"] = res_ok.count("E")
                    q["distribuci車n"] = {
                        "L": res_ok.count("L"),
                        "E": res_ok.count("E"),
                        "V": res_ok.count("V"),
                    }
                    cambios += 1
            if cambios == 0:
                break

        # Paso?2: distribuci車n por posici車n (balancing)
        for _ in range(5):
            cambios = 0
            for pos in range(14):
                cambios += self._balancear_posicion(port, pos, partidos)
            if cambios == 0:
                break

        # Paso?3: verificaci車n global
        conc_ok = self.validator._validar_concentracion_70_60(port)
        dist_ok = self.validator._validar_distribucion_equilibrada(port)
        self.logger.info("?? Resultado final?每 concentraci車n=%s, distribuci車n=%s", conc_ok, dist_ok)
        return port

    # --------------------------------------------------------------
    # RUTINAS DE CORRECCI車N DE CONCENTRACI車N
    # --------------------------------------------------------------

    def _forzar_concentracion(
        self, resultados: List[str], partidos: List[Dict[str, Any]]
    ) -> List[str]:
        """Corrige una quiniela para no superar 70?% global / 60?% inicial."""

        res = resultados.copy()
        anclas = {i for i, p in enumerate(partidos) if p.get("clasificacion") == "Ancla"}
        libres = [i for i in range(14) if i not in anclas]

        # ---------- Global ≒?70?% ----------
        max_global = int(14 * 0.70)  # 9
        for signo in "LEV":
            while res.count(signo) > max_global:
                cand = [i for i in libres if res[i] == signo]
                if not cand:
                    break
                cand.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                i_idx = cand[0]
                dest = min(
                    (s for s in "LEV" if s != signo),
                    key=lambda s: res.count(s),
                )
                res[i_idx] = dest

        # ---------- Primeros tres ≒?60?% ----------
        max_ini = int(3 * 0.60)  # 1
        for signo in "LEV":
            while res[:3].count(signo) > max_ini:
                idxs = [i for i in range(3) if res[i] == signo and i not in anclas]
                if not idxs:
                    break
                idxs.sort(key=lambda i: partidos[i][f"prob_{self._resultado_a_clave(signo)}"])
                i_idx = idxs[0]
                dest = min(
                    (s for s in "LEV" if s != signo),
                    key=lambda s: res[:3].count(s),
                )
                res[i_idx] = dest
        return res

    # --------------------------------------------------------------
    # BALANCEO POR POSICI車N
    # --------------------------------------------------------------

    def _balancear_posicion(
        self, port: List[Dict[str, Any]], pos: int, partidos: List[Dict[str, Any]]
    ) -> int:
        """Equilibra la posici車n *pos* en todas las quinielas sat谷lite."""

        if partidos[pos].get("clasificacion") == "Ancla":
            return 0

        total = len(port)
        max_ap = int(total * 0.67)
        min_ap = int(total * 0.10)

        conteo = {s: 0 for s in "LEV"}
        idx_por_s = {s: [] for s in "LEV"}

        for i, q in enumerate(port):
            if q["tipo"] != "Satelite":
                continue
            signo = q["resultados"][pos]
            conteo[signo] += 1
            idx_por_s[signo].append(i)

        cambios = 0

        # Exceso
        for signo in "LEV":
            while conteo[signo] > max_ap:
                destinos = [s for s in "LEV" if conteo[s] < max_ap]
                if not destinos:
                    break
                idxs = idx_por_s[signo]
                idxs.sort(
                    key=lambda i: partidos[pos][f"prob_{self._resultado_a_clave(signo)}"]
                )
                q_idx = idxs.pop(0)
                dest = min(destinos, key=lambda s: conteo[s])
                port[q_idx]["resultados"][pos] = dest
                conteo[signo] -= 1
                conteo[dest] += 1
                cambios += 1

        # Defecto
        for signo in "LEV":
            while conteo[signo] < min_ap:
                fuente_opts = [s for s in "LEV" if conteo[s] > min_ap]
                if not fuente_opts:
                    break
                fuente = max(fuente_opts, key=lambda s: conteo[s])
                idxs = idx_por_s[fuente]
                if not idxs:
                    break
                idxs.sort(
                    key=lambda i: partidos[pos][f"prob_{self._resultado_a_clave(fuente)}"],
                    reverse=True,
                )
                q_idx = idxs.pop(0)
                port[q_idx]["resultados"][pos] = signo
                conteo[fuente] -= 1
                conteo[signo] += 1
                cambios += 1

        # Actualizar distribuciones
        if cambios:
            mod_idxs = {idx for lst in idx_por_s.values() for idx in lst}
            for idx in mod_idxs:
                q = port[idx]
                res = q["resultados"]
                q["empates"] = res.count("E")
                q["distribuci車n"] = {s: res.count(s) for s in "LEV"}

        return cambios

    # --------------------------------------------------------------
    # EVALUACI車N DE OBJETIVO
    # --------------------------------------------------------------

    def _precalcular_prob_matriz(self, partidos: List[Dict[str, Any]]) -> None:
        self._prob_mat = np.zeros((14, 3))
        for i, p in enumerate(partidos):
            self._prob_mat[i] = [p["prob_local"], p["prob_empate"], p["prob_visitante"]]

    def _objetivo_f(
        self, portafolio: List[Dict[str, Any]], partidos: List[Dict[str, Any]]
    ) -> float:
        key = self._crear_cache_key(portafolio)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._miss += 1

        prod = 1.0
        for q in portafolio:
            prob_11p = self._prob_11_montecarlo(q["resultados"], partidos)
            prod *= 1 - prob_11p
        f_val = 1 - prod

        self._cache[key] = f_val
        if len(self._cache) > 2000:
            self._cache.clear()
        return f_val

    def _prob_11_montecarlo(
        self, resultados: List[str], partidos: List[Dict[str, Any]]
    ) -> float:
        """Estimaci車n r芍pida (1000?simulaciones) de ≡11 aciertos."""

        n_sim = 1000
        prob_acierto = np.array(
            [partidos[i][f"prob_{self._resultado_a_clave(r)}"] for i, r in enumerate(resultados)]
        )
        sim = np.random.rand(n_sim, 14)
        aciertos = (sim < prob_acierto).sum(axis=1)
        return (aciertos >= 11).mean()
