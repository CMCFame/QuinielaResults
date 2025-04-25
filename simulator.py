# ===========================================================
# simulator.py
# ===========================================================
from __future__ import annotations
import numpy as np, pandas as pd
from probabilities import PROBABILITIES

_OUTCOMES = np.array(["L", "E", "V"])


def _sample_actual(n: int) -> np.ndarray:
    """Return an *n × 21* array of simulated outcomes."""
    prob = np.array([PROBABILITIES[i] for i in range(21)])  # 21×3
    cum  = prob.cumsum(1)
    rnd  = np.random.rand(n, 21, 1)
    idx  = (rnd < cum).argmax(2)
    return _OUTCOMES[idx]


def simulate(grid: pd.DataFrame, n_iter: int = 10_000) -> tuple[float, float]:
    data = grid.values.T  # 20×21  → quinielas × partidos
    actual = _sample_actual(n_iter)  # n_iter×21

    # matriz aciertos: quiniela × n_iter × partido
    hits = (data[:, None, :] == actual[None, :, :])

    # soportar dobles ― ej.: "L/E"
    doubles = np.vectorize(lambda s: "/" in s)(data)
    if doubles.any():
        alt = np.zeros_like(hits)
        for q, row in enumerate(data):
            for m, pred in enumerate(row):
                if "/" in pred:
                    for opt in pred.split("/"):
                        alt[q, :, m] |= (opt == actual[:, m])
        hits = np.where(doubles[:, None, :], alt, hits)

    scores = hits.sum(2)                # quiniela × n_iter
    best_reg = scores[:, :, :14].max(0)  # mejores 14
    best_rev = scores[:, :, 14:].max(0)  # mejores 7
    return (best_reg >= 11).mean(), (best_rev >= 6).mean()
