from __future__ import annotations
import numpy as np
import pandas as pd
from probabilities import PROBABILITIES

_OUT = np.array(["L", "E", "V"], dtype="<U1")


def _sample_outcomes(n_iter: int) -> np.ndarray:
    """Devuelve matriz  (n_iter × 21)  de resultados simulados."""
    prob = np.array([PROBABILITIES[i] for i in range(21)])         # 21×3
    cum  = prob.cumsum(axis=1)                                     # CDF
    rnd  = np.random.rand(n_iter, 21, 1)
    idx  = (rnd < cum).argmax(axis=2)                              # 0-1-2
    return _OUT[idx]                                               # n_iter×21


def simulate(grid: pd.DataFrame, n_iter: int = 10_000,
             *, debug: bool = False) -> tuple[float, float]:
    """Monte-Carlo → (P≥11, P≥6).  Acepta dobles «L/E»."""
    data = grid.values.astype(str).T                               # 20×21
    actual = _sample_outcomes(n_iter)                              # n_iter×21

    # matriz aciertos  (20 × n_iter × 21)
    hits = data[:, None, :] == actual[None, :, :]

    # manejar dobles
    doubles = np.char.find(data, "/") >= 0
    if doubles.any():
        alt = np.zeros_like(hits)
        for q, row in enumerate(data):
            for m, pred in enumerate(row):
                if "/" in pred:
                    for opt in pred.split("/"):
                        alt[q, :, m] |= (opt == actual[:, m])
        hits = np.where(doubles[:, None, :], alt, hits)

    scores = hits.sum(axis=2)                       # 20 × n_iter
    best_regular  = scores[:, :, :14].max(axis=0)
    best_revancha = scores[:, :, 14:].max(axis=0)

    if debug:
        print("Shapes -> data:", data.shape, "actual:", actual.shape, "hits:", hits.shape)

    return (best_regular >= 11).mean(), (best_revancha >= 6).mean()
