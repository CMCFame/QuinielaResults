# ===========================================================
# simulator.py
# ===========================================================
from __future__ import annotations
import numpy as np, pandas as pd
from probabilities import PROBABILITIES

_OUTCOMES = np.array(["L", "E", "V"])

def _sample_actual(n_iter: int = 10_000) -> np.ndarray:
    """Generate matrix [n_iter × 21] of simulated outcomes."""
    prob_arr = np.array([PROBABILITIES[i] for i in range(21)])  # 21×3
    cum = prob_arr.cumsum(axis=1)
    rnd = np.random.rand(n_iter, 21, 1)
    draws = (rnd < cum).argmax(axis=2)
    return _OUTCOMES[draws]  # n_iter × 21

def simulate(df: pd.DataFrame, n_iter: int = 10_000) -> tuple[float, float]:
    """Return (prob_regular, prob_revancha) for ≥11 and ≥6 aciertos."""
    data = df.values  # rows = matches, cols = quinielas
    data = data.T      # quinielas × 21
    actuals = _sample_actual(n_iter)  # n_iter × 21
    # Broadcast compare
    hits = (data[:, None, :] == actuals[None, :, :])  # q × n × 21
    # Support doubles (e.g. "L/E")
    doubles_mask = np.vectorize(lambda s: "/" in s)(data)
    if doubles_mask.any():
        alt_hits = np.zeros_like(hits)
        for qi, row in enumerate(data):
            for mi, pred in enumerate(row):
                if "/" in pred:
                    for option in pred.split("/"):
                        alt_hits[qi, :, mi] |= (option == actuals[:, mi])
        hits = np.where(doubles_mask[:, None, :], alt_hits, hits)
    scores = hits.sum(axis=2)  # q × n
    best_reg = scores[:, :, :14].max(axis=0)
    best_rev = scores[:, :, 14:].max(axis=0)
    p_reg = (best_reg >= 11).mean()
    p_rev = (best_rev >= 6).mean()
    return p_reg, p_rev