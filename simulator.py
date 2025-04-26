# ==========================================================
# simulator.py  (compatible con Python 3.12)
# ==========================================================
from __future__ import annotations
import numpy as np, pandas as pd
from probabilities import PROBABILITIES

_OUT = np.array(["L","E","V"], dtype=str)


def _sample(n:int)->np.ndarray:
    prob=np.array([PROBABILITIES[i] for i in range(21)])
    cum=prob.cumsum(1)
    rnd=np.random.rand(n,21,1)
    idx=(rnd<cum).argmax(2)
    return _OUT[idx]


def simulate(df:pd.DataFrame, n:int=10000) -> tuple[float, float, dict]:
    data=df.values.astype(str).T             # 20×21
    act=_sample(n)                           # n×21
    hits=data[:,None,:]==act[None,:,:]
    dbl=np.char.find(data,"/")>=0
    if dbl.any():
        alt=np.zeros_like(hits)
        for q,row in enumerate(data):
            for m,pred in enumerate(row):
                if "/" in pred:
                    for opt in pred.split("/"):
                        alt[q,:,m]|=(opt==act[:,m])
        hits=np.where(dbl[:,None,:],alt,hits)
    sc=hits.sum(2)
    
    # Calcular probabilidades generales
    p_reg = (sc[:,:14].max(0)>=11).mean()
    p_rev = (sc[:,14:].max(0)>=6).mean()
    
    # Calcular detalle por quiniela
    detail = {}
    column_names = [f"Q{i+1}" for i in range(20)]
    
    # Para cada quiniela (columna), calcular su efectividad
    for q in range(20):
        # Número promedio de aciertos por simulación para esta quiniela
        avg_hits_reg = sc[q,:14].mean() if q < sc.shape[0] and sc.shape[1] > 14 else 0
        avg_hits_rev = sc[q,14:].mean() if q < sc.shape[0] and sc.shape[1] > 14 else 0
        
        # Probabilidad de ganar por quiniela
        prob_win_reg = (sc[q,:14] >= 11).mean() if q < sc.shape[0] and sc.shape[1] > 14 else 0
        prob_win_rev = (sc[q,14:] >= 6).mean() if q < sc.shape[0] and sc.shape[1] > 14 else 0
        
        # Guardar los datos
        detail[column_names[q]] = {
            "avg_hits_reg": float(avg_hits_reg),
            "avg_hits_rev": float(avg_hits_rev),
            "prob_win_reg": float(prob_win_reg),
            "prob_win_rev": float(prob_win_rev)
        }
    
    return p_reg, p_rev, detail