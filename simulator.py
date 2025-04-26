# ==========================================================
# simulator.py  (Monte‑Carlo core)
# ==========================================================
from __future__ import annotations
import numpy as np, pandas as pd
from probabilities import PROBABILITIES

_OUT = np.array(["L","E","V"],"<U1")


def _sample(n:int)->np.ndarray:
    prob=np.array([PROBABILITIES[i] for i in range(21)])
    cum=prob.cumsum(1)
    rnd=np.random.rand(n,21,1)
    idx=(rnd<cum).argmax(2)
    return _OUT[idx]


def simulate(df:pd.DataFrame,n:int=10000)->tuple[float,float]:
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
    
    # Corregido: accedemos correctamente a las dimensiones del array
    # Dividimos los 21 partidos en 14 (regular) y 7 (revancha)
    return (sc[:,:14].max(0)>=11).mean(), (sc[:,14:].max(0)>=6).mean()