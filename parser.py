# ==========================================================
# parser.py  (texto → DataFrame con 21×20)
# ==========================================================
from __future__ import annotations
import re, pandas as pd

_HEADER_RE = re.compile(r"^\s*Partido\b", re.I)
_CODE_RE   = re.compile(r"^[LEV](?:/[LEV])?$", re.I)


def _tokens_to_row(tokens: list[str], row: int) -> tuple[str, list[str]]:
    """Devuelve (match, [20 códigos]). Acepta últimos 20 códigos pegados."""
    if len(tokens) < 21:  # quizá venga todo junto
        blob = tokens[-1]
        if len(blob) == 20 and set(blob.upper()) <= {"L","E","V"}:
            codes = list(blob.upper())
            match = " ".join(tokens[:-1])
        else:
            raise ValueError(f"Fila {row}: se requieren 20 códigos L/E/V.")
    else:
        codes = tokens[-20:]
        match = " ".join(tokens[:-20])

    if any(not _CODE_RE.fullmatch(c) for c in codes):
        raise ValueError(f"Fila {row}: código inválido. Use L, E, V o dobles L/E.")
    return match.strip(), codes


def parse_heat_map(text: str, *, debug: bool=False) -> pd.DataFrame:
    rows=[]
    for i, raw in enumerate(text.splitlines(),1):
        line=raw.strip()
        if not line or _HEADER_RE.match(line) or " vs " not in line:
            continue
        match,codes=_tokens_to_row(line.split(), i)
        rows.append([match,*codes])
    df=pd.DataFrame(rows,columns=["match"]+[f"Q{i+1}" for i in range(20)]).set_index("match")
    if not debug and len(df)!=21:
        raise ValueError(f"Se esperaban 21 filas (14 regular + 7 revancha); hay {len(df)}.")
    return df