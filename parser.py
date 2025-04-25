# ===========================================================
# parser.py  (robust & debug‑friendly)
# ===========================================================
from __future__ import annotations
import re, pandas as pd

_HEADER_RE = re.compile(r"^\s*Partido\b", re.I)


def _split_row(line: str) -> tuple[str, list[str]]:
    """Split a row using *rsplit* so the match name may contain spaces.

    Returns ``(match_name, [20 codes])``.
    """
    parts = line.rstrip().rsplit(maxsplit=20)  # last 20 tokens = codes
    if len(parts) < 21:
        raise ValueError("Fila incompleta: se requieren 20 códigos L/E/V.")
    match = parts[0]
    codes = parts[1:]
    if len(codes) != 20:
        raise ValueError("Fila inválida: exactamente 20 códigos al final.")
    return match, codes


def parse_heat_map(text: str, *, debug: bool = False) -> pd.DataFrame:
    """Parse user‑pasted heat‑map.

    • Ignora líneas vacías, cabeceras y cualquier fila sin « vs ».
    • Valida que existan 21 filas × 20 columnas, salvo que ``debug=True``.
    """
    rows: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _HEADER_RE.match(line) or " vs " not in line:
            continue
        match, codes = _split_row(line)
        rows.append([match, *codes])

    df = pd.DataFrame(rows, columns=["match"] + [f"Q{i+1}" for i in range(20)]).set_index("match")

    if not debug and len(df) != 21:
        raise ValueError(f"La malla debe tener 21 filas; se encontraron {len(df)}.")

    return df
