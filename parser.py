# ===========================================================
# parser.py  (updated to ignore header lines)
# ===========================================================
from __future__ import annotations
import re, pandas as pd

_HEADER_RE = re.compile(r"^\s*Partido\b", re.I)
_SPLIT_RE  = re.compile(r"\s+")

def parse_heat_map(text: str) -> pd.DataFrame:
    """Convert pasted 20-column heat-map into a DataFrame.

    •   Ignora líneas vacías o que no contengan " vs ".
    •   Descarta cabeceras que empiecen por «Partido …».
    •   Exige exactamente 21 filas y 20 códigos L/E/V (o dobles) por fila.
    """
    rows: list[list[str]] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or " vs " not in line or _HEADER_RE.match(line):
            continue  # descartar cabeceras, líneas vacías o mal formadas
        parts = _SPLIT_RE.split(line)
        if len(parts) < 21:
            raise ValueError("Cada fila debe tener 1 descripción + 20 códigos L/E/V.")
        match = " ".join(parts[:-20])
        codes = parts[-20:]
        rows.append([match, *codes])
    if len(rows) != 21:
        raise ValueError("La malla debe contener exactamente 21 filas de partidos.")
    cols = ["match"] + [f"Q{i+1}" for i in range(20)]
    df = pd.DataFrame(rows, columns=cols).set_index("match")
    return df