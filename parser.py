from __future__ import annotations
import re
import pandas as pd

_HEADER_RE = re.compile(r"^\s*Partido\b", re.I)


def _split_row(line: str, row_no: int) -> tuple[str, list[str]]:
    """
    Devuelve  (nombre_del_partido, [20 códigos]).
    Lanza error detallado si la fila no trae 20 códigos.
    """
    parts = line.rstrip().rsplit(maxsplit=20)       # últimos 20 = códigos
    if len(parts) < 21:
        raise ValueError(f"Fila {row_no}: sólo {len(parts)-1} códigos, faltan {21-len(parts)}")
    match, codes = parts[0], parts[1:]
    if len(codes) != 20:
        raise ValueError(f"Fila {row_no}: se esperaban 20 códigos, llegaron {len(codes)}")
    return match, codes


def parse_heat_map(text: str, *, debug: bool = False) -> pd.DataFrame:
    """
    Convierte el bloque pegado en DataFrame  (21×20).  
    Ignora cabeceras, líneas vacías y valida longitud exacta.
    """
    rows: list[list[str]] = []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or _HEADER_RE.match(line) or " vs " not in line:
            continue
        rows.append([*_split_row(line, n)])

    df = pd.DataFrame(rows, columns=["match"] + [f"Q{i+1}" for i in range(20)]).set_index("match")

    if not debug and len(df) != 21:
        raise ValueError(f"La malla debe contener 21 filas; se encontraron {len(df)}.")

    return df
