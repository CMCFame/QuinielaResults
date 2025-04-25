# ===========================================================
# parser.py
# ===========================================================
from __future__ import annotations
import re, pandas as pd

_PATTERN_ROW = re.compile(r"^(.*?)(L|E|V)(?:\s+|\t)(.*)$", re.I)

def parse_heat_map(text: str) -> pd.DataFrame:
    """Parse pasted 20‑column heat‑map into DataFrame (index = match#)."""
    rows = []
    for line in text.strip().splitlines():
        parts = re.split(r"\s+", line.strip())
        # Expect: description then 20 codes
        if len(parts) < 21:
            continue
        match = " ".join(parts[:-20])
        codes = parts[-20:]
        rows.append([match, *codes])
    cols = ["match"] + [f"Q{i+1}" for i in range(20)]
    df = pd.DataFrame(rows, columns=cols)
    df = df.set_index("match")
    return df
