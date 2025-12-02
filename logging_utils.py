# ecoeval/logging_utils.py
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


def append_run_to_csv(path: str, row: Dict):
    """
    Append a single run row to CSV, adding a timestamp.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row_with_time = {"timestamp": datetime.utcnow().isoformat() + "Z"}
    row_with_time.update(row)

    df_new = pd.DataFrame([row_with_time])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)


def load_leaderboard(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None
