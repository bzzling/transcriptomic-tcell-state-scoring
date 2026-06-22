import re
import pandas as pd


def _parse_timepoint(sample_id: str) -> str:
    m = re.match(r"\d+E\d(\d+h)", sample_id)
    return m.group(1) if m else "unknown"


def _parse_condition(sample_id: str) -> str:
    if "SINACT" in sample_id:
        return "resting"
    if "PDL1" in sample_id:
        return "activated_PD1_PDL1"
    if "ACT" in sample_id:
        return "activated_CD3_CD28"
    return "unknown"


def build_metadata(series_df: pd.DataFrame) -> pd.DataFrame:
    meta = series_df.copy()
    meta["timepoint"] = meta["sample_id"].map(_parse_timepoint)
    meta["condition"] = meta["sample_id"].map(_parse_condition)
    return meta.set_index("sample_id")

