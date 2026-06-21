"""Build and validate sample metadata."""

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
    """Enrich series matrix DataFrame with condition and timepoint columns.

    Args:
        series_df: Output of load_series_matrix() — must have 'sample_id' column.

    Returns:
        DataFrame indexed by sample_id with geo_accession, condition, timepoint.
    """
    meta = series_df.copy()
    meta["timepoint"] = meta["sample_id"].map(_parse_timepoint)
    meta["condition"] = meta["sample_id"].map(_parse_condition)
    return meta.set_index("sample_id")


def validate_samples(counts: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Raise ValueError if counts columns and metadata index don't match.

    Args:
        counts: Genes x samples expression DataFrame.
        meta: DataFrame indexed by sample_id.
    """
    count_samples = set(counts.columns)
    meta_samples = set(meta.index)
    missing_in_meta = count_samples - meta_samples
    missing_in_counts = meta_samples - count_samples
    if missing_in_meta or missing_in_counts:
        raise ValueError(
            f"Sample mismatch.\n"
            f"  In counts but not metadata: {missing_in_meta}\n"
            f"  In metadata but not counts: {missing_in_counts}"
        )
