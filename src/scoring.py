"""Signature scoring via per-gene z-score averaging."""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_signatures(path: Path) -> dict[str, list[str]]:
    """Load gene signatures from a YAML file.

    Args:
        path: Path to signatures YAML (e.g. config/signatures.yml).

    Returns:
        Dict mapping signature name -> list of gene symbols.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def log2_transform(counts: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(x + 1) transformation.

    Args:
        counts: Genes x samples raw/normalized counts.

    Returns:
        log2-transformed DataFrame.
    """
    return np.log2(counts + 1)


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (row) across samples.

    Args:
        expr: Genes x samples expression DataFrame.

    Returns:
        Z-scored DataFrame; genes with zero std are dropped.
    """
    mean = expr.mean(axis=1)
    std = expr.std(axis=1)
    expr_z = expr.subtract(mean, axis=0).divide(std, axis=0)
    return expr_z.dropna(how="all")


def score_signatures(
    expr_z: pd.DataFrame,
    signatures: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each signature per sample by averaging z-scored gene expression.

    Args:
        expr_z: Z-scored genes x samples DataFrame.
        signatures: Dict mapping signature name -> gene list.

    Returns:
        Tuple of:
          - scores: samples x signatures DataFrame
          - missing: DataFrame listing genes absent from the dataset per signature
    """
    scores: dict[str, pd.Series] = {}
    missing_rows: list[dict] = []

    for sig_name, genes in signatures.items():
        present = [g for g in genes if g in expr_z.index]
        absent = [g for g in genes if g not in expr_z.index]
        for g in absent:
            missing_rows.append({"signature": sig_name, "gene": g})
        if present:
            scores[sig_name] = expr_z.loc[present].mean(axis=0)

    scores_df = pd.DataFrame(scores)
    missing_df = pd.DataFrame(missing_rows, columns=["signature", "gene"])
    return scores_df, missing_df


def summarize_scores_by_group(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Average signature scores by metadata groups.

    Args:
        scores: samples x signatures DataFrame (indexed by sample_id).
        meta: Samples x attributes DataFrame (indexed by sample_id).
        group_cols: Columns in meta to group by (e.g. ['condition', 'timepoint']).

    Returns:
        grouped x signatures DataFrame with a MultiIndex of group_cols.
    """
    joined = scores.join(meta[group_cols])
    return joined.groupby(group_cols).mean(numeric_only=True)
