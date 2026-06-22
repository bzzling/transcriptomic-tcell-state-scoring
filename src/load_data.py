import gzip
from pathlib import Path

import pandas as pd


def load_counts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="gzip")
    gene_id_col = df.columns[0]  # "Probe Set ID"

    if "Gene Symbol" in df.columns:
        df = df.drop(columns=[gene_id_col]).set_index("Gene Symbol")
    else:
        df = df.set_index(gene_id_col)

    df.index.name = "gene"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.groupby(df.index).mean()
    return df


def load_series_matrix(path: str | Path) -> pd.DataFrame:
    rows: dict[str, list[str]] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            key = line.split("\t")[0]
            if key in {"!Sample_geo_accession", "!Sample_title"}:
                values = [v.strip().strip('"') for v in line.split("\t")[1:]]
                rows[key] = values

    titles = rows["!Sample_title"]
    sample_ids = [t.split(":")[0].strip() for t in titles]

    return pd.DataFrame({
        "sample_id": sample_ids,
        "geo_accession": rows["!Sample_geo_accession"],
        "title": titles,
    })
