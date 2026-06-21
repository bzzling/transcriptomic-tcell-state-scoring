"""Plotting helpers for PCA, signature scores, and marker gene heatmaps."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def set_plot_style() -> None:
    """Set global plotting style. Call once at the top of a notebook."""
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _align_meta(expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Return metadata rows ordered to match expression matrix columns."""
    missing = set(expr.columns) - set(meta.index)
    extra = set(meta.index) - set(expr.columns)
    if missing or extra:
        raise ValueError(
            f"Sample mismatch.\n"
            f"  Missing from metadata: {sorted(missing)}\n"
            f"  Extra in metadata:     {sorted(extra)}"
        )
    return meta.loc[expr.columns]


def plot_pca(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: Path,
    color_col: str = "condition",
    style_col: str = "timepoint",
) -> None:
    """Run PCA on samples; save scatter colored by condition, shaped by timepoint.

    Args:
        expr: Genes x samples log2-transformed expression DataFrame.
        meta: Samples x attributes DataFrame indexed by sample_id.
        out_path: File path to save the figure.
        color_col: Metadata column for point color.
        style_col: Metadata column for point marker style.
    """
    meta = _align_meta(expr, meta)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(expr.T)

    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=expr.columns).join(meta)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2",
        hue=color_col,
        style=style_col if style_col in pca_df.columns else None,
        s=85, edgecolor="black", linewidth=0.4, ax=ax,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA of GSE122149 CD8 T-cell RNA-seq samples")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_signature_scores(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: Path,
) -> None:
    """Strip plot of per-sample signature scores grouped by condition.

    Args:
        scores: samples x signatures DataFrame indexed by sample_id.
        meta: Samples x attributes DataFrame indexed by sample_id.
        out_path: File path to save the figure.
    """
    meta = _align_meta(scores.T, meta)
    long_df = (
        scores.join(meta[["condition", "timepoint"]])
        .reset_index(names="sample_id")
        .melt(id_vars=["sample_id", "condition", "timepoint"],
              var_name="signature", value_name="score")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(
        data=long_df, x="signature", y="score", hue="condition",
        dodge=True, alpha=0.8, size=6, linewidth=0.4, edgecolor="black", ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Mean z-scored expression")
    ax.set_title("T-cell transcriptional program scores — GSE122149")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_signature_heatmap(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: Path,
    sort_cols: list[str] = ["condition", "timepoint"],
) -> None:
    """Seaborn heatmap of per-sample signature scores sorted by condition/timepoint.

    Args:
        scores: samples x signatures DataFrame indexed by sample_id.
        meta: Samples x attributes DataFrame indexed by sample_id.
        out_path: File path to save the figure.
        sort_cols: Metadata columns to sort rows by.
    """
    meta = _align_meta(scores.T, meta)
    valid_sort = [c for c in sort_cols if c in meta.columns]
    order = meta.sort_values(valid_sort).index if valid_sort else scores.index
    plot_data = scores.loc[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_data) * 0.28)))
    sns.heatmap(
        plot_data, cmap="vlag", center=0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Mean z-score"}, ax=ax,
    )
    ax.set_title("Signature scores — per sample (relative within dataset)")
    ax.set_xlabel("")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_heatmap(
    grouped_scores: pd.DataFrame,
    out_path: Path,
) -> None:
    """Seaborn heatmap of condition/timepoint-averaged signature scores.

    Args:
        grouped_scores: Output of summarize_scores_by_group() — MultiIndex rows x signatures.
        out_path: File path to save the figure.
    """
    if isinstance(grouped_scores.index, pd.MultiIndex):
        plot_data = grouped_scores.copy()
        plot_data.index = [" | ".join(str(v) for v in idx) for idx in grouped_scores.index]
    else:
        plot_data = grouped_scores

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_data) * 0.4)))
    sns.heatmap(
        plot_data, cmap="vlag", center=0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Mean z-score"}, annot=True, fmt=".2f", ax=ax,
    )
    ax.set_title("Signature scores — condition/timepoint mean")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_clustermap(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: Path,
    n_top_var: int = 200,
) -> None:
    """Hierarchical clustering clustermap of the top most-variable genes.

    Columns (samples) are color-annotated by condition × timepoint.
    Dendrogram on top, color bar below, z-score colorbar on the right.

    Args:
        expr: Genes x samples log2-transformed expression DataFrame.
        meta: Samples x attributes DataFrame indexed by sample_id.
        out_path: File path to save the figure.
        n_top_var: Number of top-variance genes to include.
    """
    from matplotlib.colors import LinearSegmentedColormap

    meta = _align_meta(expr, meta)

    top_genes = expr.var(axis=1).nlargest(n_top_var).index
    plot_data = expr.loc[top_genes]

    groups = (meta["condition"] + "  " + meta["timepoint"]).rename("condition_timepoint")
    unique_groups = sorted(groups.unique())
    palette = dict(zip(unique_groups, sns.color_palette("tab10", len(unique_groups))))
    col_colors = groups.map(palette)

    # Red → white → purple gradient
    rwp = LinearSegmentedColormap.from_list("rwp", ["#6A0DAD", "#FFFFFF", "#CC0000"])

    g = sns.clustermap(
        plot_data,
        col_colors=col_colors,          # color bar at bottom of columns
        row_cluster=True,
        col_cluster=True,
        cmap=rwp,
        center=0,
        vmin=-2, vmax=2,
        z_score=0,                      # z-score genes (rows)
        figsize=(13, 11),
        xticklabels=False,              # no sample names
        yticklabels=False,
        linewidths=0,
        cbar_pos=(1.02, 0.3, 0.025, 0.3),   # colorbar on the right
        dendrogram_ratio=(0.12, 0.15),  # (row, col) — col dendrogram on top
    )

    # Move col_colors bar to bottom: clustermap draws it above heatmap by default;
    # flip by repositioning the ax_col_colors axes below the heatmap axes.
    hm_pos  = g.ax_heatmap.get_position()
    cc_pos  = g.ax_col_colors.get_position()
    cc_h    = cc_pos.height
    g.ax_col_colors.set_position([
        hm_pos.x0, hm_pos.y0 - cc_h - 0.005,
        hm_pos.width, cc_h,
    ])

    # Colorbar label
    g.ax_cbar.set_title("Z-score\n(row)", fontsize=9, pad=6)
    g.ax_cbar.tick_params(labelsize=8)

    # Axes labels
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    # Title on the column dendrogram (top)
    g.ax_col_dendrogram.set_title(
        "Hierarchical clustering  |  GSE122149 CD8 T cells",
        fontsize=11, pad=8,
    )

    # Legend at top-right, no title
    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=c, markersize=9, label=lbl)
        for lbl, c in palette.items()
    ]
    g.ax_col_dendrogram.legend(
        handles=handles, bbox_to_anchor=(1.0, 1.0), loc="upper right",
        frameon=False, fontsize=8,
    )

    g.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close("all")


def plot_marker_heatmap(
    expr_z: pd.DataFrame,
    meta: pd.DataFrame,
    genes: list[str],
    out_path: Path,
    sort_cols: list[str] = ["condition", "timepoint"],
) -> None:
    """Seaborn heatmap of curated marker gene z-scores (genes x samples).

    Only genes present in expr_z are plotted.

    Args:
        expr_z: Z-scored genes x samples DataFrame.
        meta: Samples x attributes DataFrame indexed by sample_id.
        genes: Ordered list of marker genes to display.
        out_path: File path to save the figure.
        sort_cols: Metadata columns to sort samples by.
    """
    meta = _align_meta(expr_z, meta)
    valid_sort = [c for c in sort_cols if c in meta.columns]
    col_order = meta.sort_values(valid_sort).index if valid_sort else expr_z.columns

    present = [g for g in genes if g in expr_z.index]
    plot_data = expr_z.loc[present, col_order]

    fig, ax = plt.subplots(figsize=(max(8, len(col_order) * 0.45), max(5, len(present) * 0.35)))
    sns.heatmap(
        plot_data, cmap="vlag", center=0, vmin=-2, vmax=2,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Z-score"}, ax=ax,
    )
    ax.set_title("Marker gene expression — GSE122149 CD8 T cells")
    ax.set_xlabel("Sample (sorted by condition → timepoint)")
    ax.set_ylabel("Gene")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
