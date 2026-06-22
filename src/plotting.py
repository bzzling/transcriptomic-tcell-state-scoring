from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def set_plot_style() -> None:
    """Set global plotting style. Call once at the top of a notebook."""
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 200,
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


_GROUP_PALETTE: list[tuple[str, str, str]] = [
    ("resting",            "6h",  "#D3D3D3"),
    ("activated_CD3_CD28", "6h",  "#FFD700"),
    ("activated_CD3_CD28", "24h", "#FF8C00"),
    ("activated_CD3_CD28", "48h", "#CC0000"),
    ("activated_PD1_PDL1", "6h",  "#ADD8E6"),
    ("activated_PD1_PDL1", "24h", "#4682B4"),
    ("activated_PD1_PDL1", "48h", "#00008B"),
]

_GROUP_LABELS: dict[tuple[str, str], str] = {
    ("resting",            "6h"):  r"$T_{\mathrm{CTRL}}$",
    ("activated_CD3_CD28", "6h"):  r"$T_{\mathrm{ACT}}$ 6h",
    ("activated_CD3_CD28", "24h"): r"$T_{\mathrm{ACT}}$ 24h",
    ("activated_CD3_CD28", "48h"): r"$T_{\mathrm{ACT}}$ 48h",
    ("activated_PD1_PDL1", "6h"):  r"$T_{\mathrm{ACT+PD1}}$ 6h",
    ("activated_PD1_PDL1", "24h"): r"$T_{\mathrm{ACT+PD1}}$ 24h",
    ("activated_PD1_PDL1", "48h"): r"$T_{\mathrm{ACT+PD1}}$ 48h",
}


def plot_pca(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: str | Path,
) -> None:
    meta = _align_meta(expr, meta)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(expr.T)

    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=expr.columns).join(meta)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    legend_handles = []
    for cond, tp, colour in _GROUP_PALETTE:
        mask = (pca_df["condition"] == cond) & (pca_df["timepoint"] == tp)
        if not mask.any():
            continue
        label = _GROUP_LABELS[(cond, tp)]
        ax.scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
                   color=colour, s=60, edgecolors="black", linewidths=0.4,
                   marker="o", zorder=3)
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colour,
                       markersize=6, markeredgecolor="black", markeredgewidth=0.4, label=label)
        )

    xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)"
    ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)"
    ax.set_xlabel(xlabel, fontsize=8, fontweight="bold", color="black")
    ax.set_ylabel(ylabel, fontsize=8, fontweight="bold", color="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(7)
        label.set_fontweight("bold")
        label.set_color("black")

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", which="both", length=4, width=1.2,
                   colors="black", direction="out", bottom=True, left=True)

    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              edgecolor="black", fancybox=False, fontsize=7, title="",
              handlelength=1.2, handletextpad=0.4, borderpad=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=600)




def plot_gene_clustermap(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    genes: list[str],
    out_path: str | Path,
) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    meta = _align_meta(expr, meta)
    present = [g for g in genes if g in expr.index]

    _key_to_colour = {(c, t): colour for c, t, colour in _GROUP_PALETTE}
    group_order = [(c, t) for c, t, _ in _GROUP_PALETTE]
    sample_order = [
        s for c, t in group_order
        for s in meta[(meta["condition"] == c) & (meta["timepoint"] == t)].index
    ]

    plot_data = expr.loc[present, sample_order]

    mu = plot_data.mean(axis=1)
    sd = plot_data.std(axis=1)
    z = plot_data.sub(mu, axis=0).div(sd, axis=0).dropna(how="all")

    col_colors = pd.Series(
        [_key_to_colour[(meta.loc[s, "condition"], meta.loc[s, "timepoint"])] for s in sample_order],
        index=sample_order,
    )

    rwp = LinearSegmentedColormap.from_list("rwp", ["#6A0DAD", "#FFFFFF", "#CC0000"])

    g = sns.clustermap(
        z,
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=False,
        cmap=rwp,
        center=0, vmin=-2, vmax=2,
        figsize=(9, max(5, len(z) * 0.45)),
        xticklabels=False,
        yticklabels=True,
        cbar_pos=(1.02, 0.3, 0.025, 0.3),
        dendrogram_ratio=(0.15, 0.0),
    )
    g.fig.set_dpi(plt.rcParams["figure.dpi"])

    g.ax_col_colors.set_ylabel("")
    g.ax_col_colors.yaxis.set_visible(False)
    g.ax_col_colors.xaxis.set_visible(False)

    hm_pos = g.ax_heatmap.get_position()
    cc_h   = g.ax_col_colors.get_position().height
    g.ax_col_colors.set_position([hm_pos.x0, hm_pos.y0 - cc_h - 0.005, hm_pos.width, cc_h])

    g.ax_cbar.set_title("Z-score", fontsize=9, pad=6)
    g.ax_cbar.tick_params(labelsize=8)
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="y", which="both", length=0, labelsize=8)
    g.fig.suptitle("")

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=colour, edgecolor="black", linewidth=0.6,
              label=_GROUP_LABELS[(cond, tp)])
        for cond, tp, colour in _GROUP_PALETTE
        if cond in meta["condition"].values
    ]
    hm = g.ax_heatmap.get_position()
    legend_x = hm.x0 + hm.width / 2
    g.fig.legend(handles=handles, loc="lower center",
                 bbox_to_anchor=(legend_x, -0.06), ncol=len(handles),
                 frameon=True, edgecolor="black", fancybox=False,
                 fontsize=8, handlelength=1.5, handletextpad=0.5,
                 borderpad=0.6)

    g.savefig(out_path, bbox_inches="tight", dpi=600)


def plot_clustermap(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: str | Path,
    n_top_var: int = 200,
) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    meta = _align_meta(expr, meta)

    top_genes = expr.var(axis=1).nlargest(n_top_var).index
    plot_data = expr.loc[top_genes]

    groups = (meta["condition"] + "  " + meta["timepoint"]).rename("")
    _key_to_colour = {(c, t): colour for c, t, colour in _GROUP_PALETTE}
    _key_to_label  = _GROUP_LABELS
    col_colors = groups.map(lambda v: _key_to_colour.get(
        (v.split("  ", 1)[0], v.split("  ", 1)[1]), "#cccccc"))

    rwp = LinearSegmentedColormap.from_list("rwp", ["#6A0DAD", "#FFFFFF", "#CC0000"])

    g = sns.clustermap(
        plot_data,
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=True,
        cmap=rwp,
        center=0,
        vmin=-2, vmax=2,
        z_score=0,
        figsize=(13, 11),
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        cbar_pos=(1.02, 0.3, 0.025, 0.3),
        dendrogram_ratio=(0.12, 0.06),
    )
    g.fig.set_dpi(plt.rcParams["figure.dpi"])

    hm_pos  = g.ax_heatmap.get_position()
    cc_pos  = g.ax_col_colors.get_position()
    cc_h    = cc_pos.height
    g.ax_col_colors.set_position([
        hm_pos.x0, hm_pos.y0 - cc_h - 0.005,
        hm_pos.width, cc_h,
    ])

    cd_pos = g.ax_col_dendrogram.get_position()
    g.ax_col_dendrogram.set_position([
        hm_pos.x0, hm_pos.y1,
        hm_pos.width, cd_pos.height,
    ])

    g.ax_cbar.set_title("Z-score", fontsize=9, pad=6)
    g.ax_cbar.tick_params(labelsize=8)

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    g.ax_col_colors.set_ylabel("")
    g.ax_col_colors.yaxis.set_visible(False)
    g.ax_col_colors.xaxis.set_visible(False)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=colour, edgecolor="black", linewidth=0.6,
              label=_GROUP_LABELS[(cond, tp)])
        for cond, tp, colour in _GROUP_PALETTE
    ]
    hm = g.ax_heatmap.get_position()
    legend_x = hm.x0 + hm.width / 2
    g.fig.legend(handles=handles, loc="lower center",
                 bbox_to_anchor=(legend_x, -0.10), ncol=len(handles),
                 frameon=True, edgecolor="black", fancybox=False,
                 fontsize=11, handlelength=2.0, handletextpad=0.6,
                 borderpad=0.7)

    g.savefig(out_path, bbox_inches="tight", dpi=600)

