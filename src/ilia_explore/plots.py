"""Figures for the exploratory analysis (matplotlib, saved as PNG in figures/).

Colour roles
------------
* Dimensions (3 categories): categorical slots 1-3, which stay distinguishable
  for colour-blind readers in every pairing.
* Clusters: categorical slots in fixed order, always paired with a marker
  shape, so identity never rests on colour alone.
* Correlations / z-score profiles: blue <-> red diverging map with a neutral
  grey midpoint.
* Counts and magnitudes: single-hue blue ramp.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram

from .data import DIM_LABEL, FIGURES, ISO3, SUBDIM_CODE

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e4e3df"
NEUTRAL = "#f0efec"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DIM_COLOR = {"FH": SERIES[0], "GOB": SERIES[1], "IDA": SERIES[2]}
DIM_ORDER = ["FH", "GOB", "IDA"]
SUBDIM_ORDER = list(SUBDIM_CODE.values())

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED_ARM = ["#fbd5d4", "#f2a09f", "#e34948", "#b52f2f", "#7d1f1f"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", BLUE_RAMP[::-1][:5] + [NEUTRAL] + RED_ARM, N=256
)
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", [SURFACE] + BLUE_RAMP, N=256)


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")


def _iso(index) -> list[str]:
    return [ISO3.get(c, c) for c in index]


def _dim_legend(ax, loc="lower right", **kw):
    handles = [Patch(color=DIM_COLOR[d], label=DIM_LABEL[d]) for d in DIM_ORDER]
    ax.legend(handles=handles, loc=loc, **kw)


def _cluster_style(c: int) -> tuple[str, str]:
    return SERIES[(c - 1) % len(SERIES)], MARKERS[(c - 1) % len(MARKERS)]


def _cluster_legend(ax, labels, loc="best", title=None):
    handles = []
    for c in sorted(set(labels)):
        col, mk = _cluster_style(c)
        n = int(np.sum(np.asarray(labels) == c))
        handles.append(Line2D([], [], marker=mk, color=col, ls="", ms=8, label=f"Cluster {c} (n={n})"))
    ax.legend(handles=handles, loc=loc, title=title)


def _order_codes(codes, codebook: pd.DataFrame) -> list[str]:
    cb = codebook.loc[list(codes)]
    key = cb["dim_code"].map({d: i for i, d in enumerate(DIM_ORDER)})
    return list(cb.assign(_k=key).sort_values(["_k", "subdim_code"], kind="stable").index)


def _dim_band(ax, codes, codebook, axis="x"):
    """Thin coloured band marking dimension blocks along an axis of a heatmap."""
    dims = codebook.loc[codes, "dim_code"].to_numpy()
    for i, d in enumerate(dims):
        if axis == "x":
            ax.add_patch(Rectangle((i, -1.6), 1, 1.1, color=DIM_COLOR[d], clip_on=False, lw=0))
        else:
            ax.add_patch(Rectangle((-1.6, i), 1.1, 1, color=DIM_COLOR[d], clip_on=False, lw=0))


# --------------------------------------------------------------------- data ---

def missingness_map(masked_wide: pd.DataFrame, codebook: pd.DataFrame):
    """Countries x subindicators; dark cells are imputed in ``final``."""
    codes = _order_codes(masked_wide.index, codebook)
    miss = masked_wide.loc[codes].isna().T
    miss = miss.loc[miss.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.imshow(miss.to_numpy(), aspect="auto", interpolation="nearest",
              cmap=ListedColormap([GRID, BLUE_RAMP[5]]), vmin=0, vmax=1,
              extent=(0, miss.shape[1], miss.shape[0], 0))
    ax.set_yticks(np.arange(len(miss)) + 0.5, [f"{ISO3[c]}  ({int(n)})" for c, n in miss.sum(axis=1).items()])
    ax.set_xticks([])
    ax.grid(False)
    _dim_band(ax, codes, codebook, "x")
    ax.set_ylim(len(miss), -1.8)
    ax.set_xlabel("Subindicators, grouped by dimension (number of imputed cells per country in brackets)")
    ax.set_title("Where the imputed values are: dark cells were empty in the raw data")
    _dim_legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def missing_by_dimension(masked_wide: pd.DataFrame, codebook: pd.DataFrame):
    miss = masked_wide.isna().groupby(codebook["dim_code"]).sum().T
    miss = miss.loc[miss.sum(axis=1).sort_values().index, DIM_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    left = np.zeros(len(miss))
    for d in DIM_ORDER:
        ax.barh(_iso(miss.index), miss[d], left=left, color=DIM_COLOR[d], label=DIM_LABEL[d],
                height=0.7, edgecolor=SURFACE, linewidth=1.5)
        left += miss[d].to_numpy()
    for y, tot in enumerate(left):
        if tot:
            ax.text(tot + 0.4, y, f"{int(tot)}", va="center", color=INK_2, fontsize=8.5)
    ax.set_xlabel("Imputed cells")
    ax.grid(axis="y", visible=False)
    ax.set_title("Imputed cells per country, by dimension")
    ax.legend(loc="lower right")
    return fig


def level_vs_missing(level: pd.Series, n_imputed: pd.Series, rho: float, pval: float):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(n_imputed, level.loc[n_imputed.index], s=46, color=SERIES[0],
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for c in n_imputed.index:
        ax.annotate(ISO3[c], (n_imputed[c], level[c]), xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=INK_2)
    ax.set_xlabel("Imputed cells in the country's row")
    ax.set_ylabel("Mean z-score on the 86 complete subindicators")
    p_txt = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
    ax.set_title(f"Do countries with gaps score lower on what we observe?  Spearman ρ = {rho:.2f} ({p_txt})",
                 fontsize=10)
    ax.axhline(0, color=INK_3, lw=0.8, zorder=1)
    return fig


# -------------------------------------------------------------- correlation ---

def correlation_heatmap(corr: pd.DataFrame, codebook: pd.DataFrame, title: str):
    codes = _order_codes(corr.index, codebook)
    C = corr.loc[codes, codes].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 9.4))
    im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest",
                   extent=(0, len(codes), len(codes), 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    _dim_band(ax, codes, codebook, "x"); _dim_band(ax, codes, codebook, "y")
    ax.set_xlim(-1.8, len(codes)); ax.set_ylim(len(codes), -1.8)
    # dimension block outlines
    dims = codebook.loc[codes, "dim_code"].to_numpy()
    edges = [0] + [i for i in range(1, len(dims)) if dims[i] != dims[i - 1]] + [len(dims)]
    for a, b in zip(edges[:-1], edges[1:]):
        ax.add_patch(Rectangle((a, a), b - a, b - a, fill=False, ec=INK_2, lw=1))
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Spearman ρ (pairwise, real data only)")
    cb.outline.set_visible(False)
    ax.set_title(title)
    _dim_legend(ax, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    return fig


def threshold_bars(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    th = summary["threshold"].tolist()
    x = np.arange(len(th))
    w = 0.26
    for i, d in enumerate(DIM_ORDER):
        kept = summary[f"kept_{d}"]
        of = summary[f"of_{d}"]
        bars = ax.bar(x + (i - 1) * w, kept, width=w - 0.02, color=DIM_COLOR[d],
                      label=f"{DIM_LABEL[d]} (of {int(of.iloc[0])})")
        for b, k in zip(bars, kept):
            ax.text(b.get_x() + b.get_width() / 2, k + 0.6, str(int(k)), ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [f"|ρ| ≥ {t}\n{int(k)} of {int(k + d)} kept"
                      for t, k, d in zip(th, summary["kept_total"], summary["dropped_total"])])
    ax.set_ylabel("Subindicators kept")
    ax.grid(axis="x", visible=False)
    ax.set_title("Subindicators kept after removing redundancy, by threshold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


# ---------------------------------------------------------------------- PCA ---

def scree(pa: pd.DataFrame, explained: pd.Series, title: str, n_show: int = 10):
    qcol = [c for c in pa.columns if c.startswith("null_q")][0]
    pa = pa.iloc[:n_show]
    x = np.arange(1, len(pa) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, pa["eigenvalue"], marker="o", color=SERIES[0], lw=2, ms=6, label="Observed eigenvalue")
    ax.plot(x, pa[qcol], marker="s", color=INK_3, lw=1.5, ms=5, ls="--",
            label="95th pct. of permuted data (noise)")
    for xi, e, s in zip(x, pa["eigenvalue"], explained.iloc[:n_show]):
        ax.annotate(f"{s:.0%}", (xi, e), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    k = int(pa["retain"].sum())
    ax.axvspan(0.5, k + 0.5, color=SERIES[0], alpha=0.07, lw=0)
    ax.set_xticks(x, [f"PC{i}" for i in x])
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{title}: {k} component{'s' if k != 1 else ''} above noise")
    ax.legend(loc="upper right")
    return fig


def biplot(res, codebook: pd.DataFrame, title: str, labels=None, n_arrows: int = 12,
           pcs=("PC1", "PC2")):
    a, b = pcs
    S = res.scores[[a, b]]
    L = res.loadings[[a, b]]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.axhline(0, color=INK_3, lw=0.7, zorder=1); ax.axvline(0, color=INK_3, lw=0.7, zorder=1)
    if labels is None:
        ax.scatter(S[a], S[b], s=50, color=INK_2, edgecolor=SURFACE, lw=1.5, zorder=4)
    else:
        for c in sorted(set(labels)):
            m = np.asarray(labels) == c
            col, mk = _cluster_style(c)
            ax.scatter(S[a][m], S[b][m], s=62, color=col, marker=mk, edgecolor=SURFACE, lw=1.5, zorder=4)
    country_texts = [ax.text(S.loc[c, a], S.loc[c, b], ISO3[c], fontsize=8.5, color=INK, zorder=5)
                     for c in S.index]
    # arrows: the subindicators most associated with the plane
    strength = np.sqrt((L**2).sum(axis=1)).sort_values(ascending=False)
    top = list(strength.index[:n_arrows])
    scale = 0.85 * np.abs(S.to_numpy()).max() / max(np.abs(L.loc[top].to_numpy()).max(), 1e-9)
    tips = L.loc[top] * scale
    # arrows ending at (almost) the same point get one shared label
    span = float(np.ptp(S.to_numpy()))
    bundles: list[list[str]] = []
    for code in top:
        for bnd in bundles:
            if np.hypot(*(tips.loc[code] - tips.loc[bnd[0]])) < 0.03 * span:
                bnd.append(code)
                break
        else:
            bundles.append([code])
    arrow_texts = []
    for bnd in bundles:
        for code in bnd:
            x, y = tips.loc[code, a], tips.loc[code, b]
            col = DIM_COLOR[codebook.loc[code, "dim_code"]]
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, alpha=0.75), zorder=2)
        x, y = tips.loc[bnd[0], a], tips.loc[bnd[0], b]
        label = bnd[0] if len(bnd) == 1 else bnd[0] + "".join(f", .{c.split('.')[-1]}" if c.rsplit('.', 1)[0] == bnd[0].rsplit('.', 1)[0] else f", {c}" for c in bnd[1:])
        arrow_texts.append(ax.text(x, y, label, fontsize=7, color=INK_2, zorder=3,
                                   ha="left" if x >= 0 else "right", va="bottom" if y >= 0 else "top"))
    # annotations do not extend the axes, so set limits from points and arrow tips
    for col_, setter in ((a, ax.set_xlim), (b, ax.set_ylim)):
        vals = np.concatenate([S[col_].to_numpy(), tips[col_].to_numpy(), [0.0]])
        pad = 0.1 * (vals.max() - vals.min())
        setter(vals.min() - pad, vals.max() + pad)
    try:
        from adjustText import adjust_text
        adjust_text(country_texts, ax=ax, objects=None, expand=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.5))
    except Exception:  # labels stay at their points if adjustText is unavailable
        pass
    ax.set_xlabel(f"{a} ({res.explained[a]:.0%} of variance)")
    ax.set_ylabel(f"{b} ({res.explained[b]:.0%} of variance)")
    ax.set_title(title)
    handles = [Line2D([], [], color=DIM_COLOR[d], lw=1.5, label=DIM_LABEL[d]) for d in DIM_ORDER]
    leg = ax.legend(handles=handles, loc="upper left", title=f"Top {n_arrows} subindicators")
    ax.add_artist(leg)
    if labels is not None:
        _cluster_legend(ax, labels, loc="lower left")
    return fig


def loading_bars(res, codebook: pd.DataFrame, pc: str, title: str, n: int = 20):
    top = res.loadings[pc].abs().sort_values(ascending=False).index[:n]
    vals = res.loadings.loc[top, pc][::-1]
    names = [f"{c}  {codebook.loc[c, 'subindicador'][:48]}" for c in vals.index]
    fig, ax = plt.subplots(figsize=(8.6, 0.27 * n + 1.2))
    ax.barh(names, vals, color=[DIM_COLOR[codebook.loc[c, "dim_code"]] for c in vals.index],
            height=0.72, edgecolor=SURFACE, lw=1)
    ax.axvline(0, color=INK_3, lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(f"Correlation with {pc}")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(title)
    _dim_legend(ax, loc="lower right" if vals.mean() < 0 else "lower left")
    return fig


def subdim_loadings(res, codebook: pd.DataFrame, k: int, title: str):
    """Mean loading of each subdimension's subindicators on the first k PCs."""
    pcs = [f"PC{i + 1}" for i in range(k)]
    L = res.loadings[pcs].join(codebook[["subdim_code", "dim_code"]])
    M = L.groupby("subdim_code")[pcs].mean()
    n = L.groupby("subdim_code").size()
    order = [s for s in SUBDIM_ORDER if s in M.index]
    M, n = M.loc[order], n.loc[order]
    fig, ax = plt.subplots(figsize=(1.3 * k + 3.4, 4.6))
    im = ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.55 else INK)
    ax.set_xticks(range(k), [f"{p}\n({res.explained[p]:.0%})" for p in pcs])
    dim_of = codebook.drop_duplicates("subdim_code").set_index("subdim_code")["dim_code"]
    ax.set_yticks(range(len(order)), [f"{dim_of[s]}.{s}  (n={n[s]})" for s in order])
    for t, s in zip(ax.get_yticklabels(), order):
        t.set_color(DIM_COLOR[dim_of[s]])
    ax.grid(False)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label("Mean loading"); cb.outline.set_visible(False)
    return fig


# --------------------------------------------------------------- clustering ---

def k_diagnostics(ev: pd.DataFrame, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    e = ev.drop(index=1, errors="ignore")
    ax = axes[0]
    ax.plot(e.index, e["silhouette_ward"], marker="o", color=SERIES[0], lw=2, label="Ward")
    ax.plot(e.index, e["silhouette_kmeans"], marker="s", color=SERIES[1], lw=2, label="k-means")
    ax.set_xlabel("k"); ax.set_ylabel("Mean silhouette"); ax.set_xticks(e.index)
    ax.set_ylim(0, max(0.7, e[["silhouette_ward", "silhouette_kmeans"]].max().max() + 0.05))
    ax.set_title("Separation of clusters (higher is better)", fontsize=10)
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.errorbar(ev.index, ev["gap"], yerr=ev["gap_se"], marker="o", color=SERIES[2], lw=2, capsize=3)
    ax.set_xlabel("k"); ax.set_ylabel("Gap statistic"); ax.set_xticks(ev.index)
    ax.set_title("Gap vs. structureless reference (k = 1: no clusters)", fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    return fig


def dendro(Z: np.ndarray, index, k: int, labels: np.ndarray, title: str):
    """Ward dendrogram with leaves coloured by their cluster (same colours as maps)."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2 if k > 1 else Z[-1, 2] * 1.05
    with mpl.rc_context({"lines.linewidth": 1.3}):
        d = dendrogram(Z, labels=_iso(index), ax=ax, color_threshold=0,
                       above_threshold_color=INK_3, leaf_font_size=9)
    lab = dict(zip(_iso(index), labels))
    for t in ax.get_xticklabels():
        col, _ = _cluster_style(lab[t.get_text()])
        t.set_color(col); t.set_fontweight("semibold")
    for x, leaf in zip(ax.get_xticks(), d["ivl"]):
        col, mk = _cluster_style(lab[leaf])
        ax.plot(x, 0, marker=mk, color=col, ms=7, clip_on=False, zorder=4)
    ax.axhline(cut, color=SERIES[7], lw=1, ls="--")
    ax.text(ax.get_xlim()[1], cut, f"  cut: k = {k}", va="center", color=INK_2, fontsize=8.5)
    ax.set_ylabel("Ward merge height")
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    return fig


def silhouette_bars(sil: pd.Series, labels: np.ndarray, title: str):
    df = pd.DataFrame({"s": sil, "c": labels}).sort_values(["c", "s"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(df) + 1.3))
    ax.barh(_iso(df.index), df["s"], color=[_cluster_style(c)[0] for c in df["c"]],
            height=0.72, edgecolor=SURFACE, lw=1)
    ax.axvline(0, color=INK_3, lw=0.8)
    ax.axvline(sil.mean(), color=INK_2, lw=1, ls="--")
    ax.text(sil.mean(), len(df) - 0.3, f" mean {sil.mean():.2f}", color=INK_2, fontsize=8)
    ax.set_xlabel("Silhouette (negative: closer to another cluster)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    _cluster_legend(ax, labels, loc="lower right")
    return fig


def comembership(co: pd.DataFrame, labels: np.ndarray, title: str):
    order = pd.DataFrame({"c": labels}, index=co.index).sort_values("c").index
    C = co.loc[order, order]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(C.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(order)), _iso(order), rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)), _iso(order), fontsize=8)
    lab = pd.Series(labels, index=co.index).loc[order].to_numpy()
    for i, c in enumerate(lab):
        col, _ = _cluster_style(c)
        ax.get_xticklabels()[i].set_color(col); ax.get_yticklabels()[i].set_color(col)
    edges = [0] + [i for i in range(1, len(lab)) if lab[i] != lab[i - 1]] + [len(lab)]
    for a, b in zip(edges[:-1], edges[1:]):
        ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec=INK, lw=1.2))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Share of bootstrap resamples in the same cluster"); cb.outline.set_visible(False)
    ax.set_title(title)
    return fig


def membership_across_k(table: pd.DataFrame, title: str):
    """Countries (rows) x k (columns); cell colour = cluster id at that k."""
    order = table.sort_values(list(table.columns)).index
    T = table.loc[order]
    fig, ax = plt.subplots(figsize=(0.75 * T.shape[1] + 2.4, 0.26 * len(T) + 1.2))
    for j, k in enumerate(T.columns):
        for i, c in enumerate(T[k]):
            col, mk = _cluster_style(int(c))
            ax.add_patch(Rectangle((j, i), 0.92, 0.88, color=col, alpha=0.9, lw=0))
            ax.text(j + 0.46, i + 0.46, str(int(c)), ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xlim(0, T.shape[1]); ax.set_ylim(len(T), 0)
    ax.set_xticks(np.arange(T.shape[1]) + 0.46, [f"k={k}" for k in T.columns])
    ax.set_yticks(np.arange(len(T)) + 0.44, _iso(T.index), fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title)
    return fig


def cluster_profiles(Z: pd.DataFrame, labels: np.ndarray, codebook: pd.DataFrame, title: str,
                     value_label: str = "Mean z-score"):
    """Mean standardized score per subdimension (rows) and cluster (columns)."""
    lab = pd.Series(labels, index=Z.index)
    by_cluster = Z.groupby(lab).mean().T.join(codebook["subdim_code"])
    M = by_cluster.groupby("subdim_code").mean()
    order = [s for s in SUBDIM_ORDER if s in M.index]
    M = M.loc[order]
    n_sub = codebook.loc[Z.columns].groupby("subdim_code").size().loc[order]
    sizes = lab.value_counts()
    vmax = max(1.0, float(np.nanmax(np.abs(M.to_numpy()))))
    fig, ax = plt.subplots(figsize=(1.25 * M.shape[1] + 3.6, 4.6))
    im = ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.6 * vmax else INK)
    ax.set_xticks(range(M.shape[1]), [f"Cluster {c}\n(n={sizes[c]})" for c in M.columns])
    for t, c in zip(ax.get_xticklabels(), M.columns):
        t.set_color(_cluster_style(int(c))[0]); t.set_fontweight("semibold")
    dim_of = codebook.drop_duplicates("subdim_code").set_index("subdim_code")["dim_code"]
    ax.set_yticks(range(len(order)), [f"{dim_of[s]}.{s}  (n={n_sub[s]})" for s in order])
    for t, s in zip(ax.get_yticklabels(), order):
        t.set_color(DIM_COLOR[dim_of[s]])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label(value_label); cb.outline.set_visible(False)
    ax.set_title(title)
    return fig


# ------------------------------------------------------------- comparisons ---

def procrustes_arrows(ref: pd.DataFrame, other: pd.DataFrame, n_imputed: pd.Series, title: str,
                      ref_name: str = "A (real data)", other_name: str = "I (imputed)"):
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.axhline(0, color=INK_3, lw=0.7, zorder=1); ax.axvline(0, color=INK_3, lw=0.7, zorder=1)
    for c in ref.index:
        ax.annotate("", xy=(other.loc[c, "PC1"], other.loc[c, "PC2"]),
                    xytext=(ref.loc[c, "PC1"], ref.loc[c, "PC2"]),
                    arrowprops=dict(arrowstyle="-|>", color=INK_3, lw=1), zorder=2)
    ax.scatter(ref["PC1"], ref["PC2"], s=40, color=SERIES[0], edgecolor=SURFACE, lw=1.3, zorder=3, label=ref_name)
    sc = ax.scatter(other["PC1"], other["PC2"], s=40, color=SERIES[1], marker="s", edgecolor=SURFACE, lw=1.3,
                    zorder=3, label=other_name)
    for c in ref.index:
        lbl = ISO3[c] + (f" ({int(n_imputed[c])})" if n_imputed[c] else "")
        ax.annotate(lbl, (other.loc[c, "PC1"], other.loc[c, "PC2"]), xytext=(4, 3),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("PC1 (aligned)"); ax.set_ylabel("PC2 (aligned)")
    ax.set_title(title)
    ax.legend(loc="best", title="Arrow: from A to I; (n) imputed cells")
    return fig


def displacement_vs_imputed(disp: pd.DataFrame, n_imputed: pd.Series, title: str):
    """One panel per comparison: displacement vs number of imputed cells."""
    cols = list(disp.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4), sharey=True, squeeze=False)
    for ax, col in zip(axes[0], cols):
        ax.scatter(n_imputed, disp[col].loc[n_imputed.index], s=44, color=SERIES[0],
                   edgecolor=SURFACE, lw=1.3, zorder=3)
        for c in n_imputed.index:
            ax.annotate(ISO3[c], (n_imputed[c], disp.loc[c, col]), xytext=(4, 2),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
        rho = pd.Series(n_imputed).corr(disp[col], method="spearman")
        ax.set_title(f"{col}   Spearman ρ = {rho:.2f}", fontsize=10)
        ax.set_xlabel("Imputed cells in the country's row")
    axes[0][0].set_ylabel("Displacement on aligned PC1–PC2 map")
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    return fig


def ari_matrix(M: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(0.75 * len(M) + 2.6, 0.62 * len(M) + 1.6))
    im = ax.imshow(M.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(len(M)):
        for j in range(len(M)):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(M)), M.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(M)), M.index, fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label("Adjusted Rand Index (1 = identical partitions)"); cb.outline.set_visible(False)
    ax.set_title(title)
    return fig
