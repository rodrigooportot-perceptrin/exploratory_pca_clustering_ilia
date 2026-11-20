"""One function that runs the whole PCA + clustering chain for a configuration.

A configuration is a data source (A = real data only, I = MICE-imputed) and a
variable set (full, pruned). Each is analysed in two views:

* ``level``   -- standardized scores; countries differ mostly by overall level.
* ``profile`` -- each country's mean z-score removed; what remains is its
  pattern of relative strengths and weaknesses.

Every step is seeded, so notebooks that call ``run`` get identical results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import clustering as cl
from . import pca as pc
from .data import RESULTS, load_processed

CONFIGS = ["A-full", "A-pruned", "I-full", "I-pruned"]
MODES = ["level", "profile"]
KS = range(2, 7)
N_BOOT = 1000
N_PERM = 1000


def tag(config: str, mode: str) -> str:
    return f"{config}_{mode}"


@dataclass
class Run:
    config: str
    mode: str
    X: pd.DataFrame                 # countries x subindicators, final scores
    Z: pd.DataFrame                 # standardized (level) or level-removed (profile)
    pca: pc.PCAResult
    pa: pd.DataFrame                # parallel analysis
    k_pc: int                       # PCs used for clustering
    S: pd.DataFrame                 # clustering space: first k_pc PC scores
    evaluation: pd.DataFrame        # diagnostics per k
    k: int                          # chosen number of clusters
    linkage: np.ndarray
    ward: np.ndarray                # Ward labels at k
    kmeans: np.ndarray              # k-means labels at k, matched to Ward's numbering
    silhouette: pd.Series
    jaccard: pd.Series | None = None
    comembership: pd.DataFrame | None = None
    stability: pd.Series | None = None
    membership_by_k: pd.DataFrame | None = None

    @property
    def tag(self) -> str:
        return tag(self.config, self.mode)

    def labels_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ward": self.ward, "kmeans": self.kmeans, "silhouette": self.silhouette,
            "co_membership_with_own_cluster": self.stability,
        }, index=self.X.index)


def prepare(X: pd.DataFrame, mode: str) -> pd.DataFrame:
    Z = pc.standardize(X)
    return pc.remove_level(Z) if mode == "profile" else Z


def anchored_membership(S: pd.DataFrame, labels_at_k: np.ndarray, k: int) -> pd.DataFrame:
    """Ward labels for every k in KS, numbered consistently with the chosen k.

    Moving away from the chosen k one step at a time, each solution is
    relabelled to overlap best with its neighbour, so a group keeps its number
    (and colour) while it persists and a split creates a new number.
    """
    out = {k: np.asarray(labels_at_k)}
    for kk in range(k + 1, max(KS) + 1):
        out[kk] = cl.match_labels(out[kk - 1], cl.ward_labels(S, kk))
    for kk in range(k - 1, min(KS) - 1, -1):
        out[kk] = cl.match_labels(out[kk + 1], cl.ward_labels(S, kk))
    return pd.DataFrame({kk: out[kk] for kk in KS}, index=S.index)


def run(config: str, mode: str, k: int | None = None, bootstrap: bool = True) -> Run:
    """Full chain. ``k=None`` picks k with ``clustering.choose_k``."""
    X = load_processed(config)
    Z = prepare(X, mode)
    res = pc.run_pca(Z)
    pa = pc.parallel_analysis(Z, n_iter=N_PERM)
    k_pc = pc.n_retained(pa)
    S = res.scores.iloc[:, :k_pc]
    ev = cl.evaluate_k(S, KS)
    k = k or cl.choose_k(ev)
    w = cl.ward_labels(S, k)
    km = cl.match_labels(w, cl.kmeans_labels(S, k))
    r = Run(config, mode, X, Z, res, pa, k_pc, S, ev, k, cl.ward(S), w, km, cl.silhouettes(S, w))
    r.membership_by_k = anchored_membership(S, w, k)
    if bootstrap:
        r.jaccard, r.comembership = cl.bootstrap_stability(S, k, "ward", N_BOOT)
        r.stability = cl.country_stability(r.comembership, w)
    return r


def save(r: Run) -> None:
    """Write every table behind the figures to results/<config>_<mode>/."""
    out = RESULTS / r.tag
    out.mkdir(parents=True, exist_ok=True)
    k = r.pca.n_components
    pd.DataFrame({"eigenvalue": r.pca.eigenvalues, "explained": r.pca.explained,
                  "cumulative": r.pca.explained.cumsum()}).iloc[:k].to_csv(out / "pca_variance.csv")
    r.pa.to_csv(out / "parallel_analysis.csv")
    r.pca.scores.iloc[:, :k].to_csv(out / "pca_scores.csv")
    r.pca.loadings.iloc[:, :k].to_csv(out / "pca_loadings.csv")
    r.evaluation.to_csv(out / "k_evaluation.csv")
    r.labels_frame().to_csv(out / "cluster_labels.csv")
    r.membership_by_k.to_csv(out / "ward_membership_by_k.csv")
    if r.jaccard is not None:
        r.jaccard.to_csv(out / "cluster_stability_jaccard.csv")
        r.comembership.to_csv(out / "bootstrap_comembership.csv")
    pd.Series({"n_countries": len(r.X), "n_subindicators": r.X.shape[1], "pcs_retained": r.k_pc,
               "k_chosen": r.k, "k_gap_rule": cl.gap_k(r.evaluation),
               "silhouette_at_k": float(r.silhouette.mean())}, name="value").to_csv(out / "summary.csv")
