"""PCA on standardized subindicators, with helpers for small-n data.

With 20 countries and up to 126 subindicators there are more variables than
observations, so at most 19 components carry variance. Horn's parallel
analysis decides how many are distinguishable from noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.decomposition import PCA


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """z-scores per subindicator (population sd, so each column has variance 1)."""
    sd = X.std(axis=0, ddof=0)
    if (sd == 0).any():
        raise ValueError(f"Constant subindicators: {list(sd[sd == 0].index)}")
    return (X - X.mean(axis=0)) / sd


def remove_level(Z: pd.DataFrame) -> pd.DataFrame:
    """Profile view: subtract each country's mean z-score.

    What remains is the country's pattern of relative strengths and weaknesses,
    independent of its overall level.
    """
    return Z.sub(Z.mean(axis=1), axis=0)


@dataclass
class PCAResult:
    scores: pd.DataFrame          # countries x PCs
    loadings: pd.DataFrame        # codes x PCs; correlation of each variable with each PC
    eigenvalues: pd.Series
    explained: pd.Series          # share of total variance
    n_components: int             # components with non-negligible variance

    def top_loadings(self, pc: str, n: int = 10) -> pd.DataFrame:
        s = self.loadings[pc]
        idx = s.abs().sort_values(ascending=False).index[:n]
        return s.loc[idx].to_frame("loading")


def run_pca(Z: pd.DataFrame) -> PCAResult:
    """PCA of an already standardized (or profile) matrix.

    Signs are fixed for reproducibility: PC1 is oriented so its mean loading is
    positive (high score = high on most subindicators); other PCs so their
    largest absolute loading is positive.
    """
    n, p = Z.shape
    k = min(n - 1, p)
    model = PCA(n_components=k, svd_solver="full").fit(Z.to_numpy())
    comps = model.components_.copy()
    for i in range(k):
        ref = comps[i].mean() if i == 0 else comps[i][np.argmax(np.abs(comps[i]))]
        if ref < 0:
            comps[i] *= -1
    names = [f"PC{i + 1}" for i in range(k)]
    scores = pd.DataFrame(Z.to_numpy() @ comps.T, index=Z.index, columns=names)
    eig = pd.Series(model.explained_variance_, index=names, name="eigenvalue")
    # population-variance eigenvalues so loadings are correlations with unit-variance inputs
    eig_pop = eig * (n - 1) / n
    col_sd = Z.std(axis=0, ddof=0).replace(0, np.nan)
    loadings = pd.DataFrame(comps.T * np.sqrt(eig_pop.to_numpy()), index=Z.columns, columns=names)
    loadings = loadings.div(col_sd, axis=0)
    explained = pd.Series(model.explained_variance_ratio_, index=names, name="explained")
    return PCAResult(scores, loadings, eig, explained, int((explained > 1e-10).sum()))


def parallel_analysis(
    Z: pd.DataFrame, n_iter: int = 1000, quantile: float = 0.95, seed: int = 0
) -> pd.DataFrame:
    """Horn's parallel analysis by permuting each column independently.

    Permuting keeps every subindicator's distribution (including categorical
    ones) but destroys correlations. A component is retained while its
    eigenvalue exceeds the chosen quantile of the permuted eigenvalues.
    """
    rng = np.random.default_rng(seed)
    n, p = Z.shape
    k = min(n - 1, p)
    A = Z.to_numpy()
    null = np.empty((n_iter, k))
    for b in range(n_iter):
        P = np.column_stack([rng.permutation(A[:, j]) for j in range(p)])
        P = P - P.mean(axis=0)
        s = np.linalg.svd(P, compute_uv=False)[:k]
        null[b] = s**2 / (n - 1)
    real = run_pca(Z).eigenvalues.to_numpy()
    names = [f"PC{i + 1}" for i in range(k)]
    out = pd.DataFrame({
        "eigenvalue": real,
        "null_mean": null.mean(axis=0),
        f"null_q{int(quantile * 100)}": np.quantile(null, quantile, axis=0),
    }, index=names)
    above = out["eigenvalue"] > out[f"null_q{int(quantile * 100)}"]
    # retain the leading run of components above the null, not isolated later ones
    out["retain"] = above.cumprod().astype(bool)
    return out


def n_retained(pa: pd.DataFrame, minimum: int = 2) -> int:
    """Components to keep: parallel analysis, but never fewer than ``minimum``."""
    return max(int(pa["retain"].sum()), minimum)


def compare_maps(
    ref: pd.DataFrame, other: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Align ``other``'s first k PCs onto ``ref``'s with Procrustes.

    Both maps are centred and scaled to unit size, then ``other`` is rotated /
    reflected to best match ``ref``. Returns the two aligned maps, and the
    disparity (sum of squared differences; 0 = identical, 1 = unrelated).
    """
    cols = [f"PC{i + 1}" for i in range(k)]
    common = ref.index.intersection(other.index)
    a, b, disparity = procrustes(ref.loc[common, cols].to_numpy(), other.loc[common, cols].to_numpy())
    return (pd.DataFrame(a, index=common, columns=cols),
            pd.DataFrame(b, index=common, columns=cols),
            float(disparity))


def displacement(ref_aligned: pd.DataFrame, other_aligned: pd.DataFrame) -> pd.Series:
    """Euclidean distance each country moves between two aligned maps."""
    return np.sqrt(((ref_aligned - other_aligned) ** 2).sum(axis=1)).rename("displacement")


def loading_congruence(L1: pd.DataFrame, L2: pd.DataFrame, k: int) -> pd.DataFrame:
    """Tucker's congruence between matching PCs over shared subindicators.

    Values above ~0.95 mean the two PCs are essentially the same; 0.85-0.95
    fair similarity; below 0.85 they describe different things. The absolute
    value is reported since PC signs are arbitrary.
    """
    common = L1.index.intersection(L2.index)
    rows = {}
    for i in range(k):
        pc = f"PC{i + 1}"
        x, y = L1.loc[common, pc], L2.loc[common, pc]
        rows[pc] = abs(float((x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum())))
    return pd.Series(rows, name="tucker_congruence").to_frame()
