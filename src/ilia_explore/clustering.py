"""Clustering countries in the space of retained principal components.

Hierarchical clustering (Ward) is the main tool: with 20 countries its
dendrogram shows the structure at every level. k-means is run alongside as a
check; agreement between the two is measured with the Adjusted Rand Index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

SEED = 20261120


def ward(S: pd.DataFrame) -> np.ndarray:
    return linkage(S.to_numpy(), method="ward")


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Number clusters 1..k from largest to smallest (ties: first appearance)."""
    labels = np.asarray(labels)
    order = pd.Series(labels).value_counts(sort=True)
    first = {lab: i for i, lab in enumerate(dict.fromkeys(labels))}
    ranked = sorted(order.index, key=lambda lab: (-order[lab], first[lab]))
    mapping = {lab: i + 1 for i, lab in enumerate(ranked)}
    return np.array([mapping[lab] for lab in labels])


def ward_labels(S: pd.DataFrame, k: int) -> np.ndarray:
    return relabel_by_size(fcluster(ward(S), t=k, criterion="maxclust"))


def kmeans_labels(S: pd.DataFrame, k: int, seed: int = SEED) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=200, random_state=seed).fit(S.to_numpy())
    return relabel_by_size(km.labels_)


def match_labels(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Relabel ``other`` to maximise overlap with ``ref`` (Hungarian matching)."""
    from scipy.optimize import linear_sum_assignment

    ref, other = np.asarray(ref), np.asarray(other)
    r_lab, o_lab = np.unique(ref), np.unique(other)
    overlap = np.array([[np.sum((other == o) & (ref == r)) for r in r_lab] for o in o_lab])
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {o_lab[i]: r_lab[j] for i, j in zip(rows, cols)}
    spare = iter(range(int(max(r_lab.max(), o_lab.max())) + 1, 10_000))
    return np.array([mapping[x] if x in mapping else mapping.setdefault(x, next(spare)) for x in other])


def gap_statistic(S: pd.DataFrame, ks, n_ref: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Tibshirani's gap statistic with a uniform reference in the PC bounding box.

    ``S`` is already in principal-component coordinates, so sampling the box of
    ``S`` is the recommended rotated-reference variant.
    """
    rng = np.random.default_rng(seed)
    X = S.to_numpy()
    lo, hi = X.min(axis=0), X.max(axis=0)

    def log_wk(data, k):
        if k == 1:
            return np.log(((data - data.mean(axis=0)) ** 2).sum())
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(data)
        return np.log(km.inertia_)

    rows = []
    for k in ks:
        ref = np.array([log_wk(rng.uniform(lo, hi, size=X.shape), k) for _ in range(n_ref)])
        rows.append({"k": k, "gap": ref.mean() - log_wk(X, k),
                     "gap_se": ref.std(ddof=0) * np.sqrt(1 + 1 / n_ref)})
    out = pd.DataFrame(rows).set_index("k")
    # Tibshirani rule: smallest k with gap(k) >= gap(k+1) - se(k+1)
    nxt = out["gap"].shift(-1) - out["gap_se"].shift(-1)
    out["tibshirani_ok"] = out["gap"] >= nxt
    return out


def evaluate_k(S: pd.DataFrame, ks=range(2, 7), seed: int = SEED) -> pd.DataFrame:
    """Silhouette for Ward and k-means, k-means inertia, Ward/k-means agreement, gap."""
    rows = []
    for k in ks:
        w, km = ward_labels(S, k), kmeans_labels(S, k, seed)
        rows.append({
            "k": k,
            "silhouette_ward": silhouette_score(S, w),
            "silhouette_kmeans": silhouette_score(S, km),
            "ari_ward_vs_kmeans": adjusted_rand_score(w, km),
            "min_cluster_size_ward": int(np.bincount(w)[1:].min()),
        })
    out = pd.DataFrame(rows).set_index("k")
    ks = list(ks)
    gap = gap_statistic(S, [1, *ks, max(ks) + 1], seed=seed)
    # keep k=1 so the table also says whether "no clusters" is the better answer
    return out.reindex([1, *ks]).join(gap.loc[[1, *ks]])


def gap_k(ev: pd.DataFrame) -> int:
    """Smallest k (including 1 = no structure) passing Tibshirani's rule."""
    ok = ev.index[ev["tibshirani_ok"].fillna(False).astype(bool)]
    return int(ok.min()) if len(ok) else int(ev.index.max())


def choose_k(ev: pd.DataFrame) -> int:
    """Default k: highest Ward silhouette among solutions without singleton clusters.

    Singletons are allowed only if every k has one (an outlier is then a real
    feature of the data). Notebooks report the full table so the choice can be
    challenged.
    """
    ev = ev.drop(index=1, errors="ignore")
    ok = ev[ev["min_cluster_size_ward"] >= 2]
    pool = ok if len(ok) else ev
    return int(pool["silhouette_ward"].idxmax())


def silhouettes(S: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    return pd.Series(silhouette_samples(S, labels), index=S.index, name="silhouette")


def bootstrap_stability(
    S: pd.DataFrame, k: int, method: str = "ward", n_boot: int = 1000, seed: int = SEED
) -> tuple[pd.Series, pd.DataFrame]:
    """Hennig-style cluster stability and a country co-membership matrix.

    Countries are resampled with replacement; the resample is re-clustered with
    the same method and k. For each original cluster the Jaccard similarity
    with its best-matching bootstrap cluster is recorded (over the countries
    present in the resample). Mean Jaccard > 0.75 is usually read as stable,
    0.6-0.75 as a pattern, < 0.5 as dissolved.

    The co-membership matrix gives, for each pair of countries, the share of
    resamples containing both in which they fell in the same cluster.
    """
    rng = np.random.default_rng(seed)
    fn = ward_labels if method == "ward" else (lambda d, kk: kmeans_labels(d, kk, seed))
    base = fn(S, k)
    n = len(S)
    names = S.index
    jac = {c: [] for c in np.unique(base)}
    together = np.zeros((n, n))
    both = np.zeros((n, n))
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        uniq = np.unique(idx)
        if len(uniq) <= k:
            continue
        lab = fn(S.iloc[idx], k)
        # one label per unique country (duplicates share a point, hence a label)
        lab_u = pd.Series(lab, index=idx).groupby(level=0).first().loc[uniq].to_numpy()
        for c in jac:
            orig = set(uniq[base[uniq] == c])
            if not orig:
                continue
            best = max(len(orig & set(uniq[lab_u == b])) / len(orig | set(uniq[lab_u == b]))
                       for b in np.unique(lab_u))
            jac[c].append(best)
        same = lab_u[:, None] == lab_u[None, :]
        both[np.ix_(uniq, uniq)] += 1
        together[np.ix_(uniq, uniq)] += same
    jaccard = pd.Series({c: float(np.mean(v)) for c, v in jac.items()}, name="mean_jaccard")
    jaccard.index.name = "cluster"
    co = pd.DataFrame(np.divide(together, both, out=np.full_like(together, np.nan), where=both > 0),
                      index=names, columns=names)
    return jaccard, co


def country_stability(co: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Mean co-membership of each country with the other members of its cluster."""
    labels = pd.Series(labels, index=co.index)
    out = {}
    for c in co.index:
        mates = labels.index[(labels == labels[c]) & (labels.index != c)]
        out[c] = float(co.loc[c, mates].mean()) if len(mates) else np.nan
    return pd.Series(out, name="co_membership_with_own_cluster")
