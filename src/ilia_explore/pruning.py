"""Redundancy pruning of subindicators by Spearman correlation.

The correlation matrix is computed on real data only (the masked ``final``
scores), pair by pair over the countries where both subindicators are
observed. One pruning list is derived from it and applied to both the
real-data (A) and the imputed (I) configurations, so their pruned versions
differ only in the data, not in which variables were kept.

Rule
----
1. Distance between two subindicators is ``1 - |rho|``.
2. Complete-linkage clustering cut at ``1 - threshold`` forms groups in which
   *every* pair has ``|rho| >= threshold``.
3. One representative per group is kept:
   (a) the one with the most observed countries -- so A-pruned does not lose
       a whole group because its representative has a gap;
   (b) ties broken by the lowest mean ``|rho|`` with subindicators outside the
       group (the least redundant with everything else);
   (c) remaining ties broken by code order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

MIN_OVERLAP = 10  # countries both subindicators must share to get a correlation


def spearman_pairwise(X: pd.DataFrame, min_periods: int = MIN_OVERLAP) -> pd.DataFrame:
    """Spearman correlation over pairwise-complete countries (X: countries x codes)."""
    return X.corr(method="spearman", min_periods=min_periods)


def redundancy_groups(corr: pd.DataFrame, threshold: float) -> pd.Series:
    """Group id per code; pairs without a correlation count as unrelated."""
    dist = (1 - corr.abs()).fillna(1.0).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0.0, 1.0)
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    return pd.Series(labels, index=corr.index, name="group")


def prune(corr: pd.DataFrame, n_observed: pd.Series, threshold: float) -> pd.DataFrame:
    """Per-subindicator pruning table: group, group size, representative, kept."""
    groups = redundancy_groups(corr, threshold)
    absc = corr.abs()
    rows = []
    for g, members in groups.groupby(groups).groups.items():
        members = list(members)
        outside = [c for c in corr.index if c not in members]
        mean_out = absc.loc[members, outside].mean(axis=1) if outside else pd.Series(0.0, index=members)
        ranking = pd.DataFrame(
            {"n_obs": n_observed[members].to_numpy(), "mean_out": mean_out[members].to_numpy(),
             "code_key": members},
            index=members,
        ).sort_values(["n_obs", "mean_out", "code_key"], ascending=[False, True, True])
        rep = ranking.index[0]
        for c in members:
            within = absc.loc[c, [m for m in members if m != c]]
            rows.append({
                "code": c,
                "group": g,
                "group_size": len(members),
                "representative": rep,
                "kept": c == rep,
                "min_abs_rho_in_group": within.min() if len(within) else np.nan,
                "mean_abs_rho_outside": mean_out[c],
            })
    out = pd.DataFrame(rows).set_index("code").loc[corr.index]
    # renumber groups by first appearance so ids are stable and readable
    first = {g: i + 1 for i, g in enumerate(dict.fromkeys(out["group"]))}
    out["group"] = out["group"].map(first)
    return out


def threshold_summary(
    corr: pd.DataFrame,
    n_observed: pd.Series,
    dim_code: pd.Series,
    thresholds=(0.8, 0.85, 0.9),
    complete: pd.Series | None = None,
) -> pd.DataFrame:
    """How many subindicators each threshold keeps, overall and per dimension.

    ``complete`` (bool per code) adds how many kept subindicators have no gaps,
    i.e. how wide A-pruned would be.
    """
    rows = []
    for t in thresholds:
        tab = prune(corr, n_observed, t)
        kept = tab["kept"]
        row = {"threshold": t, "kept_total": int(kept.sum()), "dropped_total": int((~kept).sum()),
               "groups_merged": int((tab["group_size"] > 1).groupby(tab["group"]).first().sum())}
        for d in dim_code.unique():
            in_d = dim_code == d
            row[f"kept_{d}"] = int((kept & in_d).sum())
            row[f"of_{d}"] = int(in_d.sum())
        if complete is not None:
            row["kept_without_gaps"] = int((kept & complete).sum())
        rows.append(row)
    return pd.DataFrame(rows)
