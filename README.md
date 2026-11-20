# exploratory_pca_clustering_ilia

Exploratory PCA and clustering of the **ILIA 2026 subindicators** (126 subindicators × 20 Latin American and Caribbean countries).
The study looks at how countries behave on the raw subindicators, **before any aggregation into the index**:
which directions separate them, whether they form groups, and how much of that depends on imputed data.

## Questions

1. What are the main directions of variation across countries, and which subindicators drive them?
2. Do countries form distinct groups (typologies), and how stable are those groups?
3. How much do the results change when imputed values are left out, or when redundant subindicators are removed?

## Data

`data/raw/ilia_2026_resultados_consolidados.xlsx` has three aligned sheets: `bruto` (raw values, with empty cells where no data exists),
`normalizado`, and `final` (0–100 scores, with the empty cells imputed using MICE with Bayesian ridge and auxiliary predictors such as GDP,
population and 10-year GDP growth). A `final` value is imputed exactly when the same `bruto` cell is empty: 141 cells (5.6%).
All 20 countries are kept in every analysis.

## Configurations

|  | **all subindicators** | **pruned** (redundancy removed at Spearman \|ρ\| ≥ 0.9) |
|---|---|---|
| **A: real data only** (subindicators with no gaps) | A-full: 86 | A-pruned: 70 |
| **I: MICE-imputed `final` sheet** | I-full: 126 | I-pruned: 106 |

Each configuration is analysed in two views:

* **level**: standardized scores;
* **profile**: each country's mean z-score removed, leaving its pattern of relative strengths and weaknesses.

## Method in brief

* **Pruning:** Spearman correlation on real data only (pairwise). Complete-linkage groups in which every pair has |ρ| ≥ 0.9; one subindicator is kept per group, preferring the most complete one. One list is applied to both A and I. Thresholds 0.85 and 0.8 are reported too.
* **PCA** on z-scores. Horn's parallel analysis (1,000 column permutations, 95th percentile) decides how many components are above noise.
* **Clustering** on the retained component scores. Ward hierarchical clustering is the main method, with k-means as a check (ARI). k is chosen by silhouette among solutions without single-country clusters; the gap statistic, including k = 1, tests whether clusters exist at all.
* **Stability:** 1,000 bootstrap resamples of countries give a Jaccard score per cluster and a country co-membership matrix.
* **Comparisons:** Procrustes-aligned maps (how far each country moves), Tucker congruence of loadings, and ARI between partitions.

## Main findings

* **The gaps are not random.** Countries with more imputed cells score lower on what is observed (Spearman ρ = −0.79). Cuba has 34 imputed cells, TTO 17, JAM 15, VEN 14.
* **PC1 is overall level** (36–39% of variance; correlation 0.997 with a country's mean z-score). Only PC1 and PC2 stand out from noise.
* **PC2 contrasts national AI strategy and institutions with infrastructure, data, talent and adoption.** Cuba, El Salvador and the Dominican Republic lean to strategy; Trinidad and Tobago, Paraguay, Panama and Chile lean to capabilities.
* **The robust grouping is a split by level:** upper {ARG, BRA, CHL, COL, CRI, ECU, MEX, PER, URY} and lower {BOL, CUB, SLV, GTM, HND, JAM, PRY, TTO, VEN}. The Dominican Republic and Panama are borderline.
* **A-full also supports four groups:** leaders (BRA, CHL, URY), upper-middle, lower, and a *strategy ahead of capabilities* group (CUB, SLV, DOM), which is the least stable.
* **There are no distinct profile types.** Once level is removed the data form a gradient along the strategy–capabilities axis; the gap statistic prefers k = 1.
* **Imputation and pruning barely change the level picture** (Procrustes disparity 0.02–0.035; at k = 2, A-full and I-full give identical groups). How far a country moves is unrelated to how many of its cells are imputed. The profile view is more sensitive to which subindicators are included.

Each notebook ends with a *Reading the results* section with the details.

## Repository layout

```
data/raw/            original workbook (unchanged)
data/processed/      codebook, masked and imputed scores, the four configuration matrices
src/ilia_explore/    data.py · pruning.py · pca.py · clustering.py · pipeline.py · plots.py
notebooks/           01_data_and_missingness · 02_correlation_pruning · 03_pca · 04_clustering · 05_real_vs_imputed
figures/             every figure as PNG, prefixed by notebook number and configuration
results/             tables behind every figure; one folder per configuration and view, plus pruning/ and comparisons/
```

Subindicators are referred to by short codes `<DIM>.<SUBDIM>.<nn>` (e.g. `FH.INF.19`); `data/processed/codebook.csv` maps them to names.
Dimensions: FH enabling factors, GOB governance, IDA R&D and adoption.

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd notebooks
for f in 0*.ipynb; do ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace "$f"; done
```

Notebooks must run in order (each reads what the previous one wrote). All random steps are seeded, so results are identical between runs.
Notebook 04 takes about 4 minutes because of the bootstrap.
