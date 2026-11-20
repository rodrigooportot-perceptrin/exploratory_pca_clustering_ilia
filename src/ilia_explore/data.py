"""Loading the ILIA workbook and building the analysis matrices.

Conventions
-----------
* ``wide`` frames: one row per subindicator (indexed by ``code``), one column
  per country -- the layout of the workbook.
* ``matrix`` frames: one row per country, one column per subindicator code --
  the layout PCA and clustering expect (observations x variables).

The ``final`` sheet has no gaps because missing values were imputed (MICE with
Bayesian ridge and auxiliary predictors). A cell is *imputed* exactly when the
same cell in the ``bruto`` sheet is empty, so masking ``final`` with the
``bruto`` gaps recovers the real-data-only scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_XLSX = ROOT / "data" / "raw" / "ilia_2026_resultados_consolidados.xlsx"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

META = ["dim", "subdim", "indicador", "subindicador"]

DIM_CODE = {"FH": "FH", "Gobernanza": "GOB", "I+D+A": "IDA"}
DIM_LABEL = {"FH": "Enabling factors", "GOB": "Governance", "IDA": "R&D + adoption"}
SUBDIM_CODE = {
    "Datos": "DAT",
    "Infraestructura": "INF",
    "Talento Humano": "TAL",
    "IA Ética, Responsable y Segura": "ETI",
    "Regulación": "REG",
    "Vinculación Internacional": "VIN",
    "Visión e Institucionalidad": "VIS",
    "Adopción": "ADO",
    "I+D": "IDI",
    "Investigación": "INV",
}

ISO3 = {
    "Argentina": "ARG", "Bolivia": "BOL", "Brasil": "BRA", "Chile": "CHL",
    "Colombia": "COL", "Costa Rica": "CRI", "Cuba": "CUB", "Ecuador": "ECU",
    "El Salvador": "SLV", "Guatemala": "GTM", "Honduras": "HND",
    "Jamaica": "JAM", "México": "MEX", "Panamá": "PAN", "Paraguay": "PRY",
    "Perú": "PER", "República Dominicana": "DOM", "Trinidad y Tobago": "TTO",
    "Uruguay": "URY", "Venezuela": "VEN",
}


def load_sheets(path: Path = RAW_XLSX) -> dict[str, pd.DataFrame]:
    """Read the three sheets and check they describe the same subindicators."""
    sheets = pd.read_excel(path, sheet_name=None)
    expected = {"bruto", "normalizado", "final"}
    if set(sheets) != expected:
        raise ValueError(f"Expected sheets {expected}, found {set(sheets)}")
    ref = sheets["bruto"]
    for name, df in sheets.items():
        if not df[META].equals(ref[META]) or list(df.columns) != list(ref.columns):
            raise ValueError(f"Sheet '{name}' is not aligned with 'bruto'")
    return sheets


def countries(sheets: dict[str, pd.DataFrame]) -> list[str]:
    return [c for c in sheets["bruto"].columns if c not in META]


def build_codebook(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per subindicator with a short stable code and data-quality facts.

    Codes are ``<DIM>.<SUBDIM>.<nn>``, numbered in workbook order within each
    subdimension, e.g. ``FH.INF.01``.
    """
    bruto = sheets["bruto"]
    cols = countries(sheets)
    cb = bruto[META].copy()
    cb["dim_code"] = cb["dim"].map(DIM_CODE)
    cb["subdim_code"] = cb["subdim"].map(SUBDIM_CODE)
    if cb[["dim_code", "subdim_code"]].isna().any().any():
        raise ValueError("Unknown dimension or subdimension name in workbook")
    nn = cb.groupby(["dim_code", "subdim_code"]).cumcount() + 1
    cb["code"] = cb["dim_code"] + "." + cb["subdim_code"] + "." + nn.map("{:02d}".format)
    raw = bruto[cols]
    cb["n_observed"] = raw.notna().sum(axis=1)
    cb["n_imputed"] = raw.isna().sum(axis=1)
    cb["n_unique_observed"] = raw.nunique(axis=1)
    cb["categorical"] = cb["n_unique_observed"] <= 3
    cb["imputed_countries"] = raw.isna().apply(
        lambda r: ", ".join(ISO3[c] for c in cols if r[c]), axis=1
    )
    order = ["code", "dim_code", "subdim_code"] + META + [
        "n_observed", "n_imputed", "n_unique_observed", "categorical", "imputed_countries",
    ]
    return cb[order].set_index("code")


def wide_final(sheets: dict[str, pd.DataFrame], codebook: pd.DataFrame) -> pd.DataFrame:
    """The ``final`` sheet (imputed, no gaps), indexed by subindicator code."""
    w = sheets["final"][countries(sheets)].copy()
    w.index = codebook.index
    return w


def wide_masked(sheets: dict[str, pd.DataFrame], codebook: pd.DataFrame) -> pd.DataFrame:
    """``final`` scores with every imputed cell set to NaN (real data only)."""
    cols = countries(sheets)
    w = sheets["final"][cols].where(sheets["bruto"][cols].notna())
    w.index = codebook.index
    return w


def to_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """Transpose to countries x subindicators."""
    m = wide.T
    m.index.name = "country"
    m.columns.name = "code"
    return m


def matrix_A(masked: pd.DataFrame) -> pd.DataFrame:
    """Configuration A: real data only, keeping subindicators with no gaps."""
    return to_matrix(masked.dropna(axis=0, how="any"))


def matrix_I(final: pd.DataFrame) -> pd.DataFrame:
    """Configuration I: the full MICE-imputed ``final`` sheet."""
    return to_matrix(final)


def imputed_per_country(masked: pd.DataFrame) -> pd.Series:
    return masked.isna().sum(axis=0).rename("n_imputed")


def load_processed(name: str) -> pd.DataFrame:
    """Read a matrix written by notebook 01 (countries x codes)."""
    return pd.read_csv(PROCESSED / f"{name}.csv", index_col=0)


def load_codebook() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "codebook.csv", index_col=0)


def iso(index) -> list[str]:
    return [ISO3.get(c, c) for c in index]


def check_mask(final: pd.DataFrame, masked: pd.DataFrame) -> None:
    """Every observed cell must equal the corresponding final value."""
    obs = masked.notna()
    if not np.allclose(final[obs].fillna(0), masked.fillna(0)):
        raise AssertionError("Masked values differ from final values")
