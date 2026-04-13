#!/usr/bin/env python3
"""
Mini-projet ACP : analyse descriptive + ACP + rapport HTML stylé.

Fonctionnalités :
- lecture Excel/CSV
- nettoyage automatique des colonnes numériques stockées comme texte
- sélection automatique des variables quantitatives
- imputation par médiane
- étude univariée
- covariance / corrélation
- ACP sur corrélation (par défaut) ou covariance
- vérifications : trace, inertie, somme des valeurs propres
- export Excel + rapport HTML prêt à présenter

Exemple :
    python analyse_acp_assurance_web_final.py DATA.xlsx
    python analyse_acp_assurance_web_final.py DATA.xlsx --sheet "car_insurance_claim.csv"
    python analyse_acp_assurance_web_final.py DATA.xlsx --author "Nom Prénom" --project-title "Mini-projet ACP"
"""
from __future__ import annotations

import argparse
import base64
import io
import re
import sys
import webbrowser
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_EXCLUDE = {"ID"}
KNOWN_CATEGORICAL = {"PARENT1", "MSTATUS", "GENDER", "URBANICITY"}
_NUMERIC_CLEAN_RE = re.compile(r"[^0-9,\.-]+")


def read_table(file_path: str, sheet_name: str | None = None) -> tuple[pd.DataFrame, str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path), "CSV"

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            xls = pd.ExcelFile(path)
        except ImportError as exc:
            raise RuntimeError(
                "Impossible de lire le fichier Excel. Installe :\n"
                "python -m pip install openpyxl xlrd"
            ) from exc

        if sheet_name is not None:
            df = pd.read_excel(path, sheet_name=sheet_name)
            if df.empty:
                raise ValueError(f"La feuille '{sheet_name}' est vide.")
            return df, sheet_name

        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if not df.empty:
                return df, sheet
        raise ValueError("Le fichier Excel ne contient aucune feuille non vide.")

    raise ValueError(f"Format non supporté : {suffix}")


def _normalize_numeric_token(token: object) -> object:
    if token is None:
        return np.nan
    if isinstance(token, float) and pd.isna(token):
        return np.nan
    token = str(token).strip()
    if token == "" or token.lower() in {"nan", "none", "na"}:
        return np.nan
    if "," in token and "." in token:
        return token.replace(",", "")
    if "," in token and "." not in token:
        return token.replace(",", ".")
    return token


def coerce_money_like(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    cleaned = s.str.replace(_NUMERIC_CLEAN_RE, "", regex=True)
    normalized = cleaned.map(_normalize_numeric_token)
    numeric = pd.to_numeric(normalized, errors="coerce")

    original_non_null = int(series.notna().sum())
    converted_non_null = int(numeric.notna().sum())
    conversion_ratio = converted_non_null / max(original_non_null, 1)
    return numeric if conversion_ratio >= 0.70 else series


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.columns:
        out[col] = coerce_money_like(out[col])
    return out


def select_quantitative_columns(
    df: pd.DataFrame,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    exclude_set = set(DEFAULT_EXCLUDE) | KNOWN_CATEGORICAL
    if exclude:
        exclude_set |= set(exclude)

    if include:
        missing = [c for c in include if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes introuvables dans le fichier : {missing}")
        cols = [c for c in include if pd.api.types.is_numeric_dtype(df[c])]
        if len(cols) < 2:
            raise ValueError("--include doit contenir au moins 2 colonnes quantitatives valides.")
        return cols

    cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_set
    ]
    if len(cols) < 2:
        raise ValueError("Impossible de détecter assez de variables quantitatives.")
    return cols


def prepare_quantitative_table(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[cols].copy()
    missing_before = X.isna().sum()
    for col in X.columns:
        median = X[col].median(skipna=True)
        if pd.isna(median):
            raise ValueError(f"La colonne {col} ne contient aucune valeur exploitable.")
        X[col] = X[col].fillna(median)
    return X, missing_before


def descriptive_stats(X: pd.DataFrame, missing_before: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Variable": X.columns,
        "Moyenne": X.mean(axis=0),
        "Ecart_type": X.std(axis=0, ddof=0),
        "Minimum": X.min(axis=0),
        "Mediane": X.median(axis=0),
        "Maximum": X.max(axis=0),
        "Valeurs_manquantes_initiales": missing_before.reindex(X.columns).astype(int).values,
        "Nombre_de_zeros": (X == 0).sum(axis=0).astype(int).values,
    })


def center_reduce(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = X.to_numpy(dtype=float)
    means = values.mean(axis=0)
    centered = values - means
    stds = centered.std(axis=0, ddof=0)
    if np.any(stds == 0):
        zero_var = list(X.columns[np.where(stds == 0)[0]])
        raise ValueError(f"Colonnes à variance nulle : {zero_var}")
    Z = centered / stds
    return values, centered, stds, Z


def covariance_matrix(centered: np.ndarray) -> np.ndarray:
    n = centered.shape[0]
    return (centered.T @ centered) / n


def correlation_matrix(Z: np.ndarray) -> np.ndarray:
    n = Z.shape[0]
    return (Z.T @ Z) / n


def pca_from_matrix(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.where(np.abs(eigvals) < 1e-12, 0.0, eigvals)
    return eigvals, eigvecs


def build_pca_table(eigvals: np.ndarray) -> pd.DataFrame:
    pct = (eigvals / eigvals.sum()) * 100.0
    cum = np.cumsum(pct)
    return pd.DataFrame({
        "Axe": [f"CP{i+1}" for i in range(len(eigvals))],
        "Valeur_propre": eigvals,
        "Pourcentage_inertie": pct,
        "Cumul_pourcentage": cum,
    })


def build_eigenvectors_table(columns: list[str], eigvecs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        eigvecs,
        index=columns,
        columns=[f"u{i+1}" for i in range(eigvecs.shape[1])],
    ).reset_index(names="Variable")


def build_scores(data_matrix: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    return data_matrix @ eigvecs


def top_correlation_pairs(corr: np.ndarray, cols: list[str], n_pairs: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr[i, j])
            rows.append({
                "Variable_1": cols[i],
                "Variable_2": cols[j],
                "Correlation": val,
                "Abs_correlation": abs(val),
                "Sens": "Positive" if val >= 0 else "Negative",
            })
    out = pd.DataFrame(rows).sort_values("Abs_correlation", ascending=False).head(n_pairs)
    return out.drop(columns=["Abs_correlation"]) if not out.empty else out


def quality_summary(original_df: pd.DataFrame, X_df: pd.DataFrame, missing_before: pd.Series, excluded: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Indicateur": "Lignes initiales", "Valeur": int(original_df.shape[0])},
        {"Indicateur": "Colonnes initiales", "Valeur": int(original_df.shape[1])},
        {"Indicateur": "Variables quantitatives retenues", "Valeur": int(X_df.shape[1])},
        {"Indicateur": "Valeurs manquantes sur les variables retenues (avant imputation)", "Valeur": int(missing_before.sum())},
        {"Indicateur": "Colonnes exclues", "Valeur": len(excluded)},
    ])


def matrix_to_dataframe(M: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(M, index=cols, columns=cols).reset_index(names="Variable")


def format_float(x: object) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):,.4f}".replace(",", " ")
    except Exception:
        return str(x)


def dataframe_to_html(df: pd.DataFrame, max_rows: int | None = None, table_id: str | None = None) -> str:
    shown = df.head(max_rows).copy() if max_rows is not None and len(df) > max_rows else df.copy()
    formatted = shown.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(format_float)
    table_html = formatted.to_html(index=False, classes="table", border=0, escape=False)
    if table_id:
        table_html = table_html.replace('<table border="0" class="dataframe table">', f'<table id="{table_id}" border="0" class="dataframe table">')
    extra = ""
    if max_rows is not None and len(df) > max_rows:
        extra = f'<p class="muted">Affichage limité aux {max_rows} premières lignes du tableau ({len(df)} lignes disponibles).</p>'
    return f'<div class="table-wrap">{table_html}</div>{extra}'


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_scree_plot(eigvals: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(1, len(eigvals) + 1)
    ax.plot(x, eigvals, marker="o", linewidth=2)
    ax.set_title("Scree plot des valeurs propres")
    ax.set_xlabel("Composantes principales")
    ax.set_ylabel("Valeur propre")
    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)


def make_cumulative_plot(pca_table: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(1, len(pca_table) + 1)
    ax.plot(x, pca_table["Cumul_pourcentage"], marker="o", linewidth=2)
    ax.axhline(80, linestyle="--", linewidth=1)
    ax.axhline(90, linestyle="--", linewidth=1)
    ax.set_ylim(0, 105)
    ax.set_title("Inertie cumulée")
    ax.set_xlabel("Composantes principales")
    ax.set_ylabel("Cumul (%)")
    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)


def make_scatter_pc12(scores: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(scores[:, 0], scores[:, 1], s=20, alpha=0.75)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Projection des individus sur le plan factoriel (CP1, CP2)")
    ax.set_xlabel("CP1")
    ax.set_ylabel("CP2")
    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)


def make_correlation_heatmap(corr: np.ndarray, cols: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.set_title("Heatmap de la matrice de corrélation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Corrélation")
    return fig_to_base64(fig)


def make_histograms_grid(X: pd.DataFrame) -> str:
    n = X.shape[1]
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, X.columns):
        ax.hist(X[col].dropna().to_numpy(dtype=float), bins=20, edgecolor="black", alpha=0.8)
        ax.set_title(col)
        ax.grid(True, alpha=0.18)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Distributions des variables quantitatives", y=0.995, fontsize=14)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_circle_of_correlations(eigvecs: np.ndarray, eigvals: np.ndarray, cols: list[str]) -> str:
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    fig, ax = plt.subplots(figsize=(7, 7))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1.2)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    for i, col in enumerate(cols):
        x, y = coords[i, 0], coords[i, 1]
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.04, length_includes_head=True, alpha=0.9)
        ax.text(x * 1.08, y * 1.08, col, fontsize=8)
    lim = max(1.1, np.abs(coords).max() * 1.25)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Cercle des corrélations")
    ax.set_xlabel("CP1")
    ax.set_ylabel("CP2")
    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)


def build_summary_cards(
    original_shape: tuple[int, int],
    cols: list[str],
    excluded: list[str],
    missing_before: pd.Series,
    pca_table: pd.DataFrame,
) -> str:
    cp2 = float(pca_table["Cumul_pourcentage"].iloc[min(1, len(pca_table) - 1)])
    cp4 = float(pca_table["Cumul_pourcentage"].iloc[min(3, len(pca_table) - 1)])
    kaiser = int((pca_table["Valeur_propre"] > 1).sum())
    cards = [
        ("Lignes", original_shape[0]),
        ("Colonnes initiales", original_shape[1]),
        ("Variables retenues", len(cols)),
        ("Colonnes exclues", len(excluded)),
        ("Valeurs manquantes imputées", int(missing_before.sum())),
        ("Inertie cumulée CP1-CP2", f"{cp2:.2f}%"),
        ("Inertie cumulée CP1-CP4", f"{cp4:.2f}%"),
        ("Axes retenus (Kaiser)", kaiser),
    ]
    return "".join(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'
        for label, value in cards
    )


def auto_comments(stats: pd.DataFrame, corr_df: pd.DataFrame, pca_table: pd.DataFrame, excluded: list[str]) -> str:
    s = stats.sort_values("Ecart_type", ascending=False)
    most_disp = ", ".join(s["Variable"].head(3).tolist())

    corr_mat = corr_df.set_index("Variable")
    cols = corr_mat.columns.tolist()
    best_pair = None
    best_val = -1.0
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = float(corr_mat.loc[c1, c2])
            if abs(val) > best_val:
                best_val = abs(val)
                best_pair = (c1, c2, val)

    cp2 = float(pca_table["Cumul_pourcentage"].iloc[min(1, len(pca_table) - 1)])
    cp4 = float(pca_table["Cumul_pourcentage"].iloc[min(3, len(pca_table) - 1)])
    kaiser = int((pca_table["Valeur_propre"] > 1).sum())
    excluded_txt = ", ".join(excluded) if excluded else "aucune"
    if best_pair is None:
        pair_txt = "aucune"
    else:
        sens = "positive" if best_pair[2] >= 0 else "négative"
        pair_txt = f"{best_pair[0]} / {best_pair[1]} ({best_pair[2]:.3f}, corrélation {sens})"

    return f"""
    <ul class=\"summary-list\">
      <li><strong>Variables les plus dispersées :</strong> {most_disp}.</li>
      <li><strong>Colonnes exclues automatiquement :</strong> {excluded_txt}.</li>
      <li><strong>Corrélation la plus marquée :</strong> {pair_txt}.</li>
      <li><strong>Inertie expliquée par les 2 premiers axes :</strong> {cp2:.2f}%.</li>
      <li><strong>Inertie expliquée par les 4 premiers axes :</strong> {cp4:.2f}%.</li>
      <li><strong>Critère de Kaiser :</strong> {kaiser} axe(x) avec valeur propre &gt; 1.</li>
    </ul>
    """


def build_html_report(
    dataset_name: str,
    source_sheet: str,
    original_shape: tuple[int, int],
    cols: list[str],
    excluded: list[str],
    quality_df: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
    stats: pd.DataFrame,
    cov_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    pca_table: pd.DataFrame,
    eigvec_df: pd.DataFrame,
    trace_cov: float,
    trace_corr: float,
    inertia_cov: float,
    inertia_corr: float,
    scree_b64: str,
    cum_b64: str,
    scatter_b64: str,
    circle_b64: str,
    heatmap_b64: str,
    hist_b64: str,
    matrix_mode: str,
    author: str,
    project_title: str,
    missing_before: pd.Series,
) -> str:
    comments_html = auto_comments(stats, corr_df, pca_table, excluded)
    metrics_html = build_summary_cards(original_shape, cols, excluded, missing_before, pca_table)

    return f"""<!DOCTYPE html>
<html lang=\"fr\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{project_title} - {dataset_name}</title>
<style>
:root {{
  --bg: #f4f7fb;
  --surface: #ffffff;
  --surface-2: #f8fbff;
  --text: #172033;
  --muted: #5f6b85;
  --line: #dbe3f0;
  --primary: #2f6fed;
  --primary-soft: #e9f0ff;
  --shadow: 0 10px 28px rgba(24, 43, 87, 0.08);
  --radius: 18px;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  font-family: "Segoe UI", Arial, Helvetica, sans-serif;
  background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 220px, var(--bg) 100%);
  color: var(--text);
  line-height: 1.55;
}}
.container {{ max-width: 1360px; margin: 0 auto; padding: 28px 20px 60px; }}
.hero {{
  background: linear-gradient(135deg, #1f57d2 0%, #4d87ff 100%);
  color: white;
  border-radius: 26px;
  padding: 30px 32px;
  box-shadow: var(--shadow);
  margin-bottom: 22px;
}}
.hero h1 {{ margin: 0 0 8px; font-size: 2rem; }}
.hero p {{ margin: 6px 0; opacity: 0.96; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
.badge {{ background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.22); padding: 8px 12px; border-radius: 999px; font-size: 0.95rem; }}
.layout {{ display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 20px; }}
.sidebar {{ position: sticky; top: 18px; align-self: start; }}
.nav-card, .card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}}
.nav-card {{ padding: 18px; }}
.nav-card h3 {{ margin: 0 0 12px; font-size: 1rem; }}
.nav-card a {{ display: block; color: var(--text); text-decoration: none; padding: 8px 10px; border-radius: 10px; }}
.nav-card a:hover {{ background: var(--primary-soft); color: var(--primary); }}
.main > section {{ margin-bottom: 20px; }}
.card {{ padding: 18px 18px 16px; }}
.card h2 {{ margin: 0 0 12px; font-size: 1.25rem; }}
.card h3 {{ margin: 0 0 10px; font-size: 1.05rem; }}
.section-intro {{ color: var(--muted); margin-bottom: 12px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
.metric-card {{ background: var(--surface-2); border: 1px solid var(--line); border-radius: 16px; padding: 16px; }}
.metric-label {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 8px; }}
.metric-value {{ font-size: 1.45rem; font-weight: 700; }}
.grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
.grid-2 .card {{ height: 100%; }}
.grid-graph {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
.table-wrap {{ overflow: auto; max-height: 620px; border: 1px solid var(--line); border-radius: 14px; }}
.table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
.table th, .table td {{ border-bottom: 1px solid #e8eef8; padding: 8px 10px; text-align: right; white-space: nowrap; }}
.table th:first-child, .table td:first-child {{ text-align: left; }}
.table thead th {{ position: sticky; top: 0; z-index: 1; background: #f2f6fd; }}
.muted {{ color: var(--muted); font-size: 0.92rem; }}
img {{ max-width: 100%; display: block; border: 1px solid var(--line); border-radius: 14px; background: white; }}
.code-box {{ background: #0f172a; color: #f8fafc; border-radius: 16px; padding: 14px 16px; overflow: auto; font-family: Consolas, monospace; font-size: 0.94rem; }}
.summary-list {{ margin: 0; padding-left: 18px; }}
.summary-list li {{ margin: 7px 0; }}
.footer-note {{ color: var(--muted); font-size: 0.92rem; }}
@media (max-width: 980px) {{
  .layout {{ grid-template-columns: 1fr; }}
  .sidebar {{ position: static; }}
}}
</style>
</head>
<body>
<div class=\"container\">
  <header class=\"hero\">
    <h1>{project_title}</h1>
    <p><strong>Fichier analysé :</strong> {dataset_name}</p>
    <p><strong>Feuille utilisée :</strong> {source_sheet}</p>
    <p><strong>Auteur :</strong> {author}</p>
    <div class=\"badges\">
      <span class=\"badge\">ACP sur matrice de {matrix_mode}</span>
      <span class=\"badge\">Diviseur n</span>
      <span class=\"badge\">Rapport HTML prêt à présenter</span>
    </div>
  </header>

  <div class=\"layout\">
    <aside class=\"sidebar\">
      <div class=\"nav-card\">
        <h3>Sommaire</h3>
        <a href=\"#resume\">Résumé</a>
        <a href=\"#pretraitement\">Prétraitement</a>
        <a href=\"#univariee\">Étude univariée</a>
        <a href=\"#multivariee\">Étude multivariée</a>
        <a href=\"#acp\">ACP</a>
        <a href=\"#graphiques\">Graphiques</a>
        <a href=\"#verifications\">Vérifications théoriques</a>
        <a href=\"#conclusion\">Conclusion</a>
      </div>
    </aside>

    <main class=\"main\">
      <section id=\"resume\" class=\"card\">
        <h2>Résumé exécutif</h2>
        <p class=\"section-intro\">Vue rapide des informations essentielles à retenir avant de détailler les tableaux et les graphiques.</p>
        <div class=\"metrics\">{metrics_html}</div>
        <div style=\"margin-top:16px\">{comments_html}</div>
      </section>

      <section id=\"pretraitement\" class=\"grid-2\">
        <div class=\"card\">
          <h2>Prétraitement des données</h2>
          <p class=\"section-intro\">Le script lit le fichier, convertit les colonnes numériques stockées comme texte, écarte les colonnes non quantitatives connues et remplace les valeurs manquantes des variables retenues par la médiane.</p>
          {dataframe_to_html(quality_df)}
          <p class=\"muted\">Variables quantitatives retenues : {', '.join(cols)}</p>
          <p class=\"muted\">Colonnes exclues : {', '.join(excluded) if excluded else 'aucune'}</p>
        </div>
        <div class=\"card\">
          <h2>Corrélations les plus fortes</h2>
          <p class=\"section-intro\">Top des couples de variables les plus liés en valeur absolue.</p>
          {dataframe_to_html(top_pairs_df)}
        </div>
      </section>

      <section id=\"univariee\" class=\"grid-2\">
        <div class=\"card\">
          <h2>Étude univariée</h2>
          <p class=\"section-intro\">Moyenne, écart-type, minimum, médiane, maximum, valeurs manquantes initiales et nombre de zéros.</p>
          {dataframe_to_html(stats)}
        </div>
        <div class=\"card\">
          <h2>Distributions des variables</h2>
          <img src=\"data:image/png;base64,{hist_b64}\" alt=\"Distributions des variables\">
        </div>
      </section>

      <section id=\"multivariee\" class=\"grid-2\">
        <div class=\"card\">
          <h2>Matrice de covariance</h2>
          <p class=\"section-intro\">Calculée avec le diviseur <strong>n</strong>, conformément à la série et au corrigé.</p>
          {dataframe_to_html(cov_df)}
        </div>
        <div class=\"card\">
          <h2>Matrice de corrélation</h2>
          <p class=\"section-intro\">Plus adaptée ici car les variables n'ont pas la même échelle.</p>
          {dataframe_to_html(corr_df)}
        </div>
      </section>

      <section id=\"acp\" class=\"grid-2\">
        <div class=\"card\">
          <h2>Valeurs propres et inertie expliquée</h2>
          {dataframe_to_html(pca_table)}
        </div>
        <div class=\"card\">
          <h2>Vecteurs propres</h2>
          {dataframe_to_html(eigvec_df)}
        </div>
      </section>

      <section id=\"graphiques\" class=\"card\">
        <h2>Graphiques principaux</h2>
        <div class=\"grid-graph\">
          <div>
            <h3>Scree plot</h3>
            <img src=\"data:image/png;base64,{scree_b64}\" alt=\"Scree plot\">
          </div>
          <div>
            <h3>Inertie cumulée</h3>
            <img src=\"data:image/png;base64,{cum_b64}\" alt=\"Inertie cumulée\">
          </div>
          <div>
            <h3>Projection des individus sur CP1-CP2</h3>
            <img src=\"data:image/png;base64,{scatter_b64}\" alt=\"Projection des individus\">
          </div>
          <div>
            <h3>Cercle des corrélations</h3>
            <img src=\"data:image/png;base64,{circle_b64}\" alt=\"Cercle des corrélations\">
          </div>
          <div style=\"grid-column: 1 / -1\">
            <h3>Heatmap de corrélation</h3>
            <img src=\"data:image/png;base64,{heatmap_b64}\" alt=\"Heatmap corrélation\">
          </div>
        </div>
      </section>

      <section id=\"verifications\" class=\"card\">
        <h2>Vérifications théoriques</h2>
        <p class=\"section-intro\">Le script vérifie les relations classiques demandées dans le mini-projet.</p>
        <div class=\"grid-2\">
          <div>
            <div class=\"code-box\">V = (1/n) Xcᵀ Xc
R = (1/n) Zᵀ Z
tr(V) = I_G
tr(R) = p
Σ λₖ = tr(R) ou tr(V)</div>
          </div>
          <div>
            <p><strong>Trace de la matrice de covariance :</strong> {format_float(trace_cov)}</p>
            <p><strong>Inertie du nuage centré :</strong> {format_float(inertia_cov)}</p>
            <p><strong>Trace de la matrice de corrélation :</strong> {format_float(trace_corr)}</p>
            <p><strong>Inertie du nuage centré-réduit :</strong> {format_float(inertia_corr)}</p>
            <p><strong>Somme des valeurs propres :</strong> {format_float(pca_table['Valeur_propre'].sum())}</p>
            <p class=\"footer-note\">Le diviseur utilisé est <strong>n</strong>, pas <strong>n-1</strong>, pour rester cohérent avec votre TD.</p>
          </div>
        </div>
      </section>

      <section id=\"conclusion\" class=\"card\">
        <h2>Conclusion</h2>
        <p>
          Le rendu final permet de présenter proprement la chaîne complète de l'analyse : préparation des données,
          statistiques descriptives, étude des liaisons entre variables, ACP et vérifications théoriques.
          La page est directement exploitable en démonstration ou comme annexe visuelle du rapport.
        </p>
      </section>
    </main>
  </div>
</div>
</body>
</html>
"""


def save_excel_report(
    output_excel: str,
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    quantitative_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
    stats: pd.DataFrame,
    cov_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    pca_table: pd.DataFrame,
    eigvec_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        original_df.to_excel(writer, sheet_name="donnees_originales", index=False)
        cleaned_df.to_excel(writer, sheet_name="donnees_nettoyees", index=False)
        quantitative_df.to_excel(writer, sheet_name="variables_quantitatives", index=False)
        quality_df.to_excel(writer, sheet_name="qualite_donnees", index=False)
        top_pairs_df.to_excel(writer, sheet_name="top_correlations", index=False)
        stats.to_excel(writer, sheet_name="etude_univariee", index=False)
        cov_df.to_excel(writer, sheet_name="covariance", index=False)
        corr_df.to_excel(writer, sheet_name="correlation", index=False)
        pca_table.to_excel(writer, sheet_name="valeurs_propres", index=False)
        eigvec_df.to_excel(writer, sheet_name="vecteurs_propres", index=False)
        scores_df.to_excel(writer, sheet_name="scores_individus", index=False)


def run_analysis(args: argparse.Namespace) -> tuple[str, str]:
    original_df, used_sheet = read_table(args.input_file, args.sheet)
    cleaned_df = clean_dataframe(original_df)
    cols = select_quantitative_columns(cleaned_df, args.include, args.exclude)
    excluded = [c for c in cleaned_df.columns if c not in cols]

    X_df, missing_before = prepare_quantitative_table(cleaned_df, cols)
    _, centered, _, Z = center_reduce(X_df)

    cov = covariance_matrix(centered)
    corr = correlation_matrix(Z)

    matrix_mode = args.matrix.lower()
    matrix_for_pca = corr if matrix_mode == "correlation" else cov
    data_for_scores = Z if matrix_mode == "correlation" else centered

    eigvals, eigvecs = pca_from_matrix(matrix_for_pca)
    pca_table = build_pca_table(eigvals)
    eigvec_df = build_eigenvectors_table(cols, eigvecs)
    scores = build_scores(data_for_scores, eigvecs)

    stats = descriptive_stats(X_df, missing_before)
    cov_df = matrix_to_dataframe(cov, cols)
    corr_df = matrix_to_dataframe(corr, cols)
    quality_df = quality_summary(original_df, X_df, missing_before, excluded)
    top_pairs_df = top_correlation_pairs(corr, cols, n_pairs=10)

    scores_df = pd.DataFrame(
        scores[:, : min(5, scores.shape[1])],
        columns=[f"CP{i+1}" for i in range(min(5, scores.shape[1]))],
    )
    scores_df.insert(0, "Individu", np.arange(1, len(scores_df) + 1))

    trace_cov = float(np.trace(cov))
    trace_corr = float(np.trace(corr))
    inertia_cov = float(np.mean(np.sum(centered ** 2, axis=1)))
    inertia_corr = float(np.mean(np.sum(Z ** 2, axis=1)))

    scree_b64 = make_scree_plot(eigvals)
    cum_b64 = make_cumulative_plot(pca_table)
    scatter_b64 = make_scatter_pc12(scores)
    heatmap_b64 = make_correlation_heatmap(corr, cols)
    hist_b64 = make_histograms_grid(X_df)
    circle_b64 = make_circle_of_correlations(eigvecs, eigvals, cols)

    html = build_html_report(
        dataset_name=Path(args.input_file).name,
        source_sheet=used_sheet,
        original_shape=original_df.shape,
        cols=cols,
        excluded=excluded,
        quality_df=quality_df,
        top_pairs_df=top_pairs_df,
        stats=stats,
        cov_df=cov_df,
        corr_df=corr_df,
        pca_table=pca_table,
        eigvec_df=eigvec_df,
        trace_cov=trace_cov,
        trace_corr=trace_corr,
        inertia_cov=inertia_cov,
        inertia_corr=inertia_corr,
        scree_b64=scree_b64,
        cum_b64=cum_b64,
        scatter_b64=scatter_b64,
        circle_b64=circle_b64,
        heatmap_b64=heatmap_b64,
        hist_b64=hist_b64,
        matrix_mode=matrix_mode,
        author=args.author,
        project_title=args.project_title,
        missing_before=missing_before,
    )

    Path(args.output_html).write_text(html, encoding="utf-8")
    save_excel_report(
        args.output_excel,
        original_df,
        cleaned_df,
        X_df,
        quality_df,
        top_pairs_df,
        stats,
        cov_df,
        corr_df,
        pca_table,
        eigvec_df,
        scores_df,
    )
    return args.output_html, args.output_excel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse descriptive + ACP + rapport HTML stylé")
    parser.add_argument("input_file", help="Chemin du fichier .xlsx, .xls ou .csv")
    parser.add_argument("--sheet", help="Nom de la feuille Excel à utiliser")
    parser.add_argument(
        "--matrix",
        choices=["correlation", "covariance"],
        default="correlation",
        help="Matrice utilisée pour l'ACP (défaut: correlation)",
    )
    parser.add_argument("--include", nargs="+", help="Colonnes quantitatives exactes à utiliser")
    parser.add_argument("--exclude", nargs="+", default=[], help="Colonnes à exclure en plus des exclusions automatiques")
    parser.add_argument("--output-html", default="rapport_acp_assurance_final.html", help="Nom du rapport HTML généré")
    parser.add_argument("--output-excel", default="resultats_acp_assurance_final.xlsx", help="Nom du fichier Excel généré")
    parser.add_argument("--project-title", default="Mini-projet ACP - Assurance automobile", help="Titre affiché dans le rapport")
    parser.add_argument("--author", default="Étudiant(e)", help="Nom affiché dans le rapport")
    parser.add_argument("--no-open", action="store_true", help="Ne pas ouvrir automatiquement le rapport dans le navigateur")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output_html, output_excel = run_analysis(args)
    except Exception as exc:
        print(f"\n[ERREUR] {exc}")
        return 1

    print("\nAnalyse terminée avec succès.")
    print(f"- Rapport HTML : {output_html}")
    print(f"- Résultats Excel : {output_excel}")

    if not args.no_open:
        try:
            webbrowser.open(Path(output_html).resolve().as_uri())
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
