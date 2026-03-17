"""
KS-Test (Kolmogorov-Smirnov): Ground Truth vs. Simulierte 10k Cases
====================================================================

Führt für jedes numerische Attribut einen Zweistichproben-KS-Test durch
(scipy.stats.ks_2samp) und berechnet für kategorische Attribute die
Total Variation Distance (TVD).

Ausgabe:
  - Tabelle mit KS-Statistik, p-Wert und Bewertung pro Attribut
  - Balkendiagramm der KS-Statistiken → ks_statistics.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Pfade ────────────────────────────────────────────────────────────────────

SIM_PATH = Path(
    "/teamspace/studios/this_studio/process-simulation-engine/"
    "Instance Spawn Rate/Advanced/case_attribute_prediction/simulated_10k_cases.csv"
)

GT_POSSIBLE_PATHS = [
    Path("/teamspace/studios/this_studio/process-simulation-engine/integration/output/ground_truth_log.csv"),
    Path(__file__).parent.parent / "ground_truth_log.csv",
]

OUTPUT_PNG = Path(__file__).parent / "ks_statistics.png"

EXCLUDE_COLS = {
    "case:concept:name", "concept:name", "lifecycle:transition",
    "time:timestamp", "time:start", "time:end",
    "org:resource", "org:group", "org:role",
}

# KS-Bewertungsschwellen
THRESHOLDS = [
    (0.05, "✅ Sehr gut"),
    (0.10, "🟢 Gut"),
    (0.20, "🟡 Akzeptabel"),
    (0.30, "🟠 Mäßig"),
    (1.00, "🔴 Schlecht"),
]


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def rating(ks: float) -> str:
    for threshold, label in THRESHOLDS:
        if ks <= threshold:
            return label
    return "🔴 Schlecht"


def load_ground_truth() -> pd.DataFrame:
    for p in GT_POSSIBLE_PATHS:
        if p.exists():
            print(f"  Lade Ground Truth von: {p}")
            df = pd.read_csv(p)
            print(f"  ✓ {len(df):,} Zeilen, {df.shape[1]} Spalten")
            return df
    raise FileNotFoundError(
        "Ground Truth nicht gefunden. Geprüfte Pfade:\n"
        + "\n".join(f"  - {p}" for p in GT_POSSIBLE_PATHS)
    )


def to_case_level(df: pd.DataFrame, col: str) -> pd.Series:
    """Reduziert Ereignis-Log auf Case-Ebene (erster Wert pro Case)."""
    if "case:concept:name" in df.columns:
        return df.groupby("case:concept:name")[col].first().dropna()
    return df[col].dropna()


# ─── Hauptlogik ───────────────────────────────────────────────────────────────

def run_ks_tests(df_gt: pd.DataFrame, df_sim: pd.DataFrame) -> pd.DataFrame:
    """
    Führt KS-Tests (numerisch) bzw. TVD (kategorisch) für alle
    gemeinsamen Attribute durch.

    Returns:
        DataFrame mit Spalten:
            Attribut | Typ | KS / TVD | p-Wert | n_GT | n_Sim | Bewertung
    """
    # Gemeinsame Attribute (exkl. Metadaten)
    gt_cols  = {c for c in df_gt.columns  if c not in EXCLUDE_COLS}
    sim_cols = {c for c in df_sim.columns if c not in EXCLUDE_COLS}
    common   = sorted(gt_cols & sim_cols)

    rows = []
    for col in common:
        # Ground-Truth auf Case-Ebene reduzieren
        gt_series  = to_case_level(df_gt, col)
        sim_series = df_sim[col].dropna()

        if len(gt_series) == 0 or len(sim_series) == 0:
            print(f"  ⚠ {col}: keine Daten, übersprungen.")
            continue

        # Typ bestimmen
        is_bool = (
            gt_series.dtype == bool
            or sim_series.dtype == bool
            or set(gt_series.dropna().unique()) <= {True, False, "True", "False", 0, 1}
        )
        is_cat = gt_series.dtype == object or sim_series.dtype == object or is_bool

        if is_cat:
            # ── Kategorisch: TVD ──────────────────────────────────────────────
            gt_s  = gt_series.astype(str)
            sim_s = sim_series.astype(str)

            all_cats = set(gt_s.unique()) | set(sim_s.unique())
            gt_prob  = gt_s.value_counts(normalize=True)
            sim_prob = sim_s.value_counts(normalize=True)

            tvd = 0.5 * sum(
                abs(gt_prob.get(c, 0.0) - sim_prob.get(c, 0.0))
                for c in all_cats
            )
            rows.append({
                "Attribut":  col,
                "Typ":       "kategorisch",
                "KS / TVD":  round(tvd, 4),
                "p-Wert":    "–",
                "n_GT":      len(gt_s),
                "n_Sim":     len(sim_s),
                "Bewertung": rating(tvd),
            })

        else:
            # ── Numerisch: KS-Test ────────────────────────────────────────────
            gt_num  = pd.to_numeric(gt_series,  errors="coerce").dropna().to_numpy()
            sim_num = pd.to_numeric(sim_series, errors="coerce").dropna().to_numpy()

            if len(gt_num) < 2 or len(sim_num) < 2:
                print(f"  ⚠ {col}: zu wenige numerische Werte, übersprungen.")
                continue

            ks_result = stats.ks_2samp(gt_num, sim_num, method="auto")
            rows.append({
                "Attribut":  col,
                "Typ":       "numerisch",
                "KS / TVD":  round(ks_result.statistic, 4),
                "p-Wert":    f"{ks_result.pvalue:.2e}",
                "n_GT":      len(gt_num),
                "n_Sim":     len(sim_num),
                "Bewertung": rating(ks_result.statistic),
            })

    result_df = pd.DataFrame(rows).sort_values("KS / TVD", ascending=False).reset_index(drop=True)
    return result_df


def plot_ks_bars(result_df: pd.DataFrame, save_path: Path) -> None:
    """Creates bar chart of KS/TVD statistics."""
    df_plot = result_df.sort_values("KS / TVD", ascending=False)
    labels  = df_plot["Attribut"].tolist()
    values  = df_plot["KS / TVD"].tolist()
    types   = df_plot["Typ"].tolist()

    colors = []
    for v in values:
        if v <= 0.05:
            colors.append("#4CAF50")   # green
        elif v <= 0.10:
            colors.append("#8BC34A")   # light green
        elif v <= 0.20:
            colors.append("#FFC107")   # yellow
        elif v <= 0.30:
            colors.append("#FF9800")   # orange
        else:
            colors.append("#F44336")   # red

    fig_w = max(8, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    x_pos = np.arange(len(labels))
    bars  = ax.bar(x_pos, values, color=colors, edgecolor="white", linewidth=0.6, alpha=0.88)

    # Value labels
    for bar, v, t in zip(bars, values, types):
        label = f"{v:.3f}\n({'TVD' if t == 'kategorisch' else 'KS'})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            label, va="bottom", ha="center", fontsize=12,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=14, rotation=45, ha="right")
    ax.set_ylabel("KS Statistic / TVD", fontsize=16)
    ax.set_ylim(0, max(values) * 1.22)

    # Reference lines
    for yval, color, lbl in [
        (0.05, "#4CAF50", "Very Good (≤ 0.05)"),
        (0.10, "#8BC34A", "Good (≤ 0.10)"),
        (0.20, "#FFC107", "Acceptable (≤ 0.20)"),
        (0.30, "#FF9800", "Moderate (≤ 0.30)"),
    ]:
        if yval <= max(values) * 1.15:
            ax.axhline(yval, color=color, linestyle="--", linewidth=1.0, alpha=0.7, label=lbl)

    ax.set_title(
        "KS Test / TVD: Ground Truth vs. Simulated 10k Cases",
        fontsize=18, fontweight="bold", pad=15,
    )
    ax.legend(loc="upper right", fontsize=12, framealpha=0.75)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Diagram saved: {save_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  KS-Test: Ground Truth vs. Simulierte 10k Cases")
    print("=" * 65)

    # 1. Daten laden
    print("\n[1] Lade simulierte Daten ...")
    if not SIM_PATH.exists():
        raise FileNotFoundError(f"Simulierte Daten nicht gefunden: {SIM_PATH}")
    df_sim = pd.read_csv(SIM_PATH)
    print(f"  ✓ {len(df_sim):,} Zeilen, {df_sim.shape[1]} Spalten")

    print("\n[2] Lade Ground Truth ...")
    df_gt = load_ground_truth()

    # 2. KS-Tests durchführen
    print("\n[3] Führe KS-Tests durch ...")
    result_df = run_ks_tests(df_gt, df_sim)

    # 3. Ergebnisse ausgeben
    print("\n" + "=" * 65)
    print("  ERGEBNISSE")
    print("=" * 65)

    # Formatierte Ausgabe
    col_widths = {
        "Attribut":  max(len("Attribut"),  result_df["Attribut"].str.len().max()),
        "Typ":       max(len("Typ"),        result_df["Typ"].str.len().max()),
        "KS / TVD":  9,
        "p-Wert":    10,
        "n_GT":      7,
        "n_Sim":     7,
        "Bewertung": max(len("Bewertung"), result_df["Bewertung"].str.len().max()),
    }

    header = (
        f"{'Attribut':<{col_widths['Attribut']}}  "
        f"{'Typ':<{col_widths['Typ']}}  "
        f"{'KS / TVD':>{col_widths['KS / TVD']}}  "
        f"{'p-Wert':>{col_widths['p-Wert']}}  "
        f"{'n_GT':>{col_widths['n_GT']}}  "
        f"{'n_Sim':>{col_widths['n_Sim']}}  "
        f"{'Bewertung'}"
    )
    print(header)
    print("-" * len(header))

    for _, row in result_df.iterrows():
        print(
            f"{row['Attribut']:<{col_widths['Attribut']}}  "
            f"{row['Typ']:<{col_widths['Typ']}}  "
            f"{str(row['KS / TVD']):>{col_widths['KS / TVD']}}  "
            f"{str(row['p-Wert']):>{col_widths['p-Wert']}}  "
            f"{str(row['n_GT']):>{col_widths['n_GT']}}  "
            f"{str(row['n_Sim']):>{col_widths['n_Sim']}}  "
            f"{row['Bewertung']}"
        )

    # Zusammenfassung
    num_rows = result_df[result_df["Typ"] == "numerisch"]
    cat_rows = result_df[result_df["Typ"] == "kategorisch"]

    print(f"\n  Numerische Attribute:   {len(num_rows)}")
    if len(num_rows) > 0:
        print(f"    Ø KS-Statistik:       {num_rows['KS / TVD'].mean():.4f}")
        print(f"    Max. KS-Statistik:    {num_rows['KS / TVD'].max():.4f}  ({num_rows.loc[num_rows['KS / TVD'].idxmax(), 'Attribut']})")
        print(f"    Min. KS-Statistik:    {num_rows['KS / TVD'].min():.4f}  ({num_rows.loc[num_rows['KS / TVD'].idxmin(), 'Attribut']})")

    print(f"\n  Kategorische Attribute: {len(cat_rows)}")
    if len(cat_rows) > 0:
        print(f"    Ø TVD:                {cat_rows['KS / TVD'].mean():.4f}")
        print(f"    Max. TVD:             {cat_rows['KS / TVD'].max():.4f}  ({cat_rows.loc[cat_rows['KS / TVD'].idxmax(), 'Attribut']})")

    # p-Wert Interpretation
    if len(num_rows) > 0:
        sig_count = sum(
            float(v.replace("e", "E")) < 0.05
            for v in num_rows["p-Wert"]
            if v != "–"
        )
        print(f"\n  Signifikante Unterschiede (p < 0.05): {sig_count}/{len(num_rows)} numerische Attribute")
        print("  → Ein niedriger p-Wert bedeutet, die Verteilungen sind statistisch")
        print("    unterschiedlich. Die KS-Statistik gibt das Ausmaß an.")

    # 4. Diagramm erstellen
    print("\n[4] Erstelle Diagramm ...")
    plot_ks_bars(result_df, OUTPUT_PNG)

    print("\n" + "=" * 65)
    print("  Fertig!")
    print("=" * 65)


if __name__ == "__main__":
    main()
