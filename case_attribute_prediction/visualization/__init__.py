from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from pathlib import Path

from ..utils import to_case_level, resolve_col
from ..metrics import ks_statistic_1d


def plot_attribute_distributions(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    attribute_cols: List[str],
    *,
    original_cols: Optional[List[str]] = None,
    simulated_cols: Optional[List[str]] = None,
    n_cols: int = 3,
    bin_count: int = 30,
    figsize_per_plot: Tuple[float, float] = (5.0, 3.5),
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Vergleicht die Verteilungen aller Attribute zwischen Ground Truth und simulierten Daten.

    Für jedes Attribut wird ein eigener Subplot erstellt:
    - Numerische Attribute: überlappende Histogramme + KDE
    - Kategorische Attribute: nebeneinander stehende Balkendiagramme (Anteile)

    Args:
        df:              Original DataFrame (Ground Truth, auf Ereignis-Ebene)
        sim_df:          Simuliertes DataFrame (auf Case-Ebene)
        attribute_cols:  Liste der Attributnamen, die verglichen werden sollen
        original_cols:   Optionale abweichende Spaltennamen im Original-DF
        simulated_cols:  Optionale abweichende Spaltennamen im Sim-DF
        n_cols:          Anzahl der Spalten im Subplot-Raster
        bin_count:       Anzahl der Bins für Histogramme
        figsize_per_plot: Breite × Höhe pro Subplot in Zoll
        save_path:       Pfad zum Speichern des Diagramms
        title:           Gesamt-Titel

    Returns:
        matplotlib Figure
    """
    COLOR_GT  = "#2196F3"  # Blau  – Ground Truth
    COLOR_SIM = "#FF5722"  # Orange-Rot – Simuliert

    valid_attrs: list[str] = []
    gt_series: list[pd.Series] = []
    sim_series: list[pd.Series] = []
    is_cat: list[bool] = []

    for idx, attr in enumerate(attribute_cols):
        orig_col = resolve_col(df, original_cols[idx] if original_cols else attr)
        sim_col  = resolve_col(sim_df, simulated_cols[idx] if simulated_cols else attr)

        # Ground Truth auf Case-Ebene reduzieren
        orig = to_case_level(df, ["case:concept:name", orig_col]).copy()
        sim  = sim_df[[sim_col]].copy()

        orig_s = orig[orig_col].dropna()
        sim_s  = sim[sim_col].dropna()

        if len(orig_s) == 0 or len(sim_s) == 0:
            print(f"  Warnung: {attr} – Keine gültigen Werte, wird übersprungen.")
            continue

        # Typ bestimmen
        categorical = (
            orig_s.dtype == object
            or sim_s.dtype == object
            or orig_s.dtype.name == "bool"
            or sim_s.dtype.name == "bool"
        )

        if not categorical:
            orig_num = pd.to_numeric(orig_s, errors="coerce").dropna()
            sim_num  = pd.to_numeric(sim_s,  errors="coerce").dropna()
            if len(orig_num) == 0 or len(sim_num) == 0:
                print(f"  Warnung: {attr} – Keine numerischen Werte nach Konvertierung.")
                continue
            orig_s = orig_num
            sim_s  = sim_num

        valid_attrs.append(attr)
        gt_series.append(orig_s)
        sim_series.append(sim_s)
        is_cat.append(categorical)

    if not valid_attrs:
        raise ValueError("Keine gültigen Attribute für Verteilungsvergleich gefunden.")

    n_attrs = len(valid_attrs)
    n_rows  = math.ceil(n_attrs / n_cols)
    fig_w   = figsize_per_plot[0] * n_cols
    fig_h   = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes_flat = np.array(axes).flatten() if n_attrs > 1 else [axes]

    for i, (attr, gt_s, sim_s, categorical) in enumerate(
        zip(valid_attrs, gt_series, sim_series, is_cat)
    ):
        ax = axes_flat[i]

        if categorical:
            # ── Kategorisch: Anteile als nebeneinander stehende Balken ──────────
            gt_s  = gt_s.astype(str)
            sim_s = sim_s.astype(str)

            all_cats = sorted(set(gt_s.unique()) | set(sim_s.unique()))
            gt_prop  = gt_s.value_counts(normalize=True).reindex(all_cats, fill_value=0.0)
            sim_prop = sim_s.value_counts(normalize=True).reindex(all_cats, fill_value=0.0)

            x      = np.arange(len(all_cats))
            width  = 0.38
            ax.bar(x - width / 2, gt_prop.values,  width, color=COLOR_GT,  alpha=0.85,
                   label="Ground Truth", edgecolor="white", linewidth=0.5)
            ax.bar(x + width / 2, sim_prop.values, width, color=COLOR_SIM, alpha=0.85,
                   label="Simuliert",    edgecolor="white", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("Anteil", fontsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

            # TVD als Info-Text
            tvd = 0.5 * sum(abs(gt_prop[c] - sim_prop[c]) for c in all_cats)
            ax.set_title(f"{attr}\n(TVD = {tvd:.3f})", fontsize=9, fontweight="bold")

        else:
            # ── Numerisch: überlappende Histogramme ─────────────────────────────
            combined = pd.concat([gt_s, sim_s])
            lo, hi   = combined.quantile(0.01), combined.quantile(0.99)
            if lo == hi:
                lo, hi = combined.min(), combined.max()
            if lo == hi:
                lo -= 1; hi += 1

            bins = np.linspace(lo, hi, bin_count + 1)

            ax.hist(gt_s.clip(lo, hi),  bins=bins, density=True, alpha=0.55,
                    color=COLOR_GT,  label="Ground Truth", edgecolor="white", linewidth=0.4)
            ax.hist(sim_s.clip(lo, hi), bins=bins, density=True, alpha=0.55,
                    color=COLOR_SIM, label="Simuliert",    edgecolor="white", linewidth=0.4)

            # KDE-Linien
            try:
                from scipy.stats import gaussian_kde
                for series, color in [(gt_s, COLOR_GT), (sim_s, COLOR_SIM)]:
                    clipped = series.clip(lo, hi)
                    if clipped.nunique() > 1:
                        kde  = gaussian_kde(clipped, bw_method="scott")
                        xs   = np.linspace(lo, hi, 300)
                        ax.plot(xs, kde(xs), color=color, linewidth=1.6)
            except ImportError:
                pass

            # KS-Statistik als Info-Text
            try:
                ks = ks_statistic_1d(gt_s.to_numpy(), sim_s.to_numpy())
                ks_label = f"KS = {ks:.3f}"
            except Exception:
                ks_label = ""

            ax.set_ylabel("Dichte", fontsize=8)
            ax.set_title(
                f"{attr}\n({ks_label})" if ks_label else attr,
                fontsize=9, fontweight="bold",
            )
            ax.tick_params(axis="x", labelsize=7)

        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # Überschüssige Subplots ausblenden
    for j in range(n_attrs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    suptitle = title or (
        f"Verteilungsvergleich: Ground Truth vs. Simuliert – {n_attrs} Attribute"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Diagramm gespeichert: {save_path}")

    return fig


def plot_ks_statistics_for_attributes(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    attribute_cols: List[str],
    *,
    original_cols: Optional[List[str]] = None,
    simulated_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Visualisiert die KS-Statistik aller angegebenen Attribute in einem Balkendiagramm.

    Args:
        df: Original DataFrame (Ground Truth)
        sim_df: Simuliertes DataFrame
        attribute_cols: Liste von Attributnamen
        original_cols: Optionale Liste der Spaltennamen im Original-DF
        simulated_cols: Optionale Liste der Spaltennamen im Sim-DF
        figsize: Größe der Figur
        save_path: Pfad zum Speichern des Diagramms
        title: Titel des Diagramms

    Returns:
        matplotlib Figure
    """
    ks_values: list[float] = []
    labels: list[str] = []

    for idx, attr in enumerate(attribute_cols):
        orig_col = resolve_col(df, original_cols[idx] if original_cols else attr)
        sim_col = resolve_col(sim_df, simulated_cols[idx] if simulated_cols else attr)

        # Case-level extrahieren
        orig = to_case_level(df, ["case:concept:name", orig_col]).copy()
        sim = sim_df[[sim_col]].copy()

        orig = orig.dropna(subset=[orig_col])
        sim = sim.dropna(subset=[sim_col])

        if len(orig) == 0 or len(sim) == 0:
            print(f"  Warnung: {attr} - Keine gültigen Daten für Vergleich")
            continue

        # Prüfe ob kategorisch (string/object) oder numerisch
        is_categorical = orig[orig_col].dtype == object or sim[sim_col].dtype == object

        if is_categorical:
            orig_counts = orig[orig_col].value_counts(normalize=True)
            sim_counts = sim[sim_col].value_counts(normalize=True)

            all_categories = set(orig_counts.index.tolist() + sim_counts.index.tolist())

            tvd = 0.0
            for cat in all_categories:
                p_orig = orig_counts.get(cat, 0.0)
                p_sim = sim_counts.get(cat, 0.0)
                tvd += abs(p_orig - p_sim)
            tvd = 0.5 * tvd

            ks_values.append(float(tvd))
            labels.append(attr)
            print(f"  ✓ {attr}: TVD = {tvd:.4f} (kategorisch)")
        else:
            if orig[orig_col].dtype == bool:
                orig[orig_col] = orig[orig_col].astype(int)
            if sim[sim_col].dtype == bool:
                sim[sim_col] = sim[sim_col].astype(int)

            orig[orig_col] = pd.to_numeric(orig[orig_col], errors="coerce")
            sim[sim_col] = pd.to_numeric(sim[sim_col], errors="coerce")

            orig = orig.dropna(subset=[orig_col])
            sim = sim.dropna(subset=[sim_col])

            if len(orig) == 0 or len(sim) == 0:
                print(f"  Warnung: {attr} - Keine numerischen Werte nach Konvertierung")
                continue

            x = orig[orig_col].to_numpy()
            y = sim[sim_col].to_numpy()

            unique_x = len(np.unique(x))
            unique_y = len(np.unique(y))

            if unique_x < 1 or unique_y < 1:
                print(f"  Warnung: {attr} - Keine Werte vorhanden")
                continue

            if unique_x == 1 and unique_y == 1:
                print(f"  Warnung: {attr} - Keine Variation (beide haben nur einen Wert)")
                continue

            try:
                ks = ks_statistic_1d(x, y)
                ks_values.append(float(ks))
                labels.append(attr)
                print(f"  ✓ {attr}: KS = {ks:.4f}")
            except Exception as e:
                print(f"  ✗ Fehler bei {attr} (KS-Berechnung): {e}")
                continue

    if not ks_values:
        raise ValueError("Keine gültigen KS-Statistiken berechnet (prüfe Attribute und Daten).")

    sorted_pairs = sorted(zip(ks_values, labels), key=lambda x: x[0])
    ks_values = [v for v, _ in sorted_pairs]
    labels = [l for _, l in sorted_pairs]

    n_attrs = len(labels)
    if n_attrs > 10:
        figsize = (max(12, n_attrs * 0.8), 6)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x_pos = np.arange(len(labels))

    colors = []
    for ks in ks_values:
        if ks < 0.1:
            colors.append("green")
        elif ks < 0.2:
            colors.append("orange")
        elif ks < 0.3:
            colors.append("yellow")
        else:
            colors.append("red")

    ax.bar(x_pos, ks_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("KS-Statistik", fontsize=10)
    ax.set_ylim(0, max(max(ks_values) * 1.15, 0.35))

    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, linewidth=1, label="Sehr gut (< 0.1)")
    ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.5, linewidth=1, label="Gut (< 0.2)")
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Akzeptabel (< 0.3)")

    if title is None:
        title = f"KS/TVD-Statistiken pro Attribut (Ground Truth vs. Simuliert) - {n_attrs} Attribute"
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("KS-Statistik / TVD (Total Variation Distance)", fontsize=10)

    max_val = max(ks_values) if ks_values else 0.1
    for i, v in enumerate(ks_values):
        ax.text(i, v + max_val * 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Diagramm gespeichert: {save_path}")

    return fig
