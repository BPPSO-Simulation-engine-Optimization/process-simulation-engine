"""
resource_overlap_analysis.py
============================
Analysiert, wie viele Aktivitaeten jede Ressource gleichzeitig bearbeitet.

Fuer jede Ressource werden aus dem Event-Log (XES oder CSV) Start/End-Intervalle
pro (Case, Aktivitaet) rekonstruiert. Dann werden gezaehlt:

  - intervals          : Gesamtzahl der Intervalle (= bearbeitete Aktivitaeten)
  - max_concurrent     : Maximum gleichzeitig aktiver Intervalle (Sweep-Line)
  - overlap_pair_count : Anzahl sich ueberschneidender Paare (i, j) mit i < j
  - overlap_total_min  : Summe aller Ueberschneidungsdauern in Minuten

Verwendung
----------
    # Originallog (xes / xes.gz)
    python evaluation/resource_overlap_analysis.py \
        --input eventlog/eventlog.xes.gz

    # Simulierter Log (csv)
    python evaluation/resource_overlap_analysis.py \
        --input integration/output/simulated_log.csv

    # Mit CSV-Export
    python evaluation/resource_overlap_analysis.py \
        --input eventlog/eventlog.xes.gz \
        --output-summary  integration/output/resource_overlap_summary.csv \
        --output-segments integration/output/resource_active_segments.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Lifecycle-Definitionen (BPIC17)
# ---------------------------------------------------------------------------
START_TRANSITIONS = {"start", "resume"}
END_TRANSITIONS   = {"complete", "ate_abort", "withdraw", "suspend"}


# ---------------------------------------------------------------------------
# Log laden
# ---------------------------------------------------------------------------

def load_log(path: Path) -> pd.DataFrame:
    """Laedt XES (.xes / .xes.gz) oder CSV und gibt einen DataFrame zurueck."""
    suffix = "".join(path.suffixes).lower()
    if ".xes" in suffix:
        import pm4py
        df = pm4py.convert_to_dataframe(pm4py.read_xes(str(path)))
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Nicht unterstuetztes Format: {path}")

    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["time:timestamp"])
    return df


# ---------------------------------------------------------------------------
# Intervall-Extraktion: start/resume -> complete/ate_abort/withdraw/suspend
# ---------------------------------------------------------------------------

def extract_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baut fuer jedes (case, activity, resource) Start/End-Paare auf.

    Logik:
      - Events chronologisch sortieren.
      - START-Event oeffnet einen Slot (FIFO-Stack),
        END-Event schliesst den aeltesten offenen Slot.
      - Nur Paare mit end > start werden behalten.
    """
    df = df.copy()
    df["_lc"] = df["lifecycle:transition"].astype(str).str.lower()

    df = df.sort_values(
        ["case:concept:name", "concept:name", "org:resource", "time:timestamp"]
    )

    rows = []
    group_cols = ["case:concept:name", "concept:name", "org:resource"]

    for (case_id, activity, resource), grp in df.groupby(group_cols, sort=False):
        if pd.isna(resource) or str(resource).strip() in ("", "nan"):
            continue

        open_starts = []

        for _, row in grp.iterrows():
            lc = row["_lc"]
            ts = row["time:timestamp"]

            if lc in START_TRANSITIONS:
                open_starts.append(ts)

            elif lc in END_TRANSITIONS and open_starts:
                start_ts = open_starts.pop(0)   # FIFO
                if ts > start_ts:
                    rows.append({
                        "case_id":      case_id,
                        "activity":     activity,
                        "resource":     str(resource),
                        "start":        start_ts,
                        "end":          ts,
                        "duration_sec": (ts - start_ts).total_seconds(),
                    })

    if not rows:
        return pd.DataFrame(
            columns=["case_id", "activity", "resource", "start", "end", "duration_sec"]
        )
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Overlap-Berechnung pro Ressource  (Sweep-Line + Paar-Zaehlung)
# ---------------------------------------------------------------------------

def compute_overlaps(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet pro Ressource:
      max_concurrent     : maximale Anzahl gleichzeitig aktiver Intervalle
      overlap_pair_count : Anzahl sich ueberschneidender Paare (i,j), i<j
      overlap_total_min  : aufsummierte Ueberschneidungszeit in Minuten

    Zwei Intervalle [s1,e1) und [s2,e2) ueberschneiden sich, wenn s2 < e1
    (nach Sortierung nach s, also s2 >= s1).
    """
    stats = []

    for resource, grp in segments.groupby("resource"):
        ivs = grp[["start", "end"]].sort_values("start")
        starts = ivs["start"].tolist()
        ends   = ivs["end"].tolist()
        n      = len(starts)

        # --- Max Concurrent via Sweep-Line ---------------------------------
        events = []
        for s, e in zip(starts, ends):
            events.append((s, +1))
            events.append((e, -1))
        # Gleicher Zeitpunkt: Ende vor Start (kein falsches Overlap zählen)
        events.sort(key=lambda x: (x[0], x[1]))

        cur = max_conc = 0
        for _, delta in events:
            cur += delta
            if cur > max_conc:
                max_conc = cur

        # --- Overlap-Paare & Gesamtzeit ------------------------------------
        pair_count  = 0
        overlap_sec = 0.0

        for i in range(n):
            s_i = starts[i]
            e_i = ends[i]
            for j in range(i + 1, n):
                s_j = starts[j]
                if s_j >= e_i:
                    break              # alle weiteren j beginnen noch spaeter
                e_j          = ends[j]
                ov_start     = max(s_i, s_j)
                ov_end       = min(e_i, e_j)
                if ov_end > ov_start:
                    pair_count  += 1
                    overlap_sec += (ov_end - ov_start).total_seconds()

        stats.append({
            "resource":           str(resource),
            "intervals":          n,
            "max_concurrent":     int(max_conc),
            "overlap_pair_count": int(pair_count),
            "overlap_total_min":  round(overlap_sec / 60, 2),
        })

    if not stats:
        return pd.DataFrame(
            columns=["resource", "intervals", "max_concurrent",
                     "overlap_pair_count", "overlap_total_min"]
        )

    out = pd.DataFrame(stats)
    out = out.sort_values(
        ["max_concurrent", "overlap_pair_count"], ascending=False
    ).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def print_summary(summary: pd.DataFrame) -> None:
    total        = len(summary)
    with_overlap = int((summary["overlap_pair_count"] > 0).sum())

    print("=" * 70)
    print("RESSOURCEN-PARALLELARBEIT - Uebersicht")
    print("=" * 70)
    print(f"  Ressourcen gesamt               : {total}")
    print(f"  Ressourcen mit mind. 1 Overlap  : {with_overlap}")
    print()
    print("Top-Ressourcen nach max. Gleichzeitigkeit:")
    print("-" * 70)
    print(summary.head(30).to_string(index=False))
    print()

    if with_overlap == 0:
        print("Keine Ueberschneidungen gefunden.")
        print("Hinweis: Das Log enthaelt moeglicherweise kaum start/resume-Events.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet pro Ressource, wie viele Aktivitaeten sie gleichzeitig "
            "bearbeitet, anhand von Start/End-Intervallen aus dem Event-Log."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Pfad zum Event-Log (.xes, .xes.gz oder .csv).",
    )
    parser.add_argument(
        "--output-summary",
        help="Optionaler Pfad fuer die Zusammenfassung als CSV.",
    )
    parser.add_argument(
        "--output-segments",
        help="Optionaler Pfad fuer alle extrahierten Intervalle als CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    print(f"\nLade Event-Log: {input_path}")
    df = load_log(input_path)
    print(
        f"  {len(df):,} Events  |  "
        f"{df['case:concept:name'].nunique():,} Cases  |  "
        f"{df['org:resource'].nunique():,} Ressourcen"
    )
    print("\nLifecycle-Verteilung:")
    print(df["lifecycle:transition"].value_counts().to_string())

    print("\nExtrahiere Intervalle (start/resume -> complete/ate_abort/withdraw/suspend) ...")
    segments = extract_segments(df)
    print(f"  {len(segments):,} Intervalle extrahiert")

    if segments.empty:
        print("Keine Intervalle gefunden - Abbruch.")
        return

    print("\nBeispiel-Intervalle (erste 5):")
    print(
        segments.head(5)[["resource", "activity", "start", "end", "duration_sec"]]
        .to_string(index=False)
    )

    print("\nBerechne Ueberschneidungen ...")
    summary = compute_overlaps(segments)
    print()
    print_summary(summary)

    if args.output_summary:
        out = Path(args.output_summary).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"Summary geschrieben: {out}")

    if args.output_segments:
        seg_out = Path(args.output_segments).resolve()
        seg_out.parent.mkdir(parents=True, exist_ok=True)
        segments.to_csv(seg_out, index=False)
        print(f"Segments geschrieben: {seg_out}")


if __name__ == "__main__":
    main()
