#!/usr/bin/env python3
"""
Simulation Resume Tool
======================
Parst einen sim_run_log.txt, extrahiert die ursprünglichen Simulationsparameter,
sichert vorhandene Teilergebnisse und startet die Simulation neu (ggf. nur noch die
fehlenden Cases, wenn abgeschlossene Cases im CSV erkennbar sind).

Typische Nutzung:
    # Aus dem Projektverzeichnis (process-simulation-engine/):
    python resume_simulation.py

    # Mit explizitem Log-Pfad:
    python resume_simulation.py --log integration/sim_run_log.txt

    # Nur anzeigen, was gemacht würde (kein echter Lauf):
    python resume_simulation.py --dry-run

    # Log der neuen Simulation in Datei schreiben:
    python resume_simulation.py --output-log integration/sim_run_log.txt
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# BPIC17-spezifische Terminal-Aktivitäten im *simulierten* Log.
# Diese werden vom Next-Activity-Predictor als Abschluss einer Case geliefert.
# Wird erweitert, sobald der Predictor weitere Endaktivitäten produziert.
# ---------------------------------------------------------------------------
SIMULATED_TERMINAL_ACTIVITIES = {
    "A_Cancelled",
    "A_Denied",
    "A_Approved",
    "O_Cancelled",
    "O_Accepted",
    "O_Refused",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_sim_log(log_path: str) -> dict:
    """
    Parst sim_run_log.txt und gibt ein dict mit allen relevanten
    Simulationsparametern zurück.
    """
    params: dict = {}

    with open(log_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Eventlog-Pfad
    for line in lines:
        m = re.match(r"Loading event log from:\s+(.+)", line)
        if m:
            params["event_log"] = m.group(1).strip()
            break

    # SIMULATION CONFIGURATION-Block auslesen
    in_config = False
    config_seen = 0
    for line in lines:
        stripped = line.strip()

        if "SIMULATION CONFIGURATION" in stripped:
            in_config = True
            continue

        if in_config and stripped.startswith("====="):
            config_seen += 1
            if config_seen >= 2:          # zweite Trennlinie = Block-Ende
                in_config = False
            continue

        if not in_config:
            continue

        def _val(pattern):
            m = re.match(pattern, stripped)
            return m.group(1).strip() if m else None

        v = _val(r"Processing time mode:\s+(.+)")
        if v: params["processing"] = v

        v = _val(r"Case arrival mode:\s+(.+)")
        if v: params["arrivals"] = v

        v = _val(r"Resource selection:\s+(.+)")
        if v: params["resource_strategy"] = v

        v = _val(r"Resource allocation mode:\s+(.+)")
        if v: params["resource_allocation_mode"] = v

        v = _val(r"PMSP delta:\s+(.+)")
        if v: params["pmsp_dummy_delta"] = v

        # "2.0s" oder "2.0" – Einheit abschneiden
        v = _val(r"PMSP solver time limit:\s+(.+?)s?$")
        if v: params["pmsp_solver_time_limit"] = v.rstrip("s")

        v = _val(r"PMSP prediction batch size:\s+(.+)")
        if v: params["pmsp_prediction_batch_size"] = v

        v = _val(r"PMSP optimization batch size:\s+(.+)")
        if v: params["pmsp_optimization_batch_size"] = v

        v = _val(r"Number of cases:\s+(\d+)")
        if v: params["num_cases"] = int(v)

        v = _val(r"Batch policy:\s+(.+)")
        if v: params["batch_policy"] = v

        v = _val(r"DRL model:\s+(.+)")
        if v: params["drl_model_path"] = v

        v = _val(r"PT lifecycle mode:\s+(.+)")
        if v: params["pt_lifecycle_mode"] = v

        v = _val(r"PT max duration:\s+\S+\s+\((\d+) days\)")
        if v: params["pt_max_duration_days"] = float(v)

    # Letzten Progress-Stand ermitteln
    last_progress = None
    for line in lines:
        m = re.match(
            r"Progress:\s+(\d+) cases started,\s+(\d+) completed,\s+(\d+) events logged",
            line.strip(),
        )
        if m:
            last_progress = {
                "cases_started": int(m.group(1)),
                "cases_completed": int(m.group(2)),
                "events_logged": int(m.group(3)),
            }
    params["last_progress"] = last_progress

    return params


# ---------------------------------------------------------------------------
# CSV-Analyse
# ---------------------------------------------------------------------------

def find_completed_cases(csv_path: str) -> set:
    """
    Liest simulated_log.csv und gibt die IDs der Cases zurück,
    deren letzte Aktivität eine bekannte Endaktivität ist.
    """
    if not os.path.exists(csv_path):
        return set()

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if df.empty:
            return set()

        case_col = "case:concept:name"
        act_col  = "concept:name"
        ts_col   = "time:timestamp"

        if case_col not in df.columns or act_col not in df.columns:
            return set()

        df[ts_col] = pd.to_datetime(df[ts_col], format="mixed")
        last_act = df.sort_values(ts_col).groupby(case_col)[act_col].last()

        return {
            case_id
            for case_id, act in last_act.items()
            if act in SIMULATED_TERMINAL_ACTIVITIES
        }

    except Exception as exc:
        print(f"  [Warnung] CSV-Analyse fehlgeschlagen: {exc}")
        return set()


def csv_stats(csv_path: str) -> dict:
    """Gibt einfache Statistiken über das vorhandene CSV zurück."""
    if not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        return {
            "events": len(df),
            "cases": df["case:concept:name"].nunique() if "case:concept:name" in df.columns else 0,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_csv(csv_path: str) -> str | None:
    """Sichert das vorhandene CSV mit Zeitstempel im Dateinamen."""
    if not os.path.exists(csv_path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = csv_path.replace(".csv", f"_backup_{ts}.csv")
    shutil.copy2(csv_path, backup)
    print(f"    → Backup erstellt: {backup}")
    return backup


# ---------------------------------------------------------------------------
# Kommandozeile aufbauen
# ---------------------------------------------------------------------------

def build_command(params: dict, num_cases_override: int | None = None) -> list[str]:
    """Baut das python -m integration.test_integration … Kommando auf."""

    num_cases = num_cases_override if num_cases_override is not None else params.get("num_cases", 1000)

    cmd = [
        sys.executable, "-m", "integration.test_integration",
        "--num-cases",           str(num_cases),
        "--event-log",           params.get("event_log", "eventlog/eventlog.xes.gz"),
        "--arrivals",            params.get("arrivals", "advanced"),
        "--processing",          params.get("processing", "advanced"),
        "--resource-strategy",   params.get("resource_strategy", "random"),
        "--resource-allocation-mode", params.get("resource_allocation_mode", "pmsp"),
    ]

    mode = params.get("resource_allocation_mode", "pmsp")

    if mode == "pmsp":
        cmd += [
            "--pmsp-dummy-delta",            params.get("pmsp_dummy_delta",            "1.5"),
            "--pmsp-solver-time-limit",      params.get("pmsp_solver_time_limit",      "2.0"),
            "--pmsp-prediction-batch-size",  params.get("pmsp_prediction_batch_size",  "25"),
            "--pmsp-optimization-batch-size",params.get("pmsp_optimization_batch_size","0"),
        ]

    if mode == "batch" and "batch_policy" in params:
        cmd += ["--batch-policy", params["batch_policy"]]

    if mode == "drl" and "drl_model_path" in params:
        cmd += ["--drl-model-path", params["drl_model_path"]]

    if "pt_lifecycle_mode" in params:
        cmd += ["--pt-lifecycle-mode", params["pt_lifecycle_mode"]]

    if "pt_max_duration_days" in params:
        cmd += ["--pt-max-duration-days", str(params["pt_max_duration_days"])]

    return cmd


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_results(completed_backup_path: str, completed_case_ids: set, new_csv_path: str):
    """
    Fügt die abgeschlossenen Cases aus dem Backup mit den neuen Simulationsergebnissen zusammen.
    Das Ergebnis wird in new_csv_path geschrieben (ersetzt die neue Datei).
    """
    try:
        import pandas as pd

        df_backup = pd.read_csv(completed_backup_path)
        df_completed = df_backup[df_backup["case:concept:name"].isin(completed_case_ids)].copy()

        df_new = pd.read_csv(new_csv_path)

        df_merged = pd.concat([df_completed, df_new], ignore_index=True)
        df_merged["time:timestamp"] = pd.to_datetime(df_merged["time:timestamp"], format="mixed")
        df_merged = df_merged.sort_values("time:timestamp").reset_index(drop=True)

        df_merged.to_csv(new_csv_path, index=False)

        n_cases  = df_merged["case:concept:name"].nunique()
        n_events = len(df_merged)
        print(f"    → Merged: {n_cases} Cases, {n_events} Events → {new_csv_path}")

    except Exception as exc:
        print(f"  [Warnung] Merge fehlgeschlagen: {exc}")
        print(f"            Backup liegt unter: {completed_backup_path}")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulation-Resume-Tool – startet eine unterbrochene Simulation neu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log",
        default="integration/sim_run_log.txt",
        help="Pfad zum sim_run_log.txt (Standard: integration/sim_run_log.txt)",
    )
    parser.add_argument(
        "--csv",
        default="integration/output/simulated_log.csv",
        help="Pfad zum (ggf. partiellen) simulated_log.csv",
    )
    parser.add_argument(
        "--output-log",
        default=None,
        metavar="LOGFILE",
        help="Simulation-Output in diese Datei schreiben (überschreibt; kein tee)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeigt an, was gemacht würde – ohne Simulation zu starten",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Kein Backup des vorhandenen CSV erstellen",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Immer alle Cases neu simulieren (auch wenn abgeschlossene Cases erkannt wurden)",
    )
    args = parser.parse_args()

    sep = "=" * 62

    print(sep)
    print("  SIMULATION RESUME TOOL")
    print(sep)

    # ── Schritt 1: Log parsen ──────────────────────────────────────────────
    print(f"\n[1/4] Log parsen …  ({args.log})")

    if not os.path.exists(args.log):
        print(f"  FEHLER: Logdatei nicht gefunden: {args.log}")
        sys.exit(1)

    params = parse_sim_log(args.log)

    print(f"  Eventlog          : {params.get('event_log', '?')}")
    print(f"  Anzahl Cases       : {params.get('num_cases', '?')}")
    print(f"  Arrivals-Modus    : {params.get('arrivals', '?')}")
    print(f"  Processing-Modus  : {params.get('processing', '?')}")
    print(f"  Ressourcen-Allok. : {params.get('resource_allocation_mode', '?')}")
    print(f"  Ressourcen-Strateg: {params.get('resource_strategy', '?')}")

    if params.get("resource_allocation_mode") == "pmsp":
        print(f"  PMSP delta        : {params.get('pmsp_dummy_delta', '?')}")
        print(f"  PMSP solver limit : {params.get('pmsp_solver_time_limit', '?')}s")
        print(f"  PMSP pred. batch  : {params.get('pmsp_prediction_batch_size', '?')}")
        print(f"  PMSP optim. batch : {params.get('pmsp_optimization_batch_size', '?')}")

    progress = params.get("last_progress")
    if progress:
        print(
            f"\n  Letzter Fortschritt: "
            f"{progress['cases_started']} gestartet, "
            f"{progress['cases_completed']} abgeschlossen, "
            f"{progress['events_logged']} Events geloggt"
        )
    else:
        print("\n  [Info] Kein Fortschritt-Eintrag im Log gefunden.")

    # ── Schritt 2: Vorhandene CSV analysieren ──────────────────────────────
    print(f"\n[2/4] Vorhandene Ergebnisse prüfen …  ({args.csv})")

    stats = csv_stats(args.csv)
    if stats:
        print(f"  CSV vorhanden: {stats['events']} Events, {stats['cases']} Cases")
    else:
        print("  Kein CSV vorhanden oder leer.")

    completed_ids: set = set()
    if not args.force_full:
        completed_ids = find_completed_cases(args.csv)

    num_completed = len(completed_ids)
    total_cases   = params.get("num_cases", 1000)
    remaining     = total_cases - num_completed

    if num_completed > 0:
        print(f"  Abgeschlossene Cases erkannt : {num_completed}")
        print(f"  Noch zu simulieren           : {remaining}")
    else:
        print(f"  Keine abgeschlossenen Cases → vollständiger Neustart ({total_cases} Cases)")

    # ── Schritt 3: Backup ─────────────────────────────────────────────────
    print(f"\n[3/4] Backup & Vorbereitung …")

    backup_path: str | None = None
    if args.dry_run:
        print("    → Backup übersprungen (dry-run)")
    elif not args.no_backup and os.path.exists(args.csv):
        backup_path = backup_csv(args.csv)
    else:
        reason = "--no-backup gesetzt" if args.no_backup else "keine CSV-Datei vorhanden"
        print(f"    → Backup übersprungen ({reason})")

    # ── Schritt 4: Simulation starten ──────────────────────────────────────
    num_cases_to_run = remaining if (num_completed > 0 and not args.force_full) else total_cases
    cmd = build_command(params, num_cases_override=num_cases_to_run)

    print(f"\n[4/4] Simulation starten …")
    print(f"  Befehl: {' '.join(cmd)}")

    if args.output_log:
        print(f"  Output → {args.output_log}")

    if args.dry_run:
        print("\n  [DRY-RUN] Kein echter Lauf – Script beendet sich hier.")
        return

    # Aus dem Projektverzeichnis heraus starten
    project_root = Path(__file__).parent
    print()

    if args.output_log:
        log_dir = os.path.dirname(os.path.abspath(args.output_log))
        os.makedirs(log_dir, exist_ok=True)
        with open(args.output_log, "w", encoding="utf-8") as log_fh:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_fh.write(result.stdout)
            sys.stdout.write(result.stdout)
    else:
        result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode != 0:
        print(f"\n  FEHLER: Simulation mit Exit-Code {result.returncode} beendet.")
        if backup_path:
            print(f"  Teilergebnisse gesichert unter: {backup_path}")
        sys.exit(result.returncode)

    # ── Optional: Abgeschlossene Cases aus Backup mergen ──────────────────
    if num_completed > 0 and backup_path:
        print(f"\n  Merge: {num_completed} vorher abgeschlossene Cases + neue Ergebnisse …")
        merge_results(backup_path, completed_ids, args.csv)

    print()
    print(sep)
    print("  RESUME ABGESCHLOSSEN")
    print(sep)


if __name__ == "__main__":
    main()
