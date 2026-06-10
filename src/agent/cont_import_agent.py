"""
Continuous import agent.

Periodically scans a watch directory for new or modified CSV files and merges
them into the existing VClinic Neo4j knowledge graph using the same loaders as
the initial ingestion pipeline.

Assumptions:
  - The graph schema is already in place (run build_graph.py first).
  - CSV filenames match the well-known Synthea names (patients.csv, etc.).
  - No new node labels or relationship types are introduced — only data changes.

After each import cycle, populate_embeddings() is called so any newly added
clinical-entity nodes receive vector embeddings automatically.

Usage (library):
    from src.agent.cont_import_agent import start
    start(interval_seconds=120)

Usage (CLI):
    python -m src.agent.cont_import_agent
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from neo4j import Driver, GraphDatabase
from rich.console import Console

from src.config import settings
from src.embeddings.vector_store import populate_embeddings
from src.graph.ingestion import (
    load_allergies,
    load_conditions,
    load_encounters,
    load_immunizations,
    load_medications,
    load_observations,
    load_organizations,
    load_patients,
    load_procedures,
    load_providers,
)

console = Console()
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

NEW_DATA_DIR = Path("/Users/yunwen/work/test_data/vclinic/new_data")

# Hidden state file written inside the watch directory.
# Schema: { filename: { "row_count": int, "imported_at": iso-timestamp } }
#   row_count    — number of data rows (excluding the CSV header) that have
#                  been successfully imported.  On the next cycle we skip
#                  exactly this many rows and only process what was appended.
#   imported_at  — human-readable timestamp of the last successful import,
#                  useful for auditing.
_STATE_FILE_NAME = ".import_state.json"

# Maps each expected CSV filename to its ingestion loader function.
# Any CSV file not in this map is skipped with a warning.
FILE_LOADERS = {
    "patients.csv":      load_patients,
    "organizations.csv": load_organizations,
    "providers.csv":     load_providers,
    "encounters.csv":    load_encounters,
    "conditions.csv":    load_conditions,
    "medications.csv":   load_medications,
    "procedures.csv":    load_procedures,
    "allergies.csv":     load_allergies,
    "immunizations.csv": load_immunizations,
    "observations.csv":  load_observations,
}

# ── State helpers ─────────────────────────────────────────────────────────────

def _state_path(data_dir: Path) -> Path:
    return data_dir / _STATE_FILE_NAME

# state — { "conditions.csv": { "row_count": 1500, "imported_at": "2026-06-10T12:00:00+00:00" }
# First cycle:   conditions.csv has 1000 rows  → imports all 1000, cursor = 1000
# Second cycle:  conditions.csv has 1000 rows  → current == cursor, skip
# Third cycle:   conditions.csv has 1200 rows  → imports rows 1001–1200, cursor = 1200
def _load_state(data_dir: Path) -> dict[str, dict]:
    """
    Return the persisted state dict.
    Schema: { filename: {"row_count": int, "imported_at": str} }
    """
    path = _state_path(data_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read import state file: %s — starting fresh.", exc)
    return {}


def _save_state(data_dir: Path, state: dict[str, dict]) -> None:
    """Persist the import state so the next cycle knows what was already done."""
    _state_path(data_dir).write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


# ── Partial CSV helpers ────────────────────────────────────────────────────────

def _count_data_rows(path: Path) -> int:
    """Return the number of data rows in a CSV (header row excluded)."""
    with open(path, newline="", encoding="utf-8") as f:
        # DictReader consumes the header automatically.
        return sum(1 for _ in csv.DictReader(f))


def _read_new_rows(path: Path, skip_rows: int) -> list[dict]:
    """
    Read a CSV and return only the rows that come AFTER the first skip_rows
    data rows (i.e. rows appended since the last import).
    The header row is never counted in skip_rows.
    """
    with open(path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    return all_rows[skip_rows:]


def _write_partial_csv(rows: list[dict], dest: Path) -> None:
    """Write a list of dicts to dest as a CSV (with header)."""
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── Driver helper ─────────────────────────────────────────────────────────────

def _get_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


# ── Core import cycle ─────────────────────────────────────────────────────────

def run_once(data_dir: Path = NEW_DATA_DIR) -> list[str]:
    """
    Scan data_dir for CSV files that have new rows appended since the last
    import cycle and merge only those new rows into Neo4j.

    How it works:
      - The state file records the number of data rows successfully imported
        for each CSV file (row_count cursor).
      - On each call, the current row count of the file is compared to the
        stored cursor.  Only rows beyond the cursor are processed.
      - After a successful import the cursor is advanced to the new total,
        so the same rows are never imported twice.
      - If a file has FEWER rows than the cursor (truncated / replaced),
        the cursor resets to 0 and the whole file is re-imported.

    Returns the list of filenames imported in this cycle.
    """
    if not data_dir.exists():
        console.print(f"[yellow]Watch directory does not exist: {data_dir}[/yellow]")
        return []

    state = _load_state(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        console.print("[dim]No CSV files found in watch directory.[/dim]")
        return []

    imported: list[str] = []
    driver = _get_driver()

    # A single temp directory is reused across all files in this cycle and
    # cleaned up automatically when the with-block exits.
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)

        try:
            for csv_path in csv_files:
                name = csv_path.name
                loader = FILE_LOADERS.get(name)

                if loader is None:
                    console.print(f"[yellow]⚠  Unknown file '{name}' — skipping (no loader registered).[/yellow]")
                    continue

                # How many rows did we import last time for this file?
                prev_count: int = state.get(name, {}).get("row_count", 0)
                current_count: int = _count_data_rows(csv_path)

                if current_count == prev_count:
                    # No new rows appended — nothing to do.
                    continue

                if current_count < prev_count:
                    # File was truncated or replaced; reset and re-import all.
                    log.warning("%s shrank (%d → %d rows) — re-importing from start.",
                                name, prev_count, current_count)
                    prev_count = 0

                new_rows = _read_new_rows(csv_path, skip_rows=prev_count)
                if not new_rows:
                    continue

                console.print(
                    f"[cyan]→  Importing {name} "
                    f"(rows {prev_count + 1}–{prev_count + len(new_rows)})...[/cyan]"
                )

                # Write only the new rows to a temp CSV so the existing
                # loader functions (which read from a directory) work unchanged.
                _write_partial_csv(new_rows, tmp_dir / name)

                try:
                    n = loader(driver, settings.neo4j_database, tmp_dir)
                    console.print(f"   [green]✓[/green] {name}: {n} records merged")
                    state[name] = {
                        "row_count": prev_count + len(new_rows),
                        "imported_at": datetime.now(timezone.utc).isoformat(),
                    }
                    imported.append(name)
                except Exception as exc:
                    console.print(f"   [red]✗[/red] {name}: {exc}")
                    log.error("Import failed for %s: %s", name, exc, exc_info=True)
                    # Do NOT advance the cursor — failed rows will be retried.

            if imported:
                # Refresh embeddings only for nodes that don't have one yet.
                console.print("\n[cyan]Refreshing embeddings for new nodes...[/cyan]")
                populate_embeddings(driver, settings.neo4j_database)
                console.print("[green]Embeddings refreshed.[/green]")

        finally:
            driver.close()

    # Persist state outside the try/finally so a crash mid-import doesn't
    # record a cursor advance for a file that actually failed.
    _save_state(data_dir, state)
    return imported


# ── Polling loop ──────────────────────────────────────────────────────────────

def start(
    data_dir: Path = NEW_DATA_DIR,
    interval_seconds: int = 60,
) -> None:
    """
    Start the continuous import loop.  Polls data_dir every interval_seconds.

    Blocks forever — run in a background thread or process, or stop with
    Ctrl-C.

    Args:
        data_dir: Directory to watch for new/modified CSV files.
        interval_seconds: Seconds to wait between scans.
    """
    console.print(
        f"[bold cyan]Continuous import agent started.[/bold cyan]\n"
        f"  Watch dir : {data_dir}\n"
        f"  Interval  : {interval_seconds}s\n"
        "Press Ctrl-C to stop.\n"
    )

    while True:
        console.print(f"[dim]── Scanning {data_dir} ──[/dim]")
        try:
            imported = run_once(data_dir)
        except Exception as exc:
            # Catch-all so a transient error (e.g. network blip) doesn't crash
            # the loop — log it and wait for the next cycle.
            console.print(f"[red]Unexpected error during import cycle: {exc}[/red]")
            log.error("Import cycle error: %s", exc, exc_info=True)
            imported = []

        if imported:
            console.print(
                f"[bold green]Cycle complete.[/bold green] "
                f"Imported: {', '.join(imported)}\n"
            )
        else:
            console.print(
                f"[dim]No new files. Next scan in {interval_seconds}s.[/dim]\n"
            )

        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            console.print("\n[yellow]Continuous import agent stopped.[/yellow]")
            break


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VClinic continuous import agent")
    parser.add_argument(
        "--dir",
        type=Path,
        default=NEW_DATA_DIR,
        help="Directory to watch for new CSV files",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single import cycle and exit (no polling loop)",
    )
    args = parser.parse_args()

    if args.once:
        imported = run_once(args.dir)
        if imported:
            console.print(f"[bold green]Done.[/bold green] Imported: {', '.join(imported)}")
        else:
            console.print("[dim]Nothing to import.[/dim]")
    else:
        start(data_dir=args.dir, interval_seconds=args.interval)
