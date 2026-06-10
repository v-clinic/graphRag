"""
CSV ingestion pipeline — loads VClinic (Synthea-format) data into Neo4j.

Expected CSV files:
  patients.csv       encounters.csv   organizations.csv  providers.csv
  conditions.csv     medications.csv  procedures.csv
  allergies.csv      immunizations.csv observations.csv
  careplans.csv      devices.csv      imaging_studies.csv  supplies.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from neo4j import Driver
from rich.console import Console
from rich.progress import track

from src.graph.schema import SCHEMA_CONSTRAINTS, SCHEMA_INDEXES

console = Console()

# Number of rows to send in each UNWIND batch to Neo4j. Tune for performance vs. memory.
BATCH = 500


# ── Schema setup ──────────────────────────────────────────────────────────────

def setup_schema(driver: Driver, database: str) -> None:
    with driver.session(database=database) as session:
        for stmt in SCHEMA_CONSTRAINTS + SCHEMA_INDEXES:
            session.run(stmt)
    console.print("[green]Schema constraints and indexes applied.[/green]")


# ── Generic helpers ───────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_batched(driver: Driver, database: str, query: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    with driver.session(database=database) as session:
        for i in range(0, len(rows), BATCH):
            session.run(query, rows=rows[i : i + BATCH])
    return len(rows)


# ── Node loaders ──────────────────────────────────────────────────────────────

def load_patients(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "patients.csv")
    query = """
    UNWIND $rows AS r
    MERGE (p:Patient {id: r.Id})
    SET p.first_name  = r.FIRST,
        p.last_name   = r.LAST,
        p.birthdate   = r.BIRTHDATE,
        p.deathdate   = r.DEATHDATE,
        p.gender      = r.GENDER,
        p.race        = r.RACE,
        p.ethnicity   = r.ETHNICITY,
        p.city        = r.CITY,
        p.state       = r.STATE,
        p.zip         = r.ZIP
    """
    return _run_batched(driver, database, query, rows)


def load_organizations(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "organizations.csv")
    query = """
    UNWIND $rows AS r
    MERGE (o:Organization {id: r.Id})
    SET o.name    = r.NAME,
        o.address = r.ADDRESS,
        o.city    = r.CITY,
        o.state   = r.STATE,
        o.zip     = r.ZIP,
        o.phone   = r.PHONE
    """
    return _run_batched(driver, database, query, rows)


def load_providers(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "providers.csv")
    # Node
    node_q = """
    UNWIND $rows AS r
    MERGE (pv:Provider {id: r.Id})
    SET pv.name       = r.NAME,
        pv.gender     = r.GENDER,
        pv.speciality = r.SPECIALITY,
        pv.city       = r.CITY,
        pv.state      = r.STATE
    """
    _run_batched(driver, database, node_q, rows)
    # WORKS_AT relationship
    rel_q = """
    UNWIND $rows AS r
    MATCH (pv:Provider {id: r.Id})
    MATCH (o:Organization {id: r.ORGANIZATION})
    MERGE (pv)-[:WORKS_AT]->(o)
    """
    return _run_batched(driver, database, rel_q, rows)


def load_encounters(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "encounters.csv")
    # Encounter nodes
    node_q = """
    UNWIND $rows AS r
    MERGE (e:Encounter {id: r.Id})
    SET e.start           = r.START,
        e.stop            = r.STOP,
        e.encounter_class = r.ENCOUNTERCLASS,
        e.description     = r.DESCRIPTION,
        e.reason          = r.REASONDESCRIPTION,
        e.total_cost      = toFloat(coalesce(r.TOTAL_CLAIM_COST, '0'))
    """
    _run_batched(driver, database, node_q, rows)
    # HAS_ENCOUNTER
    pat_q = """
    UNWIND $rows AS r
    MATCH (p:Patient {id: r.PATIENT})
    MATCH (e:Encounter {id: r.Id})
    MERGE (p)-[:HAS_ENCOUNTER]->(e)
    """
    _run_batched(driver, database, pat_q, rows)
    # PERFORMED_BY
    prov_q = """
    UNWIND $rows AS r
    MATCH (e:Encounter {id: r.Id})
    MATCH (pv:Provider {id: r.PROVIDER})
    MERGE (e)-[:PERFORMED_BY]->(pv)
    """
    _run_batched(driver, database, prov_q, rows)
    # AT organization
    org_q = """
    UNWIND $rows AS r
    MATCH (e:Encounter {id: r.Id})
    MATCH (o:Organization {id: r.ORGANIZATION})
    MERGE (e)-[:AT]->(o)
    """
    return _run_batched(driver, database, org_q, rows)


def load_conditions(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "conditions.csv")
    # Deduplicated Condition nodes
    node_q = """
    UNWIND $rows AS r
    MERGE (c:Condition {code: r.CODE})
    SET c.description = r.DESCRIPTION,
        c.system      = r.SYSTEM
    """
    _run_batched(driver, database, node_q, rows)
    # HAS_CONDITION relationship
    rel_q = """
    UNWIND $rows AS r
    MATCH (p:Patient {id: r.PATIENT})
    MATCH (c:Condition {code: r.CODE})
    MERGE (p)-[rel:HAS_CONDITION {encounter: r.ENCOUNTER, start: r.START}]->(c)
    SET rel.stop = r.STOP
    """
    return _run_batched(driver, database, rel_q, rows)


def load_medications(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "medications.csv")
    # Deduplicated Medication nodes
    node_q = """
    UNWIND $rows AS r
    MERGE (m:Medication {code: r.CODE})
    SET m.description = r.DESCRIPTION
    """
    _run_batched(driver, database, node_q, rows)
    # PRESCRIBED relationship
    rel_q = """
    UNWIND $rows AS r
    MATCH (p:Patient {id: r.PATIENT})
    MATCH (m:Medication {code: r.CODE})
    MERGE (p)-[rel:PRESCRIBED {encounter: r.ENCOUNTER, start: r.START}]->(m)
    SET rel.stop   = r.STOP,
        rel.reason = r.REASONDESCRIPTION,
        rel.cost   = toFloat(coalesce(r.TOTALCOST, '0'))
    """
    return _run_batched(driver, database, rel_q, rows)


def load_procedures(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "procedures.csv")
    node_q = """
    UNWIND $rows AS r
    MERGE (pr:Procedure {code: r.CODE})
    SET pr.description = r.DESCRIPTION,
        pr.system      = r.SYSTEM
    """
    _run_batched(driver, database, node_q, rows)
    rel_q = """
    UNWIND $rows AS r
    MATCH (p:Patient {id: r.PATIENT})
    MATCH (pr:Procedure {code: r.CODE})
    MERGE (p)-[rel:UNDERWENT {encounter: r.ENCOUNTER, start: r.START}]->(pr)
    SET rel.stop   = r.STOP,
        rel.reason = r.REASONDESCRIPTION,
        rel.cost   = toFloat(coalesce(r.BASE_COST, '0'))
    """
    return _run_batched(driver, database, rel_q, rows)


def load_allergies(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "allergies.csv")
    node_q = """
    UNWIND $rows AS r
    MERGE (a:Allergy {code: r.CODE})
    SET a.description = r.DESCRIPTION,
        a.type        = r.TYPE,
        a.category    = r.CATEGORY
    """
    _run_batched(driver, database, node_q, rows)
    rel_q = """
    UNWIND $rows AS r
    MATCH (p:Patient {id: r.PATIENT})
    MATCH (a:Allergy {code: r.CODE})
    MERGE (p)-[rel:HAS_ALLERGY {encounter: r.ENCOUNTER, start: r.START}]->(a)
    SET rel.severity = r.SEVERITY1
    """
    return _run_batched(driver, database, rel_q, rows)


def load_immunizations(driver: Driver, database: str, data_dir: Path) -> int:
    rows = _read_csv(data_dir / "immunizations.csv")
    node_q = """
    UNWIND $rows AS r
    MERGE (v:Immunization {code: r.CODE})
    SET v.description = r.DESCRIPTION
    """
    _run_batched(driver, database, node_q, rows)
    rel_q = """
    UNWIND $rows AS r
    MATCH (p:Patient {id: r.PATIENT})
    MATCH (v:Immunization {code: r.CODE})
    MERGE (p)-[rel:IMMUNIZED_WITH {encounter: r.ENCOUNTER, date: r.DATE}]->(v)
    """
    return _run_batched(driver, database, rel_q, rows)


def load_observations(driver: Driver, database: str, data_dir: Path) -> int:
    """Load deduplicated Observation type nodes + relationships (vital-signs & labs only)."""
    rows = _read_csv(data_dir / "observations.csv")
    # Filter to vital-signs and laboratory categories to keep graph manageable
    filtered = [r for r in rows if r.get("CATEGORY") in ("vital-signs", "laboratory")]
    node_q = """
    UNWIND $rows AS r
    MERGE (ob:Observation {code: r.CODE})
    SET ob.description = r.DESCRIPTION,
        ob.category    = r.CATEGORY,
        ob.units       = r.UNITS
    """
    _run_batched(driver, database, node_q, filtered)
    rel_q = """
    UNWIND $rows AS r
    MATCH (p:Patient {id: r.PATIENT})
    MATCH (ob:Observation {code: r.CODE})
    MERGE (p)-[rel:HAS_OBSERVATION {encounter: r.ENCOUNTER, date: r.DATE}]->(ob)
    SET rel.value = r.VALUE,
        rel.units = r.UNITS
    """
    return _run_batched(driver, database, rel_q, filtered)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_ingestion(driver: Driver, database: str, data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    console.print(f"\n[bold cyan]Starting ingestion from:[/bold cyan] {data_dir}\n")

    setup_schema(driver, database)

    steps = [
        ("Patients", load_patients),
        ("Organizations", load_organizations),
        ("Providers", load_providers),
        ("Encounters", load_encounters),
        ("Conditions", load_conditions),
        ("Medications", load_medications),
        ("Procedures", load_procedures),
        ("Allergies", load_allergies),
        ("Immunizations", load_immunizations),
        ("Observations (vital-signs & labs)", load_observations),
    ]

    for label, fn in track(steps, description="Loading..."):
        try:
            n = fn(driver, database, data_dir)
            console.print(f"  [green]✓[/green] {label}: {n} records")
        except FileNotFoundError as e:
            console.print(f"  [yellow]⚠[/yellow] {label}: skipped ({e.filename} not found)")
        except Exception as e:
            console.print(f"  [red]✗[/red] {label}: {e}")

    console.print("\n[bold green]Ingestion complete.[/bold green]")
