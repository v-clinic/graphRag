"""Full build pipeline: ingest CSV data + populate embeddings.

Run this once (or after new CSV data arrives) to:
  1. Load all VClinic CSVs into Neo4j as a property graph
  2. Create Neo4j vector indexes on clinical entity nodes
  3. Generate OpenAI embeddings for those nodes so the agent
     can do semantic similarity search at query time
"""

from pathlib import Path

from neo4j import GraphDatabase
from rich.console import Console

from src.config import settings
from src.graph.ingestion import run_ingestion
from src.embeddings.vector_store import create_vector_indexes, populate_embeddings

console = Console()

# Default location of the Synthea-format CSV export.
# Override at call time: build_graph("/path/to/other/data")
DEFAULT_DATA_DIR = Path("test_data/vclinic")


def build_graph(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    # Open a single reusable Neo4j driver for the whole pipeline.
    # Using one driver avoids repeatedly negotiating the Bolt connection.
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        # ── Step 1: Ingest CSV → graph ────────────────────────────────────────
        # Reads all VClinic CSVs and writes Patient, Condition, Medication,
        # Procedure, etc. nodes plus their relationships (HAS_CONDITION,
        # PRESCRIBED, etc.) into Neo4j using MERGE (idempotent re-runs).
        run_ingestion(driver, settings.neo4j_database, data_dir)

        # ── Step 2: Create vector indexes ─────────────────────────────────────
        # Registers an ANN (approximate nearest-neighbour) vector index in
        # Neo4j for each clinical entity label (Condition, Medication, …).
        # This is a schema-level operation — no embeddings are written yet.
        # The indexes are created with IF NOT EXISTS so re-runs are safe.
        console.print("\n[bold cyan]Creating vector indexes...[/bold cyan]")
        create_vector_indexes(driver, settings.neo4j_database)
        console.print("[green]Vector indexes ready.[/green]")

        # ── Step 3: Generate & store embeddings ───────────────────────────────
        # Calls OpenAI text-embedding-3-small for every node that has a
        # description but no embedding yet, then writes the vector back to
        # n.embedding.  The vector indexes created above then index these
        # vectors automatically, enabling cosine-similarity search at query time.
        console.print("\n[bold cyan]Generating embeddings (this may take a moment)...[/bold cyan]")
        populate_embeddings(driver, settings.neo4j_database)
        console.print("[green]Embeddings populated.[/green]")

    finally:
        # Always close the driver to release the connection pool,
        # even if an earlier step raised an exception.
        driver.close()

    console.print("\n[bold green]Knowledge graph build complete! ✓[/bold green]")
