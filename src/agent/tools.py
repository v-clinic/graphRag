"""
LangChain tools that expose the VClinic Neo4j knowledge graph to the LangGraph agent.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers.text2cypher import Text2CypherRetriever

from src.config import settings
from src.graph import queries
from src.graph.schema import SCHEMA_TEXT
from src.embeddings.vector_store import get_vector_store

# ── Shared constants ──────────────────────────────────────────────────────────

# Keywords that indicate a write operation — used by run_cypher_query.
_WRITE_KEYWORDS = {"CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP", "CALL"}

# Lazily initialised Text2CypherRetriever.
# On first use it connects to Neo4j and auto-fetches the live schema
# (node labels + properties, relationship types + properties) — no manual
# schema maintenance required.
_t2c_retriever: Text2CypherRetriever | None = None


def _get_t2c_retriever() -> Text2CypherRetriever:
    global _t2c_retriever
    if _t2c_retriever is None:
        # The driver is kept alive by the retriever for the module's lifetime.
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        llm = OpenAILLM(
            model_name=settings.openai_model,
            model_params={"temperature": 0},
            api_key=settings.openai_api_key,
        )
        # neo4j_schema is passed explicitly from our schema.py definition so
        # the retriever never needs to call apoc.meta.data (which may not be
        # installed).  The schema stays in one place and is always in sync.
        _t2c_retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema=SCHEMA_TEXT,
            neo4j_database=settings.neo4j_database,
        )
    return _t2c_retriever


def _get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def _fmt(rows: list[dict]) -> str:
    if not rows:
        return "No results found."
    return json.dumps(rows, indent=2, ensure_ascii=False)


# ── Patient tools ─────────────────────────────────────────────────────────────

@tool
def get_patient_by_name(name: str) -> str:
    """Look up a patient by first or last name. Returns patient ID and demographics."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patient_by_name(driver, name, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_patient_conditions(patient_id: str) -> str:
    """List all conditions ever diagnosed for a patient. Input: patient UUID."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patient_conditions(driver, patient_id, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_patient_medications(patient_id: str) -> str:
    """List medications prescribed to a patient. Input: patient UUID."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patient_medications(driver, patient_id, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_patient_procedures(patient_id: str) -> str:
    """List procedures performed on a patient. Input: patient UUID."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patient_procedures(driver, patient_id, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_patient_allergies(patient_id: str) -> str:
    """List known allergies for a patient. Input: patient UUID."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patient_allergies(driver, patient_id, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_patient_encounters(patient_id: str) -> str:
    """Show recent encounter history for a patient (last 20 visits). Input: patient UUID."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patient_encounters(driver, patient_id, settings.neo4j_database))
    finally:
        driver.close()


# ── Population / condition tools ──────────────────────────────────────────────

@tool
def get_patients_with_medication(medication: str) -> str:
    """Find all patients prescribed a medication (partial name match on medication name). Input: medication name."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patients_with_medication(driver, medication, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_patients_with_condition(condition: str) -> str:
    """Find all patients diagnosed with a condition (partial name match). Input: condition name."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_patients_with_condition(driver, condition, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_medications_for_condition(condition: str) -> str:
    """Find medications most commonly prescribed for a given condition. Input: condition name."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_medications_for_condition(driver, condition, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_procedures_for_condition(condition: str) -> str:
    """Find procedures most commonly performed for a given condition. Input: condition name."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_procedures_for_condition(driver, condition, settings.neo4j_database))
    finally:
        driver.close()


# ── Provider / organisation tools ─────────────────────────────────────────────

@tool
def get_provider_patients(provider_name: str) -> str:
    """List patients seen by a specific provider (partial name match). Input: provider name."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_provider_patients(driver, provider_name, settings.neo4j_database))
    finally:
        driver.close()


@tool
def get_organisation_stats(org_name: str) -> str:
    """Get encounter count and provider count for a clinic or hospital. Input: organization name."""
    driver = _get_driver()
    try:
        return _fmt(queries.get_organisation_stats(driver, org_name, settings.neo4j_database))
    finally:
        driver.close()


# ── Semantic search tool ──────────────────────────────────────────────────────

@tool
def vector_search(query: str, label: str = "Condition") -> str:
    """
    Semantic similarity search across clinical entities in the knowledge graph.
    Args:
        query: Natural language search query.
        label: Node label — one of Condition, Medication, Procedure, Allergy,
               Immunization, Observation.
    Returns top-5 most semantically similar entities.
    """
    valid_labels = {"Condition", "Medication", "Procedure", "Allergy", "Immunization", "Observation"}
    if label not in valid_labels:
        return f"Invalid label '{label}'. Choose from: {', '.join(sorted(valid_labels))}"

    store = get_vector_store(label)
    results = store.similarity_search(query, k=5)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{label}] {doc.page_content}\nMetadata: {json.dumps(doc.metadata, ensure_ascii=False)}"
        for doc in results
    )


# ── Text-to-Cypher tool ──────────────────────────────────────────────────────

@tool
def text_to_cypher(question: str) -> str:
    """
    Answer ANY question about the clinical knowledge graph by automatically
    generating and executing a Cypher query.

    Use this as the primary tool for:
    - Questions not covered by the other specialised tools.
    - Complex multi-hop traversals (e.g. patients with condition X who are also
      prescribed medication Y, or provider-level statistics).
    - Any ad-hoc analytical question.

    Input: a natural language question about patients, conditions, medications,
           procedures, providers, organisations, or any combination thereof.
    """
    retriever = _get_t2c_retriever()
    # Text2CypherRetriever.search() handles the full pipeline internally:
    #   1. Injects the auto-fetched Neo4j schema into the prompt
    #   2. Calls the LLM to generate Cypher from the question
    #   3. Executes the generated Cypher against Neo4j
    # result.records is a list[neo4j.Record]; .data() converts each to a dict.
    try:
        result = retriever.get_search_results(query_text=question)
        rows = [record.data() for record in result.records]
        return _fmt(rows)
    except Exception as e:
        return f"text_to_cypher error: {e}"


# ── Raw Cypher escape-hatch tool ──────────────────────────────────────────────

@tool
def run_cypher_query(cypher: str) -> str:
    """
    Execute a hand-written read-only Cypher query against the VClinic graph.
    Use this only when you already have a specific Cypher query you want to run.
    For natural-language questions, prefer text_to_cypher instead.
    Only MATCH/RETURN queries are permitted.
    Input: a valid Cypher query string.
    """
    upper = cypher.upper()
    for kw in _WRITE_KEYWORDS:
        if kw in upper:
            return f"Write operations are not permitted. Blocked keyword: {kw}"

    driver = _get_driver()
    try:
        rows = queries.run_cypher(driver, cypher, settings.neo4j_database)
        return _fmt(rows)
    except Exception as e:
        return f"Cypher error: {e}"
    finally:
        driver.close()


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    # Generic text-to-cypher — handles any question not covered below.
    text_to_cypher,
    # Semantic similarity search over clinical entity nodes.
    vector_search,
    # Specialised fast-path tools for the most common query patterns.
    get_patient_by_name,
    get_patient_conditions,
    get_patient_medications,
    get_patient_procedures,
    get_patient_allergies,
    get_patient_encounters,
    get_patients_with_medication,
    get_patients_with_condition,
    get_medications_for_condition,
    get_procedures_for_condition,
    get_provider_patients,
    get_organisation_stats,
    run_cypher_query,
]

