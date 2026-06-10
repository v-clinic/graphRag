"""Cypher query helpers for the VClinic knowledge graph."""

from neo4j import Driver


# ── Patient queries ────────────────────────────────────────────────────────

def get_patient_by_name(driver: Driver, name: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient)
    WHERE toLower(p.first_name) CONTAINS toLower($name)
       OR toLower(p.last_name)  CONTAINS toLower($name)
    RETURN p.id AS id, p.first_name AS first_name, p.last_name AS last_name,
           p.birthdate AS birthdate, p.gender AS gender,
           p.city AS city, p.state AS state
    LIMIT 10
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, name=name)]


def get_patient_conditions(driver: Driver, patient_id: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient {id: $patient_id})-[rel:HAS_CONDITION]->(c:Condition)
    RETURN c.code AS code, c.description AS description,
           rel.start AS start, rel.stop AS stop
    ORDER BY rel.start
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, patient_id=patient_id)]


def get_patient_medications(driver: Driver, patient_id: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient {id: $patient_id})-[rel:PRESCRIBED]->(m:Medication)
    RETURN m.code AS code, m.description AS description,
           rel.start AS start, rel.stop AS stop, rel.reason AS reason
    ORDER BY rel.start DESC
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, patient_id=patient_id)]


def get_patient_procedures(driver: Driver, patient_id: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient {id: $patient_id})-[rel:UNDERWENT]->(pr:Procedure)
    RETURN pr.code AS code, pr.description AS description,
           rel.start AS start, rel.reason AS reason, rel.cost AS cost
    ORDER BY rel.start DESC
    LIMIT 30
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, patient_id=patient_id)]


def get_patient_allergies(driver: Driver, patient_id: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient {id: $patient_id})-[rel:HAS_ALLERGY]->(a:Allergy)
    RETURN a.description AS allergy, a.type AS type, a.category AS category,
           rel.severity AS severity, rel.start AS start
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, patient_id=patient_id)]


def get_patient_encounters(driver: Driver, patient_id: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient {id: $patient_id})-[:HAS_ENCOUNTER]->(e:Encounter)
    OPTIONAL MATCH (e)-[:PERFORMED_BY]->(pv:Provider)
    OPTIONAL MATCH (e)-[:AT]->(o:Organization)
    RETURN e.id AS encounter_id, e.start AS start, e.stop AS stop,
           e.encounter_class AS class, e.description AS description,
           e.reason AS reason, e.total_cost AS cost,
           pv.name AS provider, pv.speciality AS speciality,
           o.name AS organization
    ORDER BY e.start DESC
    LIMIT 20
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, patient_id=patient_id)]


# ── Condition queries ──────────────────────────────────────────────────────

def get_patients_with_condition(driver: Driver, condition: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient)-[rel:HAS_CONDITION]->(c:Condition)
    WHERE toLower(c.description) CONTAINS toLower($condition)
    RETURN p.id AS patient_id,
           p.first_name + ' ' + p.last_name AS name,
           p.gender AS gender, p.birthdate AS birthdate,
           c.description AS condition, rel.start AS diagnosed
    ORDER BY rel.start
    LIMIT 50
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, condition=condition)]


def get_patients_with_medication(driver: Driver, medication: str, database: str) -> list[dict]:
    """Find all patients prescribed a medication (partial name match on m.description)."""
    query = """
    MATCH (p:Patient)-[rel:PRESCRIBED]->(m:Medication)
    WHERE toLower(m.description) CONTAINS toLower($medication)
    RETURN DISTINCT p.id AS patient_id,
           p.first_name + ' ' + p.last_name AS name,
           p.gender AS gender, p.birthdate AS birthdate,
           m.description AS medication, rel.start AS prescribed, rel.stop AS stopped
    ORDER BY rel.start DESC
    LIMIT 50
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, medication=medication)]


def get_medications_for_condition(driver: Driver, condition: str, database: str) -> list[dict]:
    """Find medications commonly prescribed for a given condition reason."""
    query = """
    MATCH (p:Patient)-[rel:PRESCRIBED]->(m:Medication)
    WHERE toLower(rel.reason) CONTAINS toLower($condition)
    RETURN m.description AS medication, m.code AS code,
           count(p) AS patient_count
    ORDER BY patient_count DESC
    LIMIT 20
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, condition=condition)]


def get_procedures_for_condition(driver: Driver, condition: str, database: str) -> list[dict]:
    query = """
    MATCH (p:Patient)-[rel:UNDERWENT]->(pr:Procedure)
    WHERE toLower(rel.reason) CONTAINS toLower($condition)
    RETURN pr.description AS procedure, pr.code AS code,
           count(p) AS patient_count, avg(rel.cost) AS avg_cost
    ORDER BY patient_count DESC
    LIMIT 20
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, condition=condition)]


# ── Provider / organisation queries ───────────────────────────────────────

def get_provider_patients(driver: Driver, provider_name: str, database: str) -> list[dict]:
    query = """
    MATCH (e:Encounter)-[:PERFORMED_BY]->(pv:Provider)
    WHERE toLower(pv.name) CONTAINS toLower($provider_name)
    MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e)
    RETURN DISTINCT p.id AS patient_id,
                    p.first_name + ' ' + p.last_name AS name,
                    p.gender AS gender
    LIMIT 30
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, provider_name=provider_name)]


def get_organisation_stats(driver: Driver, org_name: str, database: str) -> list[dict]:
    query = """
    MATCH (o:Organization)
    WHERE toLower(o.name) CONTAINS toLower($org_name)
    OPTIONAL MATCH (e:Encounter)-[:AT]->(o)
    OPTIONAL MATCH (pv:Provider)-[:WORKS_AT]->(o)
    RETURN o.name AS organization, o.city AS city, o.state AS state,
           count(DISTINCT e)  AS total_encounters,
           count(DISTINCT pv) AS total_providers
    """
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(query, org_name=org_name)]


# ── Generic Cypher ────────────────────────────────────────────────────────

def run_cypher(driver: Driver, cypher: str, database: str, params: dict | None = None) -> list[dict]:
    """Execute an arbitrary read-only Cypher query."""
    with driver.session(database=database) as session:
        return [dict(r) for r in session.run(cypher, **(params or {}))]
