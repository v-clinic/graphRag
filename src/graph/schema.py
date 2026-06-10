"""
VClinic Knowledge Graph Schema  (Synthea-format clinical data)
==============================================================
Nodes  (unique key shown in brackets):
  Patient      [id]           — demographics, vitals summary
  Organization [id]           — clinic / hospital
  Provider     [id]           — clinician with speciality
  Encounter    [id]           — clinical visit
  Condition    [code]         — SNOMED condition (deduplicated)
  Medication   [code]         — RxNorm drug (deduplicated)
  Procedure    [code]         — SNOMED procedure (deduplicated)
  Allergy      [code]         — allergen (deduplicated)
  Immunization [code]         — vaccine (deduplicated)
  Observation  [code]         — observation type e.g. Body Height (deduplicated)

Relationships:
  (Patient)  -[:HAS_ENCOUNTER     {start, stop, class}]->  (Encounter)
  (Encounter)-[:PERFORMED_BY]->                             (Provider)
  (Encounter)-[:AT]->                                       (Organization)
  (Provider) -[:WORKS_AT]->                                 (Organization)
  (Patient)  -[:HAS_CONDITION     {start, stop}]->          (Condition)
  (Patient)  -[:PRESCRIBED        {start, stop, reason}]->  (Medication)
  (Patient)  -[:UNDERWENT         {start, reason}]->        (Procedure)
  (Patient)  -[:HAS_ALLERGY       {start, severity}]->      (Allergy)
  (Patient)  -[:IMMUNIZED_WITH    {date}]->                 (Immunization)
  (Patient)  -[:HAS_OBSERVATION   {date, value, units}]->   (Observation)
"""

# Expose the module docstring as a named constant so other modules can import
# it directly and pass it to LLM-based tools (e.g. Text2CypherRetriever)
# without APOC being required on the Neo4j instance.
SCHEMA_TEXT: str = __doc__ or ""

SCHEMA_CONSTRAINTS = [
    "CREATE CONSTRAINT patient_id      IF NOT EXISTS FOR (n:Patient)      REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT org_id          IF NOT EXISTS FOR (n:Organization)  REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT provider_id     IF NOT EXISTS FOR (n:Provider)      REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT encounter_id    IF NOT EXISTS FOR (n:Encounter)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT condition_code  IF NOT EXISTS FOR (n:Condition)     REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT medication_code IF NOT EXISTS FOR (n:Medication)    REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT procedure_code  IF NOT EXISTS FOR (n:Procedure)     REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT allergy_code    IF NOT EXISTS FOR (n:Allergy)       REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT immunization_code IF NOT EXISTS FOR (n:Immunization) REQUIRE n.code IS UNIQUE",
    "CREATE CONSTRAINT observation_code IF NOT EXISTS FOR (n:Observation)  REQUIRE n.code IS UNIQUE",
]

SCHEMA_INDEXES = [
    "CREATE INDEX patient_name      IF NOT EXISTS FOR (n:Patient)      ON (n.last_name)",
    "CREATE INDEX condition_desc    IF NOT EXISTS FOR (n:Condition)     ON (n.description)",
    "CREATE INDEX medication_desc   IF NOT EXISTS FOR (n:Medication)    ON (n.description)",
    "CREATE INDEX procedure_desc    IF NOT EXISTS FOR (n:Procedure)     ON (n.description)",
    "CREATE INDEX provider_spec     IF NOT EXISTS FOR (n:Provider)      ON (n.speciality)",
    "CREATE INDEX encounter_class   IF NOT EXISTS FOR (n:Encounter)     ON (n.encounter_class)",
]

NODE_LABELS = [
    "Patient", "Organization", "Provider", "Encounter",
    "Condition", "Medication", "Procedure",
    "Allergy", "Immunization", "Observation",
]

REL_TYPES = [
    "HAS_ENCOUNTER", "PERFORMED_BY", "AT", "WORKS_AT",
    "HAS_CONDITION", "PRESCRIBED", "UNDERWENT",
    "HAS_ALLERGY", "IMMUNIZED_WITH", "HAS_OBSERVATION",
]
