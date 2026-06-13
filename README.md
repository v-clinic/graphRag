This is the patient analysis project with GraphRAG, for the V-Clinic project.
This version uses the Community Edition of Neo4j, smaller LLM models, and synthetic data for POC and demonstration purposes.

---

## Project Structure

```
graphRag/
├── app.py                        # CLI entrypoint (build / chat / ask)
├── requirements.txt
├── .env.example                  # Copy to .env and fill in credentials
└── src/
    ├── config.py                 # Pydantic settings (reads .env)
    ├── graph/
    │   ├── schema.py             # Node labels, constraints, index definitions + SCHEMA_TEXT
    │   ├── ingestion.py          # CSV → Neo4j ingestion pipeline
    │   └── queries.py            # Cypher query helpers
    ├── embeddings/
    │   └── vector_store.py       # Vector index creation & embedding population
    ├── agent/
    │   ├── prompts.py            # System prompt for the LangGraph agent
    │   ├── tools.py              # LangChain tools (text_to_cypher, vector_search, …)
    │   ├── graph_analysis_agent.py  # LangGraph agent (StateGraph + ToolNode)
    │   └── cont_import_agent.py  # Continuous import agent (polls new_data/)
    ├── pipeline/
    │   └── build_graph.py        # Orchestrates ingestion + embedding (run once)
    └── evaluation/               # RAG evaluation framework
        ├── metrics.py            # Custom medical metrics (entity recall, code accuracy)
        ├── runner.py             # Runs agent + captures tool-call outputs as contexts
        ├── evaluator.py          # RAGAS + custom metrics orchestration
        ├── requirements.txt      # Evaluation-specific dependencies
        ├── README.md             # Evaluation documentation
        └── data/
            └── eval_dataset.json # 30 curated QA evaluation samples
```

## Knowledge Graph Schema

**Node types** (deduplicated by code/id):
`Patient` · `Organization` · `Provider` · `Encounter` · `Condition` · `Medication` · `Procedure` · `Allergy` · `Immunization` · `Observation`

**Key relationships:**
```
(Patient)-[:HAS_ENCOUNTER]->(Encounter)-[:PERFORMED_BY]->(Provider)
                                        -[:AT]->(Organization)
(Patient)-[:HAS_CONDITION {start,stop}]->(Condition)
(Patient)-[:PRESCRIBED    {start,stop,reason}]->(Medication)
(Patient)-[:UNDERWENT     {start,reason}]->(Procedure)
(Patient)-[:HAS_ALLERGY   {severity}]->(Allergy)
(Patient)-[:IMMUNIZED_WITH {date}]->(Immunization)
(Patient)-[:HAS_OBSERVATION {date,value,units}]->(Observation)
(Provider)-[:WORKS_AT]->(Organization)
```

## Setup

### 1. Create `.env`
```bash
cp .env.example .env
# Edit .env with your Neo4j connection details and OpenAI API key
```

### 2. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the knowledge graph
```bash
python app.py build --data-dir test_data/vclinic
```
This will:
- Apply schema constraints & indexes in Neo4j
- Ingest the core CSV files (patients, encounters, conditions, medications, procedures, allergies, immunizations, observations, providers, organizations)
- Create vector indexes for clinical entities
- Generate OpenAI embeddings for Condition, Medication, Procedure, Allergy, Immunization, Observation nodes

### 4. Chat with the agent
```bash
# Interactive session
python app.py chat

# Single question
python app.py ask-once "What conditions does patient Brekke have?"
```

## Example Questions

- *"What conditions does Maurice Brekke have?"*
- *"How many patients have been diagnosed with hypertension?"*
- *"What medications are commonly prescribed for diabetes?"*
- *"Show me the encounter history for patient [UUID]"*
- *"Which procedures are most often performed for acute bronchitis?"*
- *"List all allergies for patient [name]"*
- *"Which providers work at Fitchburg Outpatient Clinic?"*

## System Architecture

![System Architecture](architecture.png)

<details>
<summary>Mermaid source</summary>

```mermaid
flowchart TB
    %% ── External ──────────────────────────────────────────────────────────
    subgraph EXT["☁  VClinic EHR  (external)"]
        direction TB
        VCLINIC["VClinic Application"]
        EXPORT["Scheduled Export Job\n(nightly / on-demand)"]
        VCLINIC -->|"triggers"| EXPORT
    end

    %% ── Storage ───────────────────────────────────────────────────────────
    subgraph FS["📁  File System"]
        direction TB
        INITIAL_DIR["vclinic/\n── patients.csv\n── conditions.csv\n── medications.csv\n── encounters.csv\n── … (10 CSVs)"]
        NEW_DIR["vclinic/new_data/\n── patients.csv  ← appended rows\n── conditions.csv\n── …\n── .import_state.json"]
    end

    EXPORT -->|"appends new rows to"| NEW_DIR

    %% ── Initial Build Pipeline ────────────────────────────────────────────
    subgraph INIT["🔧  Initial Build Pipeline  (build_graph.py)"]
        direction LR
        INGEST["run_ingestion()\nCSV → Nodes + Relationships\n(MERGE — idempotent)"]
        IDX["create_vector_indexes()\nANN indexes per entity label"]
        EMB["populate_embeddings()\ntext-embedding-3-small\n→ n.embedding"]
        INGEST --> IDX --> EMB
    end

    INITIAL_DIR -->|"reads all CSVs once"| INGEST

    %% ── Continuous Import Agent ───────────────────────────────────────────
    subgraph CIA["🔄  Continuous Import Agent  (cont_import_agent.py)"]
        direction TB
        POLL["Poll new_data/ every N seconds"]
        CURSOR["Row-count cursor\nper file (.import_state.json)\nskip already-imported rows"]
        PARTIAL["Read only NEW rows\n→ temp CSV"]
        LOADER["File loader\n(load_patients / load_conditions / …)"]
        EMBU["populate_embeddings()\nnew nodes only"]
        POLL --> CURSOR --> PARTIAL --> LOADER --> EMBU
        EMBU -->|"wait interval"| POLL
    end

    NEW_DIR -->|"scans for appended rows"| CIA

    %% ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI["☁  OpenAI API\ntext-embedding-3-small\nGPT-4o"]

    EMB & EMBU -->|"embed entity descriptions"| OPENAI

    %% ── Neo4j Knowledge Graph ─────────────────────────────────────────────
    subgraph NEO4J["🗄  Neo4j Knowledge Graph"]
        direction TB
        GRAPH["Property Graph\nPatient · Encounter · Condition\nMedication · Procedure · Allergy\nImmunization · Observation\nProvider · Organization"]
        VIDX["Vector Indexes\n(cosine ANN per entity label)"]
    end

    INIT -->|"writes nodes + relationships"| GRAPH
    LOADER -->|"MERGE new nodes + relationships"| GRAPH
    EMBU -->|"SET n.embedding"| GRAPH

    %% ── GraphRAG Agent ────────────────────────────────────────────────────
    subgraph AGENT["🤖  GraphRAG Agent  (graph_analysis_agent.py)"]
        direction TB
        LLM["LangGraph ReAct Loop\nChatOpenAI · GPT-4o"]
        subgraph TOOLS["Tools"]
            T1["text_to_cypher\nNL → Cypher via\nText2CypherRetriever"]
            T2["vector_search\nNeo4jVector ANN"]
            T3["get_patient_*\nget_patients_with_*\nget_*_for_condition\n…"]
            T4["run_cypher_query\n(manual escape-hatch)"]
        end
        LLM <-->|"tool calls / results"| TOOLS
    end

    OPENAI -->|"LLM inference"| LLM
    T1 & T4 -->|"Cypher queries"| GRAPH
    T2 -->|"ANN similarity search"| VIDX
    T3 -->|"parameterised Cypher"| GRAPH

    %% ── User / App ────────────────────────────────────────────────────────
    USER(["👤  Clinical Staff / App"])
    USER -->|"natural language question"| LLM
    LLM -->|"answer"| USER
```

</details>

## Tech Stack

| Layer | Technology |
|---|---|
| Graph DB | Neo4j (local Docker) |
| Query Language | Cypher |
| Vector Search | Neo4j built-in vector index |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o |
| Agent Framework | LangGraph (StateGraph + ToolNode) |
| Tool Integration | LangChain tools + `langchain-neo4j` |
| CLI | Typer + Rich |


This project uses OpenAI models. The OpenAI API key is stored as a local environment variable.

---

## Test Data

Synthetic patient records generated by [Synthea](https://synthetichealth.github.io/synthea/) are stored at:

```
test_data/vclinic/
├── patients.csv          (~117 patients)
├── encounters.csv        (~8,300 rows)
├── conditions.csv        (~4,000 rows)
├── medications.csv       (~5,800 rows)
├── procedures.csv        (~20,300 rows)
├── observations.csv      (~86,600 rows)
├── immunizations.csv     (~1,600 rows)
├── allergies.csv         (~130 rows)
├── providers.csv         (~270 rows)
├── organizations.csv     (~270 rows)
├── imaging_studies.csv   (~72,700 rows)  — not ingested by default
├── careplans.csv         (~400 rows)     — not ingested by default
├── devices.csv           (~620 rows)     — not ingested by default
├── supplies.csv          (~2,900 rows)   — not ingested by default
└── new_data/             — drop updated CSVs here for continuous import
```

All data is fully synthetic (no real patient information). The 10 core files are ingested by `python app.py build`. The remaining files (`imaging_studies`, `careplans`, `devices`, `supplies`) are present in the folder but are not yet mapped to the knowledge graph schema.

## ⚠️ License & Disclaimer

This project is a **Proof of Concept (POC)** and is intended solely for **demonstration and educational purposes**.

* **No Production Use:** This code is not production-ready and should not be deployed in live environments.
* **No Liability:** The code owner accepts no responsibility for any damages, data loss, or issues caused by running this software.
* **As-Is:** This software is provided *as-is*, without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, or non-infringement.
* **Not for Clinical Use:** This system **must not** be used to inform, support, or replace real clinical decisions, diagnoses, or patient care of any kind. All data used is fully synthetic and has no connection to real patients or medical records.
* **License:** Distributed under the MIT License.
* **Test Data:** Synthetic patient data is generated by [Synthea™](https://synthetichealth.github.io/synthea/), an open-source synthetic patient generator developed by The MITRE Corporation, released under the Apache License 2.0.
