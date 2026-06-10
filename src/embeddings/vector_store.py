"""
Vector store setup using Neo4j's built-in vector index.

Embeddings are built for clinical entity nodes using OpenAI text-embedding-3-small.

Bridges natural language queries to the precise entity names that the graph queries need.

Embeds clinical entities — Takes nodes like Condition, Medication, Procedure, etc., converts their descriptions into vector embeddings via OpenAI, and stores them on the node as n.embedding
Creates vector indexes in Neo4j — Neo4j natively supports vector indexes (ANN search), so the embeddings live right alongside the graph data
Exposes a vector_search tool — The agent calls this when a query is semantically vague; it returns the top-5 most similar clinical entities by cosine distance

User: "find patients with blood sugar problems"
         ↓
  vector_search("blood sugar problems", label="Condition")
         ↓
  Neo4j vector index → ["Diabetes", "Hyperglycemia", ...]
         ↓
  Agent uses those resolved names in follow-up graph queries

"""

from __future__ import annotations

from neo4j import Driver
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings

from src.config import settings

# Node label → Cypher expression used to build the text that gets embedded
EMBEDDING_TARGETS: dict[str, str] = {
    "Condition":   "coalesce(n.description, '')",
    "Medication":  "coalesce(n.description, '')",
    "Procedure":   "coalesce(n.description, '')",
    "Allergy":     "coalesce(n.description, '') + ' ' + coalesce(n.category, '')",
    "Immunization": "coalesce(n.description, '')",
    "Observation": "coalesce(n.description, '') + ' ' + coalesce(n.category, '')",
}


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def create_vector_indexes(driver: Driver, database: str) -> None:
    """Create vector indexes for each node label (idempotent)."""
    for label in EMBEDDING_TARGETS:
        index_name = f"{label.lower()}_vector_idx"
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label}) ON (n.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {settings.vector_dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        with driver.session(database=database) as session:
            session.run(query)


def populate_embeddings(driver: Driver, database: str) -> None:
    """Generate and store embeddings for all nodes that don't have one yet."""
    embeddings = get_embeddings()

    for label, text_expr in EMBEDDING_TARGETS.items():
        fetch_query = f"""
        MATCH (n:{label})
        WHERE n.embedding IS NULL AND n.description IS NOT NULL
        RETURN n.code AS id, {text_expr} AS text
        LIMIT 500
        """
        write_query = f"""
        UNWIND $batch AS item
        MATCH (n:{label} {{code: item.id}})
        SET n.embedding = item.embedding
        """

        with driver.session(database=database) as session:
            while True:
                records = [dict(r) for r in session.run(fetch_query)]
                if not records:
                    break
                texts = [r["text"] for r in records]
                vectors = embeddings.embed_documents(texts)
                batch = [{"id": r["id"], "embedding": vec}
                         for r, vec in zip(records, vectors)]
                session.run(write_query, batch=batch)


def get_vector_store(label: str = "Condition") -> Neo4jVector:
    """Return a Neo4jVector retriever for the given clinical node label."""
    index_name = f"{label.lower()}_vector_idx"
    return Neo4jVector.from_existing_index(
        embedding=get_embeddings(),
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        index_name=index_name,
        node_label=label,
        text_node_property="description",
        embedding_node_property="embedding",
    )
