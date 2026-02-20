import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# LOAD ENV VARIABLES
# --------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

PINECONE_INDEX_NAME = "legal-index"

# --------------------------------------------------
# INITIALIZE CLIENTS
# --------------------------------------------------

# OpenAI (used ONLY for triples)
client = OpenAI(api_key=OPENAI_API_KEY)

# FREE embedding model
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Neo4j
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# ==================================================
# FUNCTION: Generate Embedding (FREE)
# ==================================================
def generate_embedding(text):

    embedding = embed_model.encode(text).tolist()
    return embedding


# ==================================================
# FUNCTION: Store Text in Pinecone
# ==================================================
def store_in_pinecone(text):

    vector_id = str(uuid.uuid4())

    embedding = generate_embedding(text)

    index.upsert(
        vectors=[{
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": text}
        }]
    )

    print("✅ Stored in Pinecone | ID:", vector_id)

    return vector_id


# ==================================================
# FUNCTION: Extract Triples using LLM
# ==================================================
def extract_triplets(text):

    prompt = f"""
Extract ALL semantic legal triples from the text.

STRICT output format ONLY (no explanation):

(Subject)-[RELATION]->(Object)

If multiple triples exist, return each on a new line.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    triples = response.choices[0].message.content.strip()

    print("\n✅ Extracted Triples:\n", triples)

    return triples


# ==================================================
# FUNCTION: Store Triples in Neo4j
# ==================================================
def store_in_neo4j(subject, relation, obj, vector_id):

    relation = relation.replace(" ", "_").upper()

    query = f"""
    MERGE (a:Entity {{name: $subject}})
    MERGE (b:Entity {{name: $object}})
    MERGE (a)-[r:{relation}]->(b)
    SET r.vector_id = $vector_id
    """

    with driver.session() as session:
        session.run(
            query,
            subject=subject.strip(),
            object=obj.strip(),
            vector_id=vector_id
        )

    print(f"✅ Stored in Neo4j: ({subject})-[{relation}]->({obj})")


# ==================================================
# FUNCTION: Parse Triples
# ==================================================
def parse_and_store(triples_text, vector_id):

    lines = triples_text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        try:
            subject = line.split("(")[1].split(")")[0]
            relation = line.split("[")[1].split("]")[0]
            obj = line.split("->(")[1].split(")")[0]

            store_in_neo4j(subject, relation, obj, vector_id)

        except Exception:
            print("⚠ Skipped malformed line:", line)


# ==================================================
# MAIN INGESTION PIPELINE
# ==================================================
def ingest(text):

    print("\n🚀 Starting Ingestion...\n")

    # Step 1 — Pinecone
    vector_id = store_in_pinecone(text)

    # Step 2 — Extract triples
    triples_text = extract_triplets(text)

    # Step 3 — Neo4j
    parse_and_store(triples_text, vector_id)

    print("\n🎉 INGESTION COMPLETED SUCCESSFULLY!\n")


# ==================================================
# TEST RUN
# ==================================================
if __name__ == "__main__":

    sample_text = """
    RBI updated the international transaction limit to 50000 USD.
    Violations trigger a penalty of 10 percent of transaction value.
    """

    ingest(sample_text)

    driver.close()
