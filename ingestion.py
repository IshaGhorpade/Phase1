import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

import ollama
from pinecone import Pinecone
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# ================================
# LOAD ENV VARIABLES
# ================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ================================
# EMBEDDING MODEL
# ================================
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# ================================
# PINECONE SETUP
# ================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# ================================
# NEO4J SETUP
# ================================
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


# ================================
# GENERATE EMBEDDING
# ================================
def generate_embedding(text):

    embedding = embed_model.encode(text).tolist()
    return embedding


# ================================
# STORE DOCUMENT IN PINECONE
# ================================
def store_in_pinecone(text, document_id):

    embedding = generate_embedding(text)

    metadata = {
        "text": text,
        "document_id": document_id,
        "uploaded_at": datetime.now().isoformat()
    }

    index.upsert(
        vectors=[
            {
                "id": document_id,
                "values": embedding,
                "metadata": metadata
            }
        ]
    )

    print("Stored in Pinecone")


# ================================
# STORE DOCUMENT IN NEO4J
# ================================
def store_document_in_neo4j(document_id, text):

    query = """
    MERGE (d:Document {id:$id})
    SET d.text=$text,
        d.uploaded_at=$time
    """

    with driver.session() as session:
        session.run(
            query,
            id=document_id,
            text=text,
            time=datetime.now().isoformat()
        )

    print("Stored document in Neo4j")


# ================================
# EXTRACT TRIPLES USING OLLAMA
# ================================
def extract_triplets(text):

    prompt = f"""
You are an information extraction system.

Extract ALL knowledge triples from the text below.

STRICT RULES:
1. Use EXACT format: (Subject)-[RELATION]->(Object)
2. One triple per line
3. DO NOT number the lines
4. DO NOT add explanations
5. DO NOT add extra text

Example:
(RBI)-[UPDATED]->(International Transaction Limit)
(Violations)-[TRIGGER]->(Penalty)

Text:
{text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    triples = response["message"]["content"]

    cleaned_lines = []

    for line in triples.split("\n"):
        line = line.strip()

        if line.startswith("(") and "->" in line:
            cleaned_lines.append(line)

    triples_cleaned = "\n".join(cleaned_lines)

    print("\nTriples Extracted:\n")
    print(triples_cleaned)

    return triples_cleaned


# ================================
# STORE TRIPLE IN NEO4J
# ================================
def store_triple(subject, relation, obj, doc_id):

    relation = relation.upper().replace(" ", "_")

    query = """
    MATCH (d:Document {id:$doc})

    MERGE (a:Entity {name:$sub})
    MERGE (b:Entity {name:$obj})

    MERGE (a)-[:RELATION {type:$rel}]->(b)

    MERGE (d)-[:MENTIONS]->(a)
    MERGE (d)-[:MENTIONS]->(b)
    """

    with driver.session() as session:
        session.run(
            query,
            sub=subject,
            obj=obj,
            rel=relation,
            doc=doc_id
        )


# ================================
# PARSE TRIPLES
# ================================
def parse_triples(triples, doc_id):

    lines = triples.split("\n")

    for line in lines:

        if not line.strip():
            continue

        try:

            subject = line.split("(")[1].split(")")[0]
            relation = line.split("[")[1].split("]")[0]
            obj = line.split("->(")[1].split(")")[0]

            store_triple(subject, relation, obj, doc_id)

        except:
            print("Skipping line:", line)


# ================================
# READ TEXT FILE
# ================================
def read_text_file(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# ================================
# READ PDF FILE
# ================================
def read_pdf_file(file_path):

    import PyPDF2

    text = ""

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            text += page.extract_text()

    return text


# ================================
# INGEST PIPELINE
# ================================
def ingest(text):

    print("\nStarting ingestion...\n")

    doc_id = str(uuid.uuid4())

    store_in_pinecone(text, doc_id)

    store_document_in_neo4j(doc_id, text)

    triples = extract_triplets(text)

    parse_triples(triples, doc_id)

    print("\nIngestion complete")

    return doc_id


# ================================
# MAIN PROGRAM
# ================================
if __name__ == "__main__":

    print("Legal AI Knowledge Graph Pipeline\n")

    file_path = r"E:\internship\legal_ai_phase1\RBI_docx.pdf"   # change file here

    if file_path.endswith(".txt"):
        text = read_text_file(file_path)

    elif file_path.endswith(".pdf"):
        text = read_pdf_file(file_path)

    else:
        print("Unsupported file format")
        exit()

    ingest(text)

    driver.close()