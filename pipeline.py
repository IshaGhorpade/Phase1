from pdf_loader import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedder import create_embeddings
from vector_db import VectorDB
from regulatory_graph import insert_rule, close_driver


def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_text(text)


print("Loading PDFs...")

# RBI
rbi_text = extract_text("data/rbi_regulation.pdf")
rbi_chunks = chunk_text(rbi_text)

print("RBI chunks:", len(rbi_chunks))


# EU
eu_text = extract_text("data/eu_regulation.pdf")
eu_chunks = chunk_text(eu_text)

print("EU chunks:", len(eu_chunks))


# Add metadata
documents = []

for c in rbi_chunks:
    documents.append({"text": c, "source": "RBI"})

for c in eu_chunks:
    documents.append({"text": c, "source": "EU"})


print("Total chunks:", len(documents))


# Create embeddings
print("Generating embeddings...")

embeddings = create_embeddings(documents)

print("Embedding dimension:", len(embeddings[0]))


# Vector database
vector_db = VectorDB(len(embeddings[0]))

vector_db.add(embeddings, documents)

vector_db.save("regulatory_vectors")

print("Vector database saved.")


# Insert into Neo4j
print("Building regulatory graph...")

for doc in documents:

    insert_rule(
        doc["text"],
        doc["source"]
    )


close_driver()

print("Phase 1 Completed Successfully")