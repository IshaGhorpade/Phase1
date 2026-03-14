from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(docs):

    texts = [d["text"] for d in docs]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    return embeddings