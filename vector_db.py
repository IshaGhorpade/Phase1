import faiss
import numpy as np
import pickle


class VectorDB:

    def __init__(self, dim):

        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, docs):

        self.index.add(np.array(embeddings))
        self.metadata.extend(docs)

    def search(self, query_embedding, k=5):

        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )

        results = [self.metadata[i] for i in indices[0]]

        return results

    def save(self, path="vector_store"):

        faiss.write_index(self.index, f"{path}.faiss")

        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path="vector_store"):

        self.index = faiss.read_index(f"{path}.faiss")

        with open(f"{path}_meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)