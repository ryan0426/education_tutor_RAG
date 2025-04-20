import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path="retrieval.index", texts_path="texts.pkl", model_name="all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)

        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, query, top_k=5):
        q_vec = self.embedder.encode([query])[0]
        q_vec = q_vec / np.linalg.norm(q_vec)
        scores, indices = self.index.search(np.array([q_vec], dtype=np.float32), top_k)
        return [self.texts[i] for i in indices[0]]

if __name__ == "__main__":
    retriever = Retriever()

    query = "What happened during the Columbian Exchange?"
    results = retriever.retrieve(query)

    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc[:300]}...")