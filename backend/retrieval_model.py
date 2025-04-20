import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class RetrievalModelFAISS:
    def __init__(self, data_path, model_name='all-MiniLM-L6-v2'):
        self.data_path = data_path
        self.model_name = model_name

        self.embedder = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.texts = []
        self.index = None

    def split_by_tokens(self, text, max_tokens=512):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks

    def build_index(self, save_index=True):
        print("[1/2] Loading and processing data...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        for item in data:
            paragraph = item.get("paragraph") or item.get("document")
            chunks = self.split_by_tokens(paragraph)
            self.texts.extend(chunks)

        print(f"Encoding {len(self.texts)} chunks with {self.model_name}...")
        embeddings = self.embedder.encode(self.texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        print("[2/2] Building FAISS index...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        if save_index:
            faiss.write_index(self.index, "retrieval.index")
            with open("texts.pkl", "wb") as f:
                pickle.dump(self.texts, f)
            print("Index saved to disk.")

    def load_index(self, index_path="retrieval.index", text_path="texts.pkl"):
        self.index = faiss.read_index(index_path)
        with open(text_path, "rb") as f:
            self.texts = pickle.load(f)
        print("Index and texts loaded from disk.")

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        return [self.texts[i] for i in indices[0]]

if __name__ == '__main__':
    retriever = RetrievalModelFAISS(data_path='../data/american_yawp_paragraphs.json')

    retriever.build_index()

    query = "What was the impact of the Columbian Exchange?"
    results = retriever.retrieve(query, top_k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc[:300]}...")