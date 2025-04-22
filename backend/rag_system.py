# rag_system.py

import numpy as np
import pickle
import faiss
import ollama
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    Retriever using FAISS and SentenceTransformer for vector search.
    """
    def __init__(self, index_path="retrieval.index", texts_path="texts.pkl", model_name="all-MiniLM-L6-v2"):
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        # Load texts associated with vectors
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)
        # Load embedding model
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, query, top_k=5):
        """
        Given a query, retrieve top_k relevant documents.
        """
        q_vec = self.embedder.encode([query])[0]
        q_vec = q_vec / np.linalg.norm(q_vec)  # Normalize vector
        scores, indices = self.index.search(np.array([q_vec], dtype=np.float32), top_k)
        return [self.texts[i] for i in indices[0]]

class RAGSystem:
    """
    Retrieval-Augmented Generation system:
    1. Retrieve relevant contexts using Retriever
    2. Build prompt
    3. Generate answer via Ollama local server
    """
    def __init__(
        self,
        index_path="retrieval.index",
        texts_path="texts.pkl",
        embed_model="all-MiniLM-L6-v2",
        model_name="deepseek-r1:1.5b",
        top_k=5
    ):
        self.retriever = Retriever(
            index_path=index_path,
            texts_path=texts_path,
            model_name=embed_model
        )
        self.ollama_model = model_name
        self.top_k = top_k

    def _retrieve_contexts(self, question: str):
        """
        Retrieve top-k contexts relevant to the question.
        """
        return self.retriever.retrieve(question, top_k=self.top_k)

    '''
    def _build_prompt(self, question, contexts):
        context_text = "\n\n".join(contexts)
        prompt = f"""You are an experienced American history tutor helping a student understand historical concepts.

    Below are some relevant historical passages. Based on this information, provide a clear, structured, and educational answer to the student’s question. The answer should be easy to understand, free of jargon, and explain the key points thoroughly.

    Relevant Contexts:
    ---------------------
    {context_text}
    ---------------------

    Student's Question:
    {question}

    Answer:"""
        return prompt
    '''

    def _build_prompt(self, question: str, contexts: list) -> str:
        """
        Construct prompt combining retrieved contexts and the user question.
        """
        prompt = "Answer the question based on the following contexts:\n\n"
        for idx, ctx in enumerate(contexts, start=1):
            prompt += f"Context {idx}: {ctx}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt

    def _generate(self, prompt: str) -> str:
        """
        Send prompt to Ollama API and return generated answer.
        """
        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']

    def answer_question(self, question: str) -> str:
        """
        End-to-end pipeline to get answer from a user question.
        """
        contexts = self._retrieve_contexts(question)
        prompt = self._build_prompt(question, contexts)
        return self._generate(prompt)

if __name__ == "__main__":
    # Test Retriever alone
    retriever = Retriever()
    query = "What happened during the Columbian Exchange?"
    results = retriever.retrieve(query)
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc[:300]}...")

    # Test Full RAGSystem
    print("\n=== Full RAG System Test ===")
    rag = RAGSystem(top_k=3)
    question = "What were the main causes of the American Civil War?"
    print("Answer:", rag.answer_question(question))
