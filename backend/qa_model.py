# qa_model.py

import requests
from rag_system import Retriever

class RAGSystem:
    """
    Retrieval-Augmented Generation system using:
    - rag_system.Retriever for context retrieval
    - Ollama HTTP API (/api/chat) for local model generation
    """
    def __init__(
        self,
        index_path="retrieval.index",
        texts_path="texts.pkl",
        embed_model="all-MiniLM-L6-v2",
        ollama_url="http://127.0.0.1:11434/api/chat",
        model_name="deepseek-r1:1.5b",
        top_k=5
    ):
        # Initialize retriever
        self.retriever = Retriever(
            index_path=index_path,
            texts_path=texts_path,
            model_name=embed_model
        )
        # Store Ollama parameters
        self.ollama_url = ollama_url
        self.ollama_model = model_name
        self.top_k = top_k

    def _retrieve_contexts(self, question: str):
        """
        Retrieve top-k most relevant contexts for a given question.
        """
        return self.retriever.retrieve(question, top_k=self.top_k)

    def _build_prompt(self, question: str, contexts: list) -> str:
        """
        Build the final prompt including retrieved contexts and the user question.
        """
        prompt = "Answer the question based on the following contexts:\n\n"
        for idx, ctx in enumerate(contexts, start=1):
            prompt += f"Context {idx}: {ctx}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt

    def _generate(self, prompt: str) -> str:
        """
        Send prompt to Ollama HTTP API (/api/chat) and return generated output.
        """
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        response = requests.post(self.ollama_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    def answer_question(self, question: str) -> str:
        """
        End-to-end RAG pipeline: retrieve, build prompt, and generate answer.
        """
        contexts = self._retrieve_contexts(question)
        prompt = self._build_prompt(question, contexts)
        return self._generate(prompt)

if __name__ == "__main__":
    # Example usage
    qa_system = RAGSystem(top_k=3)
    test_questions = [
        "What were the main causes of the American Civil War?",
        "How did the Bering land bridge form?"
    ]
    for q in test_questions:
        print(f"\n=== Question: {q} ===")
        print("Answer:", qa_system.answer_question(q))
