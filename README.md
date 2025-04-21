# Educational RAG System (American History)

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions about **American History**.  
It combines document retrieval with a local LLM model (e.g., DeepSeek via Ollama) to generate accurate and context-aware answers.

---

## 🚀 How to Run Locally

### 1. Install dependencies

First, create and activate a Python virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Setup and Start Ollama Server

Make sure you have [Ollama](https://ollama.com/) installed on your machine.

Start the Ollama server:

```bash
ollama serve
```

Then in a separate terminal, pull the required model:

```bash
ollama pull deepseek-r1:1.5b
```

The server must be running before pulling models or sending any requests.

---

### 3. Launch the Streamlit app

Run:

```bash
streamlit run app.py
```

Then open your browser and visit:  
http://localhost:8501

You will see:
- An input box to ask questions
- Retrieved supporting documents
- The generated answer
- A history of your past questions and answers

---

## ⚙️ Features

- Retrieval using FAISS and Sentence-Transformers
- Local LLM generation via Ollama
- Streamlit web UI
- Question history with individual delete
- Context display with expandable sections

---

## 📋 Future Improvements

- Add multi-turn conversation memory
- Implement answer citation (highlight which context supports the answer)
- Add authentication for deployed server
- Save question-answer history to file

---

## ✨ Credits

- [Streamlit](https://streamlit.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
