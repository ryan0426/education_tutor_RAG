
# Educational RAG System (American History)

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions about **American History**.  
It combines document retrieval with a local LLM model (e.g., DeepSeek via Ollama) to generate accurate and context-aware answers.

---

## 🚀 How to Run Locally

### 1. Set up Python Environment

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

## 2. Install Ollama

### macOS

Install Ollama:

```bash
brew install ollama
```

Start the Ollama server:

```bash
ollama serve
```

Then in another terminal, pull the required model:

```bash
ollama pull deepseek-r1:1.5b
```

---

### Linux (Ubuntu/WSL2)

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server:

```bash
ollama serve
```

Then pull the model:

```bash
ollama pull deepseek-r1:1.5b
```

> **Note**: Make sure `ollama serve` is running before pulling models.

---

## 3. Install Conda and Setup FAISS for Linux (Optional for GPU Acceleration)

If you are using Linux and want to utilize GPU-accelerated FAISS:

### Install Conda (if not installed)

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts to complete installation.

Then restart the terminal or run:

```bash
source ~/.bashrc
```

### Create and activate Conda environment

```bash
conda create -n rag-env python=3.12
conda activate rag-env
```

### Install required packages

```bash
pip install -r requirements.txt
conda install faiss-gpu -c pytorch
```

✅ Now your FAISS will use GPU acceleration if available.

---

## 4. Launch the Streamlit App

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
