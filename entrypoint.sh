#!/bin/bash
# Start the Ollama server in background
ollama serve &

# Optional: wait a few seconds for Ollama to be ready
sleep 5

# Pull the model (only if not already downloaded)
ollama pull deepseek-r1:1.5b

# Run the Streamlit app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0