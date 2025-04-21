# app.py

import streamlit as st
import sys
import os
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

# Import RAGSystem
from rag_system import RAGSystem

# Initialize RAG system with correct paths
rag = RAGSystem(
    index_path="backend/retrieval.index",
    texts_path="backend/texts.pkl",
    top_k=3
)

# Setup Streamlit page
st.set_page_config(page_title="Educational RAG System", layout="wide")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # List of {"question", "contexts", "answer", "time"}
if "selected" not in st.session_state:
    st.session_state.selected = None
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# Sidebar for question history
st.sidebar.title("📜 History")
# Display history with status icon
for idx, record in enumerate(st.session_state.history):
    col1, col2 = st.sidebar.columns([6, 1])

    # Show ✅ if answered, ⏳ if pending (here all are answered immediately after retrieval)
    status_icon = "✅" if "answer" in record and record["answer"] else "⏳"
    button_label = f"{status_icon} {idx + 1}. {record['question']}"

    if col1.button(button_label, key=f"q_{idx}"):
        st.session_state.selected = idx
    if col2.button("🗑️", key=f"del_{idx}"):
        st.session_state.history.pop(idx)
        if st.session_state.selected == idx:
            st.session_state.selected = None
        st.rerun()  # Use st.rerun() for new Streamlit versions

# Clear all history button
if st.sidebar.button("🗑️ Clear All History"):
    st.session_state.history = []
    st.session_state.selected = None

# Main page layout
st.title("🎓 Educational Tutor - American History (RAG)")
st.markdown("---")

# Main input area
with st.form("question_form"):
    user_question = st.text_input(
        "Ask a question about American History:",
        value=st.session_state.current_input
    )
    submitted = st.form_submit_button("Submit")

# Handle new question submission
if submitted and user_question.strip():
    st.session_state.current_input = user_question  # Preserve input

    with st.spinner("Retrieving contexts and generating answer..."):
        start_time = time.time()
        contexts = rag._retrieve_contexts(user_question)
        prompt = rag._build_prompt(user_question, contexts)
        answer = rag._generate(prompt)
        elapsed_time = time.time() - start_time

        # Save to history
        st.session_state.history.append({
            "question": user_question,
            "contexts": contexts,
            "answer": answer,
            "time": round(elapsed_time, 2)
        })
        st.session_state.selected = len(st.session_state.history) - 1

        st.session_state.current_input = ""  # Clear input after answer ready
        st.rerun()

# Show selected QA
if st.session_state.history:
    if st.session_state.selected is None:
        st.session_state.selected = len(st.session_state.history) - 1

    current = st.session_state.history[st.session_state.selected]

    st.subheader(f"📖 Question: {current['question']}")
    st.markdown(f"⏱️ Answer generated in {current['time']} seconds")
    st.markdown("---")

    st.subheader("🔍 Retrieved Contexts")
    for idx, ctx in enumerate(current["contexts"], 1):
        with st.expander(f"Context {idx}"):
            st.write(ctx)

    st.subheader("🧠 Answer")
    st.markdown(current["answer"])
