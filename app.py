# app.py
import streamlit as st
from workflow import run_langgraph_workflow

st.set_page_config(page_title="Agentic Workflow", layout="wide")
st.title("LangGraph + Ollama AI Agentic Workflow")
query = st.text_area("🔍 Enter your query", height=150)

if st.button("Run Workflow"):
    with st.spinner("Running your agentic workflow..."):
        result = run_langgraph_workflow(query)
        st.success("Workflow completed!")
        for i, entry in enumerate(result['results'], 1):
            st.markdown(f"### 🧠 Task {i}: {entry['task']}")
            st.markdown(f"**Result:** {entry['result']}")
