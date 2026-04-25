"""
TARA Streamlit UI
====================
Interactive web interface for the Telecom RAG Assistant.
"""

import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="TARA - Telecom RAG Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .source-card {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #0f9b8e;
    }
    .confidence-high { color: #00d26a; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #ff4757; }
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown("# 📡 TARA")
st.markdown("**Telecom Agentic RAG Assistant** — AI-powered telecom knowledge assistant")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources", 1, 20, 10)
    use_reranker = st.checkbox("Use reranker", value=True)
    st.divider()
    st.markdown("### 📊 Model Info")
    st.markdown("- **LLM**: Llama-3.2-3B (Ollama)")
    st.markdown("- **Embeddings**: bge-small-en-v1.5")
    st.markdown("- **Reranker**: MiniLM-L-6-v2")
    st.markdown("- **Vector DB**: Qdrant (local)")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a telecom question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                import httpx
                response = httpx.post(
                    "http://localhost:8080/api/v1/query",
                    json={"query": prompt, "top_k": top_k, "use_reranker": use_reranker},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Answer
                    st.markdown(data["answer"])

                    # Confidence badge
                    conf = data.get("confidence", "MEDIUM")
                    conf_class = f"confidence-{conf.lower()}"
                    st.markdown(
                        f"**Confidence**: <span class='{conf_class}'>{conf}</span> "
                        f"| **Intent**: {data.get('intent', 'QNA')} "
                        f"| **Sources cited**: {data.get('citation_count', 0)}",
                        unsafe_allow_html=True,
                    )

                    # Sources expander
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for src in sources:
                                st.markdown(
                                    f"<div class='source-card'>"
                                    f"<strong>[Source {src['source_id']}]</strong> "
                                    f"{src.get('spec_number', '')} — {src.get('section', '')}<br>"
                                    f"<small>{src['text'][:200]}...</small>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    answer_text = data["answer"]
                else:
                    answer_text = f"Error: {response.text}"
                    st.error(answer_text)

            except Exception as e:
                answer_text = f"⚠️ Could not connect to TARA backend. Make sure the API is running.\n\nError: {e}"
                st.warning(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
