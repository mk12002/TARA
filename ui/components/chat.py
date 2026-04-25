"""
Chat Component
=================
Streamlit chat interface component.
"""

import streamlit as st
from typing import Dict


def render_chat_message(role: str, content: str, metadata: Dict = None):
    """Render a single chat message with metadata."""
    with st.chat_message(role):
        st.markdown(content)

        if metadata:
            cols = st.columns(3)
            with cols[0]:
                conf = metadata.get("confidence", "N/A")
                color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "⚪")
                st.caption(f"{color} Confidence: {conf}")
            with cols[1]:
                st.caption(f"📚 Sources: {metadata.get('citation_count', 0)}")
            with cols[2]:
                st.caption(f"🎯 Intent: {metadata.get('intent', 'QNA')}")
