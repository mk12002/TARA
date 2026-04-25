"""
Source Display Component
==========================
Displays retrieved sources with spec references and sections.
"""

import streamlit as st
from typing import Dict, List


def render_sources(sources: List[Dict]):
    """Render source cards in an expander."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for src in sources:
            spec = src.get("spec_number", "")
            section = src.get("section", "")
            score = src.get("score", 0)

            st.markdown(
                f"**[Source {src.get('source_id', '?')}]** {spec} — {section}  \n"
                f"*Relevance: {score:.3f}*"
            )
            st.caption(src.get("text", "")[:300])
            st.divider()
