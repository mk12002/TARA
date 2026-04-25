"""
Visualization Component
==========================
Renders reasoning chain visualizations for RCA results.
"""

import streamlit as st
from typing import Dict, List


def render_reasoning_chain(reasoning_data: Dict):
    """Render a step-by-step reasoning chain visualization."""
    steps = reasoning_data.get("steps", [])

    st.subheader("🧠 Reasoning Chain")

    for step in steps:
        status = "✅" if step.get("status") == "success" else "❌"
        st.markdown(f"{status} **{step.get('step', 'Unknown')}**")
        if "error" in step:
            st.error(step["error"])

    # Hypotheses
    hypotheses = reasoning_data.get("hypotheses", [])
    if hypotheses:
        st.subheader("📊 Hypotheses")
        for h in hypotheses:
            prob = h.get("probability", 0)
            st.progress(min(prob, 1.0), text=f"{h.get('description', '')} ({prob:.0%})")

    # Remediation
    remediation = reasoning_data.get("remediation", [])
    if remediation:
        st.subheader("🔧 Remediation Steps")
        for step in remediation:
            st.markdown(f"- {step}")
