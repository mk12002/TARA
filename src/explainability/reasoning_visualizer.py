"""
Reasoning Visualizer
======================
Visualizes the reasoning chain for explainability.
"""

from typing import Dict, List

from src.reasoning.reasoning_chain import ReasoningState


class ReasoningVisualizer:
    """Formats reasoning chains for UI display."""

    def to_text(self, state: ReasoningState) -> str:
        """Convert reasoning state to readable text."""
        lines = []
        lines.append("═" * 50)
        lines.append("  REASONING CHAIN")
        lines.append("═" * 50)

        for step in state.step_history:
            status = "✓" if step["status"] == "success" else "✗"
            lines.append(f"  {status} {step['step']}")
            if "error" in step:
                lines.append(f"      Error: {step['error']}")

        lines.append("")

        if state.root_cause:
            lines.append(f"  Root Cause: {state.root_cause}")
            lines.append(f"  Confidence: {state.confidence:.1%}")

        if state.hypotheses:
            lines.append(f"\n  Hypotheses ({len(state.hypotheses)}):")
            for h in state.hypotheses[:5]:
                prob = h.posterior or h.probability
                lines.append(f"    • {h.description} (p={prob:.2f})")

        if state.remediation_steps:
            lines.append(f"\n  Remediation:")
            for step in state.remediation_steps:
                lines.append(f"    {step}")

        lines.append("═" * 50)
        return "\n".join(lines)

    def to_dict(self, state: ReasoningState) -> Dict:
        """Convert reasoning state to dict for API/UI."""
        return {
            "steps": state.step_history,
            "root_cause": state.root_cause,
            "confidence": state.confidence,
            "hypotheses": [
                {
                    "id": h.id,
                    "description": h.description,
                    "probability": h.posterior or h.probability,
                    "supporting_evidence": h.supporting_evidence,
                    "contradicting_evidence": h.contradicting_evidence,
                }
                for h in state.hypotheses
            ],
            "remediation": state.remediation_steps,
        }
