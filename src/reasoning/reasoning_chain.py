"""
Reasoning Chain
==================
Multi-step reasoning orchestrated via LangGraph state machine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger


class ReasoningStep(str, Enum):
    SYMPTOM_ANALYSIS = "symptom_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EVIDENCE_GATHERING = "evidence_gathering"
    ROOT_CAUSE_IDENTIFICATION = "root_cause_identification"
    REMEDIATION = "remediation"
    COMPLETE = "complete"


@dataclass
class ReasoningState:
    """State maintained across the reasoning chain."""
    current_step: ReasoningStep = ReasoningStep.SYMPTOM_ANALYSIS
    problem_description: str = ""
    problem_info: Dict = field(default_factory=dict)
    hypotheses: List = field(default_factory=list)
    root_cause: str = ""
    confidence: float = 0.0
    remediation_steps: List[str] = field(default_factory=list)
    step_history: List[Dict] = field(default_factory=list)


class ReasoningChain:
    """
    Multi-step reasoning chain for complex queries.
    
    Steps:
    1. Symptom Analysis → parse problem into structured form
    2. Hypothesis Generation → generate candidate causes
    3. Evidence Gathering → retrieve evidence for/against each
    4. Root Cause Identification → select highest-posterior hypothesis
    5. Remediation → generate fix steps from KB
    
    Note: For full LangGraph integration, replace the run() method
    with a StateGraph. This implementation serves as the foundation.
    """

    def __init__(self, rca_agent=None):
        self.rca_agent = rca_agent

    def run(self, problem_description: str) -> ReasoningState:
        """Execute the reasoning chain."""
        state = ReasoningState(problem_description=problem_description)

        steps = [
            (ReasoningStep.SYMPTOM_ANALYSIS, self._symptom_analysis),
            (ReasoningStep.HYPOTHESIS_GENERATION, self._hypothesis_generation),
            (ReasoningStep.EVIDENCE_GATHERING, self._evidence_gathering),
            (ReasoningStep.ROOT_CAUSE_IDENTIFICATION, self._root_cause_identification),
            (ReasoningStep.REMEDIATION, self._remediation),
        ]

        for step_name, step_fn in steps:
            state.current_step = step_name
            logger.info(f"Reasoning step: {step_name.value}")

            try:
                state = step_fn(state)
                state.step_history.append({
                    "step": step_name.value,
                    "status": "success",
                })
            except Exception as e:
                logger.error(f"Step {step_name.value} failed: {e}")
                state.step_history.append({
                    "step": step_name.value,
                    "status": "failed",
                    "error": str(e),
                })
                break

        state.current_step = ReasoningStep.COMPLETE
        return state

    def _symptom_analysis(self, state: ReasoningState) -> ReasoningState:
        """Parse the problem description into structured info."""
        # Simple keyword extraction (upgrade to LLM for production)
        desc_lower = state.problem_description.lower()
        state.problem_info = {
            "summary": state.problem_description[:200],
            "symptom": state.problem_description,
            "network_element": "unknown",
            "procedure": "unknown",
        }

        # Try to detect network element
        for element in ["gnb", "enb", "cell", "ue", "core", "transport"]:
            if element in desc_lower:
                state.problem_info["network_element"] = element
                break

        return state

    def _hypothesis_generation(self, state: ReasoningState) -> ReasoningState:
        """Generate hypotheses using the RCA agent."""
        if self.rca_agent:
            result = self.rca_agent.analyze(state.problem_description)
            state.hypotheses = result.hypotheses
        return state

    def _evidence_gathering(self, state: ReasoningState) -> ReasoningState:
        """Gather evidence — placeholder for retrieval integration."""
        # Evidence gathering happens inside RCA agent
        return state

    def _root_cause_identification(self, state: ReasoningState) -> ReasoningState:
        """Select the highest-confidence hypothesis."""
        if state.hypotheses:
            best = max(state.hypotheses, key=lambda h: h.posterior or h.probability)
            state.root_cause = best.description
            state.confidence = best.posterior or best.probability
        return state

    def _remediation(self, state: ReasoningState) -> ReasoningState:
        """Generate remediation steps — placeholder for LLM formatting."""
        if state.root_cause:
            state.remediation_steps = [
                f"1. Verify: Confirm '{state.root_cause}' by checking related alarms",
                f"2. Isolate: Identify affected cells/elements",
                f"3. Remediate: Apply corrective action per 3GPP best practices",
                f"4. Validate: Monitor KPIs to confirm resolution",
            ]
        return state
