"""
RCA Agent
============
Root Cause Analysis agent using structured hypothesis-evidence framework.
Python-driven logic with LLM as formatter, NOT as primary reasoner.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class Hypothesis:
    """An RCA hypothesis with evidence."""
    id: str
    description: str
    probability: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    posterior: float = 0.0


@dataclass
class RCAResult:
    """Result of root cause analysis."""
    root_cause: str
    confidence: float
    hypotheses: List[Hypothesis]
    evidence_chain: List[str]
    remediation_steps: List[str] = field(default_factory=list)


# Pre-defined failure patterns for common telecom issues
FAILURE_PATTERNS: Dict[str, Dict] = {
    "packet_loss": {
        "symptoms": ["packet loss", "high bler", "retransmission"],
        "common_causes": [
            {"cause": "RF interference from adjacent cells", "indicators": ["high interference", "poor SINR"]},
            {"cause": "Congestion due to PRB exhaustion", "indicators": ["high PRB utilization", "capacity alarm"]},
            {"cause": "Hardware failure (RRU/antenna)", "indicators": ["hardware alarm", "VSWR alarm"]},
            {"cause": "Configuration error (wrong power settings)", "indicators": ["parameter change", "recent config"]},
        ],
    },
    "handover_failure": {
        "symptoms": ["handover failure", "ho failure", "call drop during handover"],
        "common_causes": [
            {"cause": "Missing neighbor relation", "indicators": ["ANR log", "neighbor list"]},
            {"cause": "Coverage gap between cells", "indicators": ["RSRP measurement", "coverage hole"]},
            {"cause": "Parameter mismatch (hysteresis, TTT)", "indicators": ["parameter settings", "threshold"]},
            {"cause": "Timing advance issue", "indicators": ["TA value", "cell radius"]},
        ],
    },
    "throughput_degradation": {
        "symptoms": ["low throughput", "throughput degradation", "slow speed"],
        "common_causes": [
            {"cause": "High interference (pilot pollution)", "indicators": ["SINR", "interference level"]},
            {"cause": "Suboptimal MCS selection", "indicators": ["CQI", "MCS distribution"]},
            {"cause": "Scheduling inefficiency", "indicators": ["PRB utilization", "scheduler logs"]},
            {"cause": "Transport backhaul bottleneck", "indicators": ["backhaul utilization", "latency"]},
        ],
    },
}


class RCAAgent:
    """
    Structured RCA agent.
    
    Strategy: Pattern-match against known failure modes first (Python),
    then use retrieval + LLM only as a grounded fallback.
    """

    def __init__(self, retrieval_pipeline=None, llm_client=None):
        self.retrieval = retrieval_pipeline
        self.llm = llm_client
        self.failure_patterns = FAILURE_PATTERNS

    def analyze(self, description: str) -> RCAResult:
        """
        Perform root cause analysis on a problem description.
        
        Steps:
        1. Parse symptoms from description
        2. Match against known failure patterns (Python)
        3. Generate hypotheses
        4. Gather evidence from retrieval
        5. Rank by posterior probability
        6. Return top hypothesis with evidence chain
        """
        logger.info(f"RCA analysis: {description[:80]}...")

        # Step 1-2: Generate hypotheses from patterns
        hypotheses = self._generate_hypotheses(description)

        if not hypotheses:
            return RCAResult(
                root_cause="Unable to determine root cause from available patterns",
                confidence=0.0,
                hypotheses=[],
                evidence_chain=["No matching failure patterns found"],
            )

        # Step 3: Set initial posteriors (simplified — no retrieval yet)
        for h in hypotheses:
            h.posterior = h.probability

        # Step 4: Sort by posterior
        hypotheses.sort(key=lambda h: h.posterior, reverse=True)

        best = hypotheses[0]

        return RCAResult(
            root_cause=best.description,
            confidence=best.posterior,
            hypotheses=hypotheses,
            evidence_chain=[f"Matched pattern: {best.id}"] + best.supporting_evidence,
        )

    def _generate_hypotheses(self, description: str) -> List[Hypothesis]:
        """Generate hypotheses from known failure patterns."""
        desc_lower = description.lower()
        hypotheses = []

        for pattern_name, pattern in self.failure_patterns.items():
            if any(s in desc_lower for s in pattern["symptoms"]):
                for i, cause in enumerate(pattern["common_causes"]):
                    hypotheses.append(Hypothesis(
                        id=f"{pattern_name}_{i}",
                        description=cause["cause"],
                        probability=1.0 / (i + 1),  # Higher prob for first causes
                        supporting_evidence=[],
                        contradicting_evidence=[],
                    ))

        logger.info(f"Generated {len(hypotheses)} hypotheses from patterns")
        return hypotheses[:5]
