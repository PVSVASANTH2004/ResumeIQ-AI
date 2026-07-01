from core.models import Engine1Output, Engine3Output
from engines.semantic_intelligence.semantic_matcher import SemanticMatcher
from knowledge_graph.resume_graph import ResumeGraph

from .evidence_validator import compute_evidence_scores
from .hallucination_detector import HallucinationDetector
from .skill_confidence import compute_skill_confidence


class CandidateVerificationEngine:
    def __init__(self):
        self.detector = HallucinationDetector()
        self._matcher = SemanticMatcher()

    def run(self, e1: Engine1Output, graph: ResumeGraph) -> Engine3Output:
        all_skills = e1.extracted_skills.all_skills

        evidence_scores = compute_evidence_scores(e1, graph)
        confidence = compute_skill_confidence(evidence_scores, e1, self._matcher)
        hallucinated = self.detector.detect(
            all_skills, evidence_scores, confidence, e1.stuffing_risk
        )

        return Engine3Output(
            evidence_scores=evidence_scores,
            skill_confidence=confidence,
            hallucinated_skills=hallucinated,
        )
