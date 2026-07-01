from core.models import AnalysisResult, Engine1Output
from knowledge_graph.resume_graph import ResumeGraph


class ResumeIQPipeline:
    """
    Orchestrates all 6 engines in sequence.
    Engines are imported lazily to avoid heavy model loading at import time.
    """

    def __init__(self):
        self._engines_loaded = False

    def _load_engines(self):
        from engines.resume_intelligence import ResumeIntelligenceEngine
        from engines.semantic_intelligence import SemanticIntelligenceEngine
        from engines.candidate_verification import CandidateVerificationEngine
        from engines.candidate_quality import CandidateQualityEngine
        from engines.recruiter_decision import RecruiterDecisionEngine
        from engines.explainability import ExplainabilityEngine

        self.e1 = ResumeIntelligenceEngine()
        self.e2 = SemanticIntelligenceEngine()
        self.e3 = CandidateVerificationEngine()
        self.e4 = CandidateQualityEngine()
        self.e5 = RecruiterDecisionEngine()
        self.e6 = ExplainabilityEngine()
        self._engines_loaded = True

    def run(self, resume_pdf: bytes, jd_text: str) -> AnalysisResult:
        if not self._engines_loaded:
            self._load_engines()

        e1_out = self.e1.run(resume_pdf, jd_text)
        graph = ResumeGraph().build(e1_out.parsed_resume)
        e2_out = self.e2.run(e1_out)
        e3_out = self.e3.run(e1_out, graph)
        e4_out = self.e4.run(e1_out, e3_out)
        e5_out = self.e5.run(e1_out, e2_out, e3_out, e4_out)
        e6_out = self.e6.run(e1_out, e2_out, e3_out, e4_out, e5_out)

        return AnalysisResult(
            resume_intelligence=e1_out,
            semantic_intelligence=e2_out,
            candidate_verification=e3_out,
            candidate_quality=e4_out,
            recruiter_decision=e5_out,
            explainability=e6_out,
        )

    def run_bulk(self, resume_pdfs: list[tuple[str, bytes]], jd_text: str) -> list[AnalysisResult]:
        """Analyze N resumes against one JD and return ranked results."""
        if not self._engines_loaded:
            self._load_engines()

        results = []
        for filename, pdf_bytes in resume_pdfs:
            result = self.run(pdf_bytes, jd_text)
            results.append((filename, result))

        ranked = self.e5.rank_bulk(results)
        return ranked
