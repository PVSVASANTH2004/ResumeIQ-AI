from core.models import Engine1Output, Engine3Output, Engine4Output

from .experience_quality import compute_experience_quality
from .learning_progression import compute_learning_progression, progression_to_score
from .project_complexity import ProjectComplexityModel
from .project_novelty import ProjectNoveltyScorer
from knowledge_graph.resume_graph import ResumeGraph


class CandidateQualityEngine:
    def __init__(self):
        self.complexity_model = ProjectComplexityModel()
        self.novelty_scorer = ProjectNoveltyScorer()

    def run(self, e1: Engine1Output, e3: Engine3Output) -> Engine4Output:
        projects = e1.parsed_resume.projects
        experience = e1.parsed_resume.experience

        project_scores = self.complexity_model.predict(projects)

        novelty_scores = self.novelty_scorer.score(projects)
        for ps, novelty in zip(project_scores, novelty_scores):
            ps.novelty_score = novelty

        exp_quality = compute_experience_quality(experience)

        graph = ResumeGraph().build(e1.parsed_resume)
        progression = compute_learning_progression(graph)

        if project_scores:
            avg_complexity = sum(
                self.complexity_model.complexity_to_score(ps.complexity)
                for ps in project_scores
            ) / len(project_scores)
        else:
            avg_complexity = 0.0

        return Engine4Output(
            project_scores=project_scores,
            experience_quality_score=exp_quality,
            learning_progression=progression,
            avg_project_complexity_score=round(avg_complexity, 4),
        )
