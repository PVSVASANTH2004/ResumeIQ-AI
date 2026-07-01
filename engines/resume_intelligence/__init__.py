from core.models import Engine1Output

from .ats_checker import ATSChecker
from .duplicate_skills import deduplicate
from .jd_parser import JDParser
from .keyword_stuffing import detect_keyword_stuffing
from .resume_classifier import ResumeClassifier
from .resume_parser import ResumeParser
from .skill_extractor import SkillExtractor


class ResumeIntelligenceEngine:
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.jd_parser = JDParser()
        self.skill_extractor = SkillExtractor()
        self.ats_checker = ATSChecker()
        self.classifier = ResumeClassifier()

    def run(self, resume_pdf: bytes, jd_text: str) -> Engine1Output:
        parsed_resume = self.resume_parser.parse(resume_pdf)
        parsed_jd = self.jd_parser.parse(jd_text)

        extracted_skills = self.skill_extractor.extract(parsed_resume)

        extracted_skills.programming = deduplicate(extracted_skills.programming)
        extracted_skills.frameworks  = deduplicate(extracted_skills.frameworks)
        extracted_skills.cloud       = deduplicate(extracted_skills.cloud)
        extracted_skills.databases   = deduplicate(extracted_skills.databases)
        extracted_skills.ai_ml       = deduplicate(extracted_skills.ai_ml)
        extracted_skills.devops      = deduplicate(extracted_skills.devops)
        extracted_skills.soft_skills = deduplicate(extracted_skills.soft_skills)

        ats_issues = self.ats_checker.check(resume_pdf)
        stuffing_risk, stuffing_detected = detect_keyword_stuffing(parsed_resume)
        predicted_role = self.classifier.predict(parsed_resume)

        return Engine1Output(
            parsed_resume=parsed_resume,
            parsed_jd=parsed_jd,
            extracted_skills=extracted_skills,
            predicted_role=predicted_role,
            ats_issues=ats_issues,
            stuffing_risk=stuffing_risk,
            keyword_stuffing_detected=stuffing_detected,
        )
