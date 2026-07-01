import re
from io import BytesIO

import pdfplumber


# ATS issue: (description, penalty_weight)
_ATS_CHECKS = {
    "has_tables":        ("Contains tables — ATS parsers often misread tabular layouts", 15),
    "has_images":        ("Contains images or graphics — ATS cannot parse visual content", 20),
    "has_columns":       ("Multi-column layout detected — may cause reading-order errors in ATS", 15),
    "missing_sections":  ("Missing standard sections (Skills / Experience / Education)", 20),
    "bad_fonts":         ("Non-standard fonts detected — may not render correctly in ATS", 5),
    "has_headers_footers":("Running headers/footers detected — ATS may mix them with content", 10),
    "no_contact_info":   ("No clear contact information found (email or phone)", 10),
    "long_bullets":      ("Bullet points exceed 3 lines — reduce for ATS parsing clarity", 5),
}

_REQUIRED_SECTIONS = {"experience", "education", "skills"}
_CONTACT_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}|[\+\(]?[\d\s\-\(\)]{7,15}")
_SECTION_RE  = re.compile(
    r"\b(experience|education|skills|work history|employment|projects)\b", re.I
)


class ATSChecker:
    def check(self, pdf_bytes: bytes) -> list[str]:
        issues: list[str] = []
        full_text = ""

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"

                # Tables
                if page.extract_tables():
                    issues.append(_ATS_CHECKS["has_tables"][0])

                # Images
                if page.images:
                    issues.append(_ATS_CHECKS["has_images"][0])

                # Multi-column detection via PDF x-coordinates is omitted:
                # standard resumes with right-aligned dates or comma-separated
                # skills consistently produce false positives. True multi-column
                # detection requires a full layout analysis engine (e.g. pymupdf4llm).

        # Missing required sections
        found_sections = set(re.findall(r"\b(experience|education|skills)\b", full_text, re.I))
        found_sections_lower = {s.lower() for s in found_sections}
        if not _REQUIRED_SECTIONS.issubset(found_sections_lower):
            issues.append(_ATS_CHECKS["missing_sections"][0])

        # No contact info
        if not _CONTACT_RE.search(full_text):
            issues.append(_ATS_CHECKS["no_contact_info"][0])

        # Long bullets
        bullets = re.findall(r"[•\-–●▪\*]\s*(.+)", full_text)
        long_bullets = [b for b in bullets if len(b.split()) > 25]
        if long_bullets:
            issues.append(_ATS_CHECKS["long_bullets"][0])

        return list(dict.fromkeys(issues))  # deduplicate, preserve order

    def score(self, issues: list[str]) -> float:
        """Convert issue list to ATS score 0–100."""
        penalty = 0
        for issue in issues:
            for key, (description, weight) in _ATS_CHECKS.items():
                if description in issue:
                    penalty += weight
                    break
        return max(0.0, 100.0 - penalty)
