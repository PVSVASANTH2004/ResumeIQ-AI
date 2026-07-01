import re
from io import BytesIO

import pdfplumber

from core.models import EducationEntry, ExperienceEntry, ParsedResume, ProjectEntry

# Section header patterns (order matters — more specific first)
_SECTION_PATTERNS = {
    "summary":     r"\b(summary|objective|profile|about me|professional summary)\b",
    "experience":  r"\b(experience|work experience|employment|work history|professional experience)\b",
    "projects":    r"\b(projects|personal projects|academic projects|key projects|portfolio)\b",
    "education":   r"\b(education|academic background|qualifications|academics)\b",
    "skills":      r"\b(skills|technical skills|core competencies|technologies|tools)\b",
    "certifications": r"\b(certifications|certificates|licenses|credentials)\b",
    "achievements": r"\b(achievements|awards|honors|accomplishments)\b",
    "publications": r"\b(publications|papers|research|conference)\b",
}

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"[\+\(]?[\d\s\-\(\)]{7,15}")
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)

# Duration patterns:  "Jan 2021 – Mar 2023"  "2020 - Present"
_DATE_RANGE_RE = re.compile(
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]*)?(\d{4})"
    r"\s*[–\-—to]+\s*"
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]*)?"
    r"(\d{4}|present|current|now)",
    re.I,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def _split_into_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current_section = "header"
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue

        matched_section = None
        for section_name, pattern in _SECTION_PATTERNS.items():
            # A line is a section header if it matches the pattern and is short
            if re.search(pattern, stripped, re.I) and len(stripped) < 60:
                matched_section = section_name
                break

        if matched_section:
            sections[current_section] = "\n".join(buffer).strip()
            current_section = matched_section
            buffer = []
        else:
            buffer.append(line)

    sections[current_section] = "\n".join(buffer).strip()
    return {k: v for k, v in sections.items() if v.strip()}


def _extract_name(header_text: str) -> str:
    """First non-empty line of the header that isn't a URL, email, or phone."""
    for line in header_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _EMAIL_RE.search(line) or _PHONE_RE.search(line) or _URL_RE.search(line):
            continue
        # Name usually has 2-4 words, all capitalised-ish
        words = line.split()
        if 1 < len(words) <= 5 and all(w[0].isupper() or not w[0].isalpha() for w in words if w):
            return line
    return ""


def _extract_contact(text: str) -> tuple[str, str]:
    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0).strip() if phone_match else ""
    return email, phone


def _parse_experience(section_text: str) -> list[ExperienceEntry]:
    entries: list[ExperienceEntry] = []
    blocks = re.split(r"\n(?=[A-Z])", section_text)

    for block in blocks:
        if not block.strip():
            continue

        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        role = lines[0] if lines else ""
        company = lines[1] if len(lines) > 1 else ""
        bullets = [l.lstrip("•–-●▪ ") for l in lines[2:] if l.startswith(("•", "-", "–", "●", "▪", "*"))]

        # Multi-column PDFs lose bullet characters during extraction.
        # Fall back to all non-date content lines that look like real bullets.
        if not bullets:
            bullets = [
                l for l in lines[2:]
                if len(l.split()) >= 4 and not _DATE_RANGE_RE.search(l)
            ]

        start_year, end_year, duration_months = 0, 0, 0
        date_match = _DATE_RANGE_RE.search(block)
        if date_match:
            start_year = int(date_match.group(2)) if date_match.group(2) else 0
            end_str = date_match.group(4) or ""
            if end_str.isdigit():
                end_year = int(end_str)
                duration_months = max(0, (end_year - start_year) * 12)
            else:
                from datetime import datetime
                end_year = datetime.now().year
                duration_months = max(0, (end_year - start_year) * 12)

        entries.append(ExperienceEntry(
            company=company,
            role=role,
            duration_months=duration_months,
            bullets=bullets,
            start_year=start_year or None,
            end_year=end_year or None,
        ))

    return entries


def _parse_projects(section_text: str) -> list[ProjectEntry]:
    entries: list[ProjectEntry] = []
    blocks = re.split(r"\n(?=[A-Z•\-])", section_text)

    for block in blocks:
        if not block.strip():
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        title = lines[0].lstrip("•–-● ")
        description = " ".join(lines[1:]) if len(lines) > 1 else ""

        # Extract tech mentions in parentheses or after pipe/colon
        tech_match = re.search(r"[\|(:](.+)$", title)
        techs: list[str] = []
        if tech_match:
            raw = tech_match.group(1)
            techs = [t.strip() for t in re.split(r"[,/]", raw) if t.strip()]
            title = title[: tech_match.start()].strip()

        entries.append(ProjectEntry(title=title, description=description, technologies=techs))

    return entries


def _parse_education(section_text: str) -> list[EducationEntry]:
    entries: list[EducationEntry] = []
    blocks = re.split(r"\n(?=[A-Z])", section_text)

    for block in blocks:
        if not block.strip():
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        institution = lines[0]
        degree = lines[1] if len(lines) > 1 else ""
        field = ""

        # "Bachelor of Science in Computer Science" → field = Computer Science
        field_match = re.search(r"\bin\s+(.+)", degree, re.I)
        if field_match:
            field = field_match.group(1).strip()

        year_match = _YEAR_RE.search(block)
        graduation_year = int(year_match.group()) if year_match else None

        entries.append(EducationEntry(
            institution=institution,
            degree=degree,
            field=field,
            graduation_year=graduation_year,
        ))

    return entries


def _extract_raw_skills(skills_text: str) -> list[str]:
    raw = re.sub(r"[•|\n\t]", ",", skills_text)
    skills = [s.strip() for s in re.split(r"[,;/]", raw) if len(s.strip()) > 1]
    return list(dict.fromkeys(skills))  # preserve order, deduplicate


class ResumeParser:
    def parse(self, pdf_bytes: bytes) -> ParsedResume:
        raw_text = _extract_text_from_pdf(pdf_bytes)
        sections = _split_into_sections(raw_text)

        header_text = sections.get("header", raw_text[:400])
        name = _extract_name(header_text)
        email, phone = _extract_contact(raw_text)

        experience = _parse_experience(sections.get("experience", ""))
        projects = _parse_projects(sections.get("projects", ""))
        education = _parse_education(sections.get("education", ""))
        skills_raw = _extract_raw_skills(sections.get("skills", ""))

        return ParsedResume(
            name=name,
            email=email,
            phone=phone,
            raw_text=raw_text,
            sections=sections,
            skills_raw=skills_raw,
            experience=experience,
            projects=projects,
            education=education,
        )
