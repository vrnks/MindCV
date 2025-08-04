import spacy
import re

# Load spaCy model for NER-based section extraction
# nlp = spacy.load("en_cv_info_extr")
nlp = spacy.load("./models/en_cv_info_extr")

def extract_summary_fallback(text: str) -> str:
    """
    Attempts to extract a summary if NER fails to provide one.
    Uses heuristics: ignores contact info, selects first valid-looking sentences.
    """
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)

    summary_candidates = []
    for sent in sentences:
        sent_clean = sent.strip()
        if re.search(r"(@|http|www\.|linkedin|github|kyiv|ukraine|\+?\d{9,})", sent_clean.lower()):
            continue
        if sent_clean.isupper() or len(sent_clean.split()) <= 2:
            continue
        if sent_clean.lower() in ["resume", "curriculum vitae"]:
            continue

        summary_candidates.append(sent_clean)
        if len(summary_candidates) >= 2:
            break

    return " ".join(summary_candidates).strip()


def refine_position(sections: dict, raw_text: str) -> dict:
    """
    Refines the 'Position' field using heuristics:
    looks for target job titles in the summary or top lines of the raw text.
    """
    ner_pos = sections.get("Position", "").lower()
    summary = sections.get("Summary", "").lower()
    top_lines = "\n".join(raw_text.splitlines()[:10]).lower()

    target_roles = [
        'AI Engineer', 'AI Ethicist', 'AI Product Owner', 'AI Researcher', 'Assembler Developer',
        'Backend Developer', 'Backend Engineer', 'Business Analyst', 'Business Developer',
        'Business Development Manager', 'Business Intelligence Analyst', 'Business Systems Analyst',
        'C++/C Developer', 'Career Coach', 'Cloud Architect', 'Cloud Engineer', 'Cloud Infrastructure Engineer',
        'Cloud Solutions Architect', 'Community Manager', 'Community Outreach Officer', 'Compliance Analyst',
        'Compliance Officer', 'Computer Vision Engineer', 'Content Strategist', 'Content Writer',
        'Creative Strategist', 'Customer Support', 'Cybersecurity Analyst', 'DSP Engineer', 'Data Analyst',
        'Data Engineer', 'Data Operations Associate', 'Data Quality Analyst', 'Data Scientist',
        'Data Visualization Specialist', 'Database Administrator', 'Database Developer', 'DevOps Engineer',
        'DevOps Specialist', 'Embedded Engineer', 'Firmware Engineer', 'Freelance Marketer',
        'Frontend Developer', 'Fullstack Developer', 'Fullstack Engineer', 'HR Assistant', 'HR Manager',
        'HR Project Coordinator', 'HR Tech Partner', 'Hardware Engineer', 'IT Project Coordinator',
        'Innovation Lead', 'Innovation Strategist', 'ML Engineer', 'MLOps Engineer', 'Machine Learning Engineer',
        'Maintenance Coordinator', 'Marketing Analyst', 'Mediator', 'Mobile App Developer', 'Mobile Developer',
        'Mobile Software Engineer', 'Mobile UX Designer', 'NLP Engineer', 'Network Engineer',
        'Network Security Engineer', 'Operations Coordinator', 'Operations Manager', 'Operations Specialist',
        'Process Manager', 'Product Designer', 'Product Manager', 'Product Owner', 'Project Manager',
        'Proofreader', 'QA Automation Engineer', 'QA Engineer', 'QA Lead', 'QA Tester',
        'Quality Control Specialist', 'Retail Shift Supervisor', 'Sales Consultant', 'Sales Rep', 'Scrum Master',
        'Security Analyst', 'Security Engineer', 'Software Architect', 'Software Development Manager',
        'Software Engineer', 'Software Tester', 'Strategy Analyst', 'Subject Matter Expert',
        'System Architect', 'Systems Administrator', 'Systems Analyst', 'Systems Architect', 'Team Lead',
        'Tech Lead', 'Technical Expert', 'Technical Program Manager', 'Technical Recruiter',
        'Technical Support Engineer', 'Technical Support Specialist', 'Technical Writer', 'UI Developer',
        'UX Designer', 'UX Researcher', 'UX/UI Designer'
    ]

    for role in target_roles:
        if role.lower() in top_lines or role.lower() in summary:
            sections["Position"] = role
            return sections

    if ner_pos:
        sections["Position"] = ner_pos.title()
        return sections

    for line in raw_text.splitlines()[:10]:
        lower_line = line.strip().lower()
        for role in target_roles:
            if role.lower() in lower_line:
                sections["Position"] = role
                return sections

    return sections


def extract_sections_ner(text: str) -> dict:
    """
    Uses a spaCy-based NER model to extract resume sections.
    Applies fallback logic for summary and refined logic for position.
    """
    doc = nlp(text)
    blocks = {"Position": [], "SoftSkills": [], "Summary": [], "Projects": []}

    for ent in doc.ents:
        label = ent.label_
        word = ent.text
        if label in ["JOB_TITLE", "PROFILE"]:
            blocks["Position"].append(word)
        elif label == "SSKILL":
            blocks["SoftSkills"].append(word)
        elif label in ["PROJECT", "EXPERIENCE"]:
            blocks["Projects"].append(word)
        elif label in ["ABOUT", "SUMMARY", "DESCRIPTION"]:
            blocks["Summary"].append(word)

    result = {k: " ".join(sorted(set(v))) if v else "" for k, v in blocks.items()}

    if not result["Summary"]:
        result["Summary"] = extract_summary_fallback(text)

    result = refine_position(result, text)
    return result
