"""Rule-based skill matching between a resume and a job description (demo feature, not a trained model)."""
import re

SKILLS = sorted({
    # programming / data
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "nosql", "html", "css", "php", "ruby", "go",
    "r", "matlab", "scala", "kotlin", "swift", "react", "angular", "vue", "node.js", "django", "flask", "spring",
    "asp.net", ".net", "rest api", "graphql", "git", "docker", "kubernetes", "aws", "azure", "gcp", "linux",
    "ci/cd", "jenkins", "terraform", "mysql", "postgresql", "mongodb", "oracle", "redis", "kafka", "spark", "hadoop",
    "machine learning", "deep learning", "nlp", "computer vision", "pytorch", "tensorflow", "scikit-learn", "pandas",
    "numpy", "data analysis", "data visualization", "statistics", "tableau", "power bi", "excel", "etl",
    "big data", "data warehousing", "cybersecurity", "networking", "agile", "scrum", "jira", "devops", "testing",
    "unit testing", "qa", "selenium", "ui/ux", "figma", "photoshop", "illustrator", "autocad", "solidworks",
    # business / finance / hr / other fields
    "project management", "product management", "stakeholder management", "budgeting", "forecasting",
    "financial analysis", "financial reporting", "accounting", "auditing", "bookkeeping", "payroll", "taxation",
    "quickbooks", "sap", "erp", "crm", "salesforce", "sales", "business development", "negotiation", "marketing",
    "digital marketing", "seo", "social media", "content writing", "copywriting", "public relations", "branding",
    "market research", "customer service", "account management", "recruitment", "talent acquisition", "onboarding",
    "employee relations", "performance management", "training", "compliance", "risk management", "procurement",
    "supply chain", "logistics", "inventory management", "quality assurance", "quality control", "lean",
    "six sigma", "operations management", "strategic planning", "leadership", "team management", "communication",
    "problem solving", "teaching", "curriculum development", "lesson planning", "classroom management",
    "patient care", "nursing", "cpr", "healthcare", "food safety", "menu planning", "construction", "mechanical",
    "electrical", "maintenance", "safety", "legal research", "contract management", "litigation", "banking",
    "investment", "insurance", "aviation", "fitness", "personal training", "graphic design", "video editing",
})

_PATTERNS = {s: re.compile(r"(?<![a-z0-9+#.])" + re.escape(s) + r"(?![a-z0-9+#])") for s in SKILLS}


def extract_skills(text):
    t = " " + re.sub(r"\s+", " ", str(text).lower()) + " "
    return {s for s, p in _PATTERNS.items() if p.search(t)}


def match(resume_text, jd_text):
    """Skills the JD asks for, split into those found / not found in the resume."""
    need, have = extract_skills(jd_text), extract_skills(resume_text)
    matched, missing = sorted(need & have), sorted(need - have)
    coverage = len(matched) / len(need) if need else None
    return dict(matched=matched, missing=missing, coverage=coverage, extra=sorted(have - need))


def tfidf_similarity(resume_text, jd_text):
    """Cosine similarity of TF-IDF vectors of the two texts (0-1). A rough signal only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform([resume_text, jd_text])
        return float(cosine_similarity(m[0], m[1])[0, 0])
    except ValueError:
        return 0.0
