"""Build paired text inputs without using the category label."""

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from itertools import islice

from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


HEADER_CLASSES = {"name", "resumetitle", "exampletitle", "name-sec",
                  "namesec", "inheader", "name-line"}
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()]+")
PHONE = re.compile(
    r"(?<![\w+])(?:\+?1[ .-]?)?(?:\([2-9]\d{2}\)[ .-]?|[2-9]\d{2}[ .-])"
    r"[2-9]\d{2}[ .-]\d{4}(?!\w)"
)
LABELED_PHONE = re.compile(
    r"(?P<label>\b(?:phone|telephone|mobile|fax|cell|tel)"
    r"(?:[ \t]+(?:number|no\.?))?[ \t]*:[ \t]*)"
    r"(?:\+?\d{2,5}[ .-]\d{6,10}|\+?\d{7,15})(?!\w)",
    re.IGNORECASE,
)
INTERNATIONAL_PHONE = re.compile(
    r"(?<![\w+])\+[2-9]\d{0,2}[ .-](?:\d{2,5}[ .-]\d{5,9}|\d{7,12})(?!\w)"
)
NANP_DIGITS = r"[2-9]\d{2}[2-9]\d{6}"
CONTACT_CONTEXT_PHONE = re.compile(
    rf"(?<![\w.-])(?:{NANP_DIGITS}(?=[ \t]+(?:CONTACT_EMAIL\b|Principal[ \t]*:))|"
    rf"\({NANP_DIGITS}\)(?=[ \t]+Okay to contact\b))",
    re.IGNORECASE,
)
MISDECODED_PUNCTUATION = {
    "\u00e2\u20ac\u00a2": "\u2022",
    "\u00e2\u20ac\u2122": "\u2019",
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00c2\u00ae": "\u00ae",
    "\u0092": "\u2019",
}
OBSERVED_LAYOUT_REPAIRS = {
    "M onitored": "Monitored",
    "d evelop": "develop",
    "con tracts": "contracts",
    "SchoolDiploma": "School Diploma",
    "educationEducator": "education Educator",
}
LAYOUT_ARTIFACT = re.compile(r"\b(?:" + "|".join(map(re.escape, OBSERVED_LAYOUT_REPAIRS)) + r")\b")
TOKEN = re.compile(r"[^\W_]+(?:_[^\W_]+)*(?:['’][^\W_]+)?", re.UNICODE)
STOP_WORDS = ENGLISH_STOP_WORDS.difference({"no", "nor", "not", "cannot", "c", "r", "go"})
CONTACT_TOKENS = {"contact_email", "contact_url", "contact_phone"}
ALIASES = (
    (r"c\+\+", "cplusplus"), (r"c#", "csharp"),
    (r"asp\.net", "asp_dotnet"), (r"\.net", "dotnet"),
    (r"node\.js", "nodejs"), (r"power[\s-]*bi", "power_bi"),
    (r"machine[\s-]+learning|(?-i:ML)", "machine_learning"),
    (r"javascript|js", "javascript"),
)
SKILL_PATTERNS = [
    (re.compile(rf"(?<![\w.])(?:{pattern})(?!\w)", re.IGNORECASE), target)
    for pattern, target in ALIASES
]
PROTECTED = CONTACT_TOKENS.union(target for _, target in ALIASES)
STEMMER = EnglishStemmer()
TEXT_COLUMNS = (
    "Resume_str", "text_raw_no_header", "text_light", "text_light_no_header",
    "text_classical", "text_classical_no_header", "text_stemmed",
    "text_stemmed_no_header",
)
LEMMA_COLUMNS = ("text_lemmatized", "text_lemmatized_no_header")


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class HeaderParser(HTMLParser):
    """Read the first explicit header element in a resume template."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tag = None
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if self.tag is None:
            attributes = dict(attrs)
            classes = (attributes.get("class") or "").lower().split()
            if HEADER_CLASSES.intersection(classes):
                self.tag = tag
                self.depth = 1
        elif self.depth and tag == self.tag:
            self.depth += 1

    def handle_endtag(self, tag):
        if self.depth and tag == self.tag:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth and data.strip():
            self.parts.append(data)


@dataclass(frozen=True)
class HeaderResult:
    body: str
    status: str
    header: str = ""


def remove_header(text, html_text):
    if not text.strip():
        return HeaderResult(text, "blank_text")

    parser = HeaderParser()
    try:
        parser.feed(html_text)
        parser.close()
    except (AssertionError, ValueError):
        return HeaderResult(text, "invalid_html")

    candidate = " ".join(" ".join(parser.parts).split())
    if parser.tag is None or not candidate:
        return HeaderResult(text, "missing_header")
    if parser.depth or len(candidate) > 110 or any(
        pattern.search(candidate) for pattern in (EMAIL, URL, PHONE)
    ):
        return HeaderResult(text, "uncertain_header")

    words = canonical(candidate).split()
    if len(words) > 10 or not any(char.isalpha() for char in candidate):
        return HeaderResult(text, "uncertain_header")

    prefix = list(islice(re.finditer(r"\S+", text), len(words)))
    if [canonical(match.group()) for match in prefix] != words:
        return HeaderResult(text, "unaligned_header")

    body = text[prefix[-1].end():].lstrip()
    if len(body.split(maxsplit=9)) < 10:
        return HeaderResult(text, "short_body")
    return HeaderResult(body, "removed", text[prefix[0].start():prefix[-1].end()])


def normalize_and_mask(text):
    for sequence, punctuation in MISDECODED_PUNCTUATION.items():
        text = text.replace(sequence, punctuation)
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(str.maketrans({"\u200b": None, "\ufeff": None,
                                        "•": " ", "●": " ", "▪": " ", "◦": " ", "\ufffd": " "}))
    text, emails = EMAIL.subn(" CONTACT_EMAIL ", text)

    def replace_url(match):
        value = match.group()
        core = value.rstrip(".,;:!?")
        return " CONTACT_URL " + value[len(core):]

    text, urls = URL.subn(replace_url, text)
    text, phones = mask_phones(text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = LAYOUT_ARTIFACT.sub(lambda match: OBSERVED_LAYOUT_REPAIRS[match.group()], text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text, {"email": emails, "url": urls, "phone": phones}


def mask_phones(text):
    text, labeled = LABELED_PHONE.subn(r"\g<label> CONTACT_PHONE ", text)
    text, contextual = CONTACT_CONTEXT_PHONE.subn(" CONTACT_PHONE ", text)
    international = 0

    def replace_international(match):
        nonlocal international
        number = match.group()
        if 8 <= sum(char.isdigit() for char in number) <= 15:
            international += 1
            return " CONTACT_PHONE "
        return number

    text = INTERNATIONAL_PHONE.sub(replace_international, text)
    text, north_american = PHONE.subn(" CONTACT_PHONE ", text)
    return text, labeled + contextual + international + north_american


def normalize_skills(text):
    for pattern, replacement in SKILL_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def classical_tokens(text):
    text = normalize_skills(text)
    return [token for token in TOKEN.findall(text.casefold()) if token not in STOP_WORDS]


def preprocess(text, header):
    """Return row-aligned raw, light, classical and Snowball variants."""
    variants = {}
    full_mask_counts = {}
    for suffix, raw in (("", text), ("_no_header", header.body)):
        light, masks = normalize_and_mask(raw)
        if not suffix:
            full_mask_counts = masks
        tokens = classical_tokens(light)
        variants[f"text_light{suffix}"] = light
        variants[f"text_classical{suffix}"] = " ".join(tokens)
        variants[f"text_stemmed{suffix}"] = " ".join(
            token if token in PROTECTED else STEMMER.stem(token) for token in tokens
        )
    return {"text_raw_no_header": header.body, **variants}, full_mask_counts


def label_phrase_present(text, category):
    """Measure a literal label cue for EDA without affecting preprocessing."""
    words = " " + " ".join(re.findall(r"[^\W_]+", text.casefold())) + " "
    label = " ".join(re.findall(r"[^\W_]+", category.casefold()))
    return f" {label} " in words


def add_lemmatized_text(rows):
    import spacy

    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "senter"])
    for suffix in ("", "_no_header"):
        texts = (normalize_skills(row[f"text_light{suffix}"]) for row in rows)
        for row, doc in zip(rows, nlp.pipe(texts, batch_size=24), strict=True):
            lemmas = []
            for token in doc:
                word = token.text.casefold()
                if word in PROTECTED:
                    lemmas.append(word)
                elif TOKEN.fullmatch(word) and word not in STOP_WORDS:
                    lemma = token.lemma_.casefold()
                    if lemma and lemma not in STOP_WORDS:
                        lemmas.append(lemma)
            row[f"text_lemmatized{suffix}"] = " ".join(lemmas)
    return {"pipeline": nlp.meta["name"], "version": nlp.meta["version"]}
