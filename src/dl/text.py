"""Light text preparation for the RNN/CNN models.

The full preprocessing pipeline (stop-words, lemmatisation, skill synonyms) lives in
src/preprocess.py (data + preprocessing owner). If you feed already-preprocessed text,
run train.py with --no-clean so it is only tokenised here.
"""
import re

# Section words that usually follow the job title at the top of a resume.
# Case-sensitive on purpose: the title itself is in capitals ("HR ADMINISTRATOR").
SECTION = re.compile(
    r"\b(Professional Summary|Executive Profile|Career Overview|Summary|Profile|Objective|"
    r"Experience|Skills|Highlights|Qualifications)\b"
)
EMAIL = re.compile(r"\S+@\S+\.\S+")
URL = re.compile(r"(https?://\S+|www\.\S+)")
PHONE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")
# keeps tech tokens such as c++, c#, node.js, asp.net; masks stay single tokens
TOKEN = re.compile(r"<email>|<url>|<phone>|[a-z0-9]+(?:[.+#][a-z0-9+#]+)*[+#]*")


def strip_title(text: str, max_chars: int = 200) -> str:
    """Remove the leading job title (label leakage), i.e. everything before the
    first section keyword, if that keyword appears within the first max_chars."""
    m = SECTION.search(text)
    return text[m.start():] if m and m.start() <= max_chars else text


def clean(text: str) -> str:
    """Normalise + mask personal info. Lowercases at the end."""
    text = EMAIL.sub(" <email> ", text)
    text = URL.sub(" <url> ", text)
    text = PHONE.sub(" <phone> ", text)
    text = text.replace("•", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def prepare(text: str, remove_title: bool = False, do_clean: bool = True) -> list[str]:
    text = str(text)
    if remove_title:
        text = strip_title(text)
    if do_clean:
        text = clean(text)
    return tokenize(text)
