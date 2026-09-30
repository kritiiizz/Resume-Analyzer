import os
import sys
import tempfile
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

from models import Resume
from prompts import resume_prompt


# -----------------------------------------------------
# Create Groq LLM
# -----------------------------------------------------

def get_llm(
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0,
) -> ChatGroq:
    """
    Returns a configured ChatGroq instance.
    """

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )

    return ChatGroq(
        model=model_name,
        temperature=temperature,
    )


# -----------------------------------------------------
# Read Resume PDF
# -----------------------------------------------------

def load_resume_text_from_path(pdf_path: str) -> str:
    """
    Load a resume PDF and return all text.
    """

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text = "\n".join(
        document.page_content
        for document in documents
    )

    return text


# -----------------------------------------------------
# Parse Resume
# -----------------------------------------------------

def parse_resume(
    resume_text: str,
    llm: Optional[ChatGroq] = None,
) -> Resume:
    """
    Convert resume text into a Resume object.
    """

    if llm is None:
        llm = get_llm()

    structured_llm = llm.with_structured_output(
        Resume,
        method="function_calling",
    )

    chain = resume_prompt | structured_llm

    result = chain.invoke(
        {
            "resume_text": resume_text,
        }
    )

    return result


# -----------------------------------------------------
# Convenience Function
# -----------------------------------------------------

def parse_resume_from_path(
    pdf_path: str,
) -> Resume:
    """
    Parse resume directly from PDF path.
    """

    text = load_resume_text_from_path(pdf_path)

    return parse_resume(text)


# -----------------------------------------------------
# Streamlit Helper
# -----------------------------------------------------

def load_resume_text_from_bytes(
    file_bytes: bytes,
    suffix: str = ".pdf",
) -> str:
    """
    Converts uploaded PDF bytes into text.
    """

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    try:
        temp_file.write(file_bytes)
        temp_file.close()

        return load_resume_text_from_path(
            temp_file.name
        )

    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


# -----------------------------------------------------
# CLI Support
# -----------------------------------------------------

if __name__ == "__main__":

    load_dotenv()

    if len(sys.argv) < 2:
        print(
            "Usage: python parser.py resume.pdf"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]

    resume = parse_resume_from_path(pdf_path)

    print(resume.model_dump_json(indent=4))