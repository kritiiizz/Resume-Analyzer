from typing import Optional
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from models import Resume, JDMatchResult
from prompts import jd_match_prompt
from parser import (
    get_llm,
    parse_resume_from_path,
)


# -----------------------------------------------------
# Match Resume with Job Description
# -----------------------------------------------------

def match_resume_to_jd(
    resume: Resume,
    jd_text: str,
    llm: Optional[ChatGroq] = None,
) -> JDMatchResult:
    """
    Compare Resume against a Job Description.
    """

    if llm is None:
        llm = get_llm()

    structured_llm = llm.with_structured_output(
        JDMatchResult,
        method="function_calling",
    )

    chain = jd_match_prompt | structured_llm

    result = chain.invoke(
        {
            "jd_text": jd_text,
            "resume_json": resume.model_dump_json(indent=2),
        }
    )

    return result


# -----------------------------------------------------
# CLI Support
# -----------------------------------------------------

if __name__ == "__main__":

    load_dotenv()

    if len(sys.argv) != 3:
        print("Usage:")
        print("python matcher.py resume.pdf jd.txt")
        sys.exit(1)

    resume_path = sys.argv[1]
    jd_path = sys.argv[2]

    # Parse Resume
    resume = parse_resume_from_path(resume_path)

    # Read Job Description
    with open(jd_path, "r", encoding="utf-8") as file:
        jd_text = file.read()

    # Match Resume
    result = match_resume_to_jd(
        resume,
        jd_text,
    )

    print("\n========== MATCH RESULT ==========\n")

    print(result.model_dump_json(indent=4))












