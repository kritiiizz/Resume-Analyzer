from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------
# Resume Parsing Prompt
# -------------------------------------------------------

resume_template = """
You are an expert Resume Parser.

Your task is to extract information from the resume.

Rules:
1. Extract ONLY the information available in the resume.
2. Do NOT invent or assume any information.
3. If any field is missing, return null or an empty list.
4. Follow the output schema exactly.

Resume:

{resume_text}
"""

resume_prompt = PromptTemplate(
    template=resume_template,
    input_variables=["resume_text"],
)


# -------------------------------------------------------
# Job Description Matching Prompt
# -------------------------------------------------------

jd_match_template = """
You are an expert Technical Recruiter.

Compare the candidate's resume with the Job Description.

Rules:

1. Calculate a match score between 0 and 100.
2. Match skills ONLY if they appear in both Resume and JD.
3. Missing skills should only include skills required by the JD.
4. Give strengths based on the resume.
5. Give gaps based on the JD.
6. Write a short recruiter summary.
7. Never invent information.

Job Description:

{jd_text}

Candidate Resume (JSON):

{resume_json}
"""

jd_match_prompt = PromptTemplate(
    template=jd_match_template,
    input_variables=[
        "jd_text",
        "resume_json",
    ],
)