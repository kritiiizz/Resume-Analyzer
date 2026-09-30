import json

import streamlit as st
from dotenv import load_dotenv

from matcher import match_resume_to_jd
from parser import (
    load_resume_text_from_bytes,
    parse_resume,
)

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume Analyzer + JD Matcher")
st.write(
    "Upload a resume, extract structured information, "
    "and compare it with a Job Description."
)

# -----------------------------------------------------
# Session State
# -----------------------------------------------------

if "resume" not in st.session_state:
    st.session_state.resume = None

# -----------------------------------------------------
# Resume Upload
# -----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"],
)

if uploaded_file is not None:

    if st.button("Parse Resume"):

        with st.spinner("Parsing resume..."):

            resume_text = load_resume_text_from_bytes(
                uploaded_file.read()
            )

            resume = parse_resume(resume_text)

            st.session_state.resume = resume

        st.success("Resume parsed successfully!")

# -----------------------------------------------------
# Display Parsed Resume
# -----------------------------------------------------

if st.session_state.resume is not None:

    resume = st.session_state.resume

    st.subheader("Extracted Resume Information")

    st.json(
        resume.model_dump()
    )

    st.download_button(
        label="⬇ Download Resume JSON",
        data=resume.model_dump_json(indent=4),
        file_name="parsed_resume.json",
        mime="application/json",
    )

# -----------------------------------------------------
# Job Description
# -----------------------------------------------------

st.divider()

st.subheader("Job Description")

jd_text = st.text_area(
    "Paste Job Description",
    height=250,
)

# -----------------------------------------------------
# JD Match
# -----------------------------------------------------

if st.button("Match Resume with JD"):

    if st.session_state.resume is None:
        st.error("Please parse a resume first.")

    elif jd_text.strip() == "":
        st.error("Please paste a Job Description.")

    else:

        with st.spinner("Matching..."):

            result = match_resume_to_jd(
                st.session_state.resume,
                jd_text,
            )

        st.success("Matching Complete!")

        st.subheader("Match Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Match Score",
                f"{result.match_score:.1f}%"
            )

        with col2:
            st.metric(
                "Missing Skills",
                len(result.missing_skills)
            )

        st.subheader("Matched Skills")
        st.write(result.matched_skills)

        st.subheader("Missing Skills")
        st.write(result.missing_skills)

        st.subheader("Strengths")
        st.write(result.strengths)

        st.subheader("Gaps")
        st.write(result.gaps)

        st.subheader("Recruiter Summary")
        st.info(result.summary)

        st.download_button(
            label="⬇ Download Match Result",
            data=result.model_dump_json(indent=4),
            file_name="jd_match_result.json",
            mime="application/json",
        )