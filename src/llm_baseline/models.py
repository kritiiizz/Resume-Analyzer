from typing import List, Optional

from pydantic import BaseModel, Field


class Education(BaseModel):
    university_name: str = Field(
        description="Name of the university or college"
    )

    degree: str = Field(
        description="Degree obtained by the candidate"
    )

    gpa: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="Candidate GPA on a scale of 0 to 10"
    )


class Experience(BaseModel):
    company_name: Optional[str] = Field(
        default=None,
        description="Company where the candidate worked"
    )

    years: Optional[str] = Field(
        default=None,
        description="Duration of work experience"
    )

    project_name: Optional[str] = Field(
        default=None,
        description="Project name"
    )

    project_description: Optional[str] = Field(
        default=None,
        description="Brief description of the project"
    )

    tech_stack: Optional[List[str]] = Field(
        default=None,
        description="Technologies used in the project"
    )


class Resume(BaseModel):
    name: str = Field(
        description="Candidate full name"
    )

    email: str = Field(
        description="Candidate email address"
    )

    phone_number: str = Field(
        description="Candidate phone number"
    )

    education: Optional[List[Education]] = Field(
        default=None,
        description="Educational qualifications"
    )

    experience: Optional[List[Experience]] = Field(
        default=None,
        description="Professional experience"
    )

    skills: Optional[List[str]] = Field(
        default=None,
        description="Technical and professional skills"
    )


class JDMatchResult(BaseModel):
    match_score: float = Field(
        ge=0,
        le=100,
        description="Resume match percentage with the Job Description"
    )

    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills found in both the resume and job description"
    )

    missing_skills: List[str] = Field(
        default_factory=list,
        description="Skills required by the JD but missing in the resume"
    )

    strengths: List[str] = Field(
        default_factory=list,
        description="Strengths of the candidate"
    )

    gaps: List[str] = Field(
        default_factory=list,
        description="Weaknesses or gaps in the resume"
    )

    summary: str = Field(
        description="Overall recruiter summary"
    )