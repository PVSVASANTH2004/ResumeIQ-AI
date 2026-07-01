import json
from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, Session, relationship
from sqlalchemy.sql import func

from core.config import DATABASE_URL


engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class ResumeRecord(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    raw_text = Column(Text)
    parsed_json = Column(Text)

    analyses = relationship("AnalysisRecord", back_populates="resume")


class JobDescriptionRecord(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    company = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    raw_text = Column(Text)
    parsed_json = Column(Text)

    analyses = relationship("AnalysisRecord", back_populates="job_description")


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    jd_id = Column(Integer, ForeignKey("job_descriptions.id"))
    final_score = Column(Float)
    recommendation = Column(String)
    score_json = Column(Text)
    full_result_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    resume = relationship("ResumeRecord", back_populates="analyses")
    job_description = relationship("JobDescriptionRecord", back_populates="analyses")


class BulkSession(Base):
    __tablename__ = "bulk_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    jd_id = Column(Integer, ForeignKey("job_descriptions.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    candidates = relationship("BulkCandidateRecord", back_populates="session")


class BulkCandidateRecord(Base):
    __tablename__ = "bulk_candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("bulk_sessions.id"))
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    rank = Column(Integer)
    final_score = Column(Float)
    recommendation = Column(String)

    session = relationship("BulkSession", back_populates="candidates")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
