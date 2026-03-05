# tests/test_triage.py
# RUN: pytest tests/test_triage.py

import pytest

from src.agents.triage import TriageAgent
from src.models.models import DocumentProfile, LayoutComplexity, OriginType


@pytest.fixture
def agent():
    return TriageAgent()


def test_analyse_document_form_fillable_single_column(agent):
    metrics = {
        "char_density": 0.002,
        "whitespace_ratio": 0.2,
        "bbox_distribution": {"x_max": [50, 60]},
        "origin_type_hint": "digital",
        "domain_hint": "finance",
        "file_path": "sample.pdf",
        "image_area_ratio": 0.1,
        "has_font_metadata": True,
    }
    profile = agent.analyse_document("doc1", metrics)

    assert isinstance(profile, DocumentProfile)
    assert profile.origin_type == OriginType.form_fillable
    assert profile.layout_complexity == LayoutComplexity.single_column
    assert profile.triage_confidence == 0.9
    assert profile.domain_hint == "finance"


def test_analyse_document_scanned_multi_column(agent):
    metrics = {
        "char_density": 0.001,
        "whitespace_ratio": 0.4,
        "bbox_distribution": {"x_max": [50, 300]},
        "origin_type_hint": "scanned",
        "file_path": "scanned.pdf",
        "image_area_ratio": 0.2,
        "has_font_metadata": False,
    }
    profile = agent.analyse_document("doc2", metrics)

    assert profile.origin_type == OriginType.scanned
    assert profile.layout_complexity == LayoutComplexity.multi_column
    assert profile.triage_confidence == 0.75


def test_analyse_document_figure_heavy(agent):
    metrics = {
        "char_density": 0.002,
        "whitespace_ratio": 0.6,
        "bbox_distribution": {"x_max": [50]},
        "origin_type_hint": "digital",
        "file_path": "figure.pdf",
        "image_area_ratio": 0.1,
        "has_font_metadata": True,
    }
    profile = agent.analyse_document("doc3", metrics)

    assert profile.layout_complexity == LayoutComplexity.figure_heavy
    assert profile.triage_confidence == 0.6


def test_analyse_document_table_heavy(agent):
    metrics = {
        "char_density": 0.0004,
        "whitespace_ratio": 0.2,
        "bbox_distribution": {"x_max": [50]},
        "origin_type_hint": "digital",
        "file_path": "table.pdf",
        "image_area_ratio": 0.1,
        "has_font_metadata": True,
    }
    profile = agent.analyse_document("doc4", metrics)

    assert profile.layout_complexity == LayoutComplexity.table_heavy
    assert profile.triage_confidence == 0.6


def test_analyse_document_mixed(agent):
    metrics = {
        "char_density": 0.0004,
        "whitespace_ratio": 0.6,
        "bbox_distribution": {"x_max": [50]},
        "origin_type_hint": "digital",
        "file_path": "mixed.pdf",
        "image_area_ratio": 0.4,
        "has_font_metadata": True,
    }
    profile = agent.analyse_document("doc5", metrics)

    assert profile.origin_type == OriginType.mixed
    assert profile.layout_complexity == LayoutComplexity.mixed


def test_domain_classification_financial(agent):
    metrics = {
        "char_density": 0.001,
        "whitespace_ratio": 0.3,
        "bbox_distribution": {"x_max": [50]},
        "origin_type_hint": "digital",
        "file_path": "budget_report.pdf",
        "image_area_ratio": 0.1,
        "has_font_metadata": True,
    }
    profile = agent.analyse_document("doc6", metrics)

    assert profile.domain_hint == "financial"
