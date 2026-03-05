# tests/test_extraction.py
# RUN: pytest tests/test_extraction.py

import pytest

from src.agents.extractor import ExtractionRouter
from src.models.models import DocumentProfile, OriginType, StrategyType


@pytest.fixture
def router(tmp_path):
    profiles_dir = tmp_path / "profiles"
    ledger_path = tmp_path / "ledger.jsonl"
    return ExtractionRouter(
        input_dir="data",
        profiles_dir=str(profiles_dir),
        ledger_path=str(ledger_path),
    )


def make_profile(char_density, whitespace_ratio, layout_complexity="single_column"):
    """Helper to build a minimal DocumentProfile for testing."""
    return DocumentProfile(
        document_id="docX",
        origin_type=OriginType.digital,
        layout_complexity=layout_complexity,
        domain_hint="test",
        char_density=char_density,
        whitespace_ratio=whitespace_ratio,
        bbox_distribution={"x_max": [50]},
        triage_confidence=0.8,
        file_path="dummy.pdf",
    )


def test_select_strategy_fasttext(router):
    profile = make_profile(char_density=0.002, whitespace_ratio=0.2)
    assert router.select_strategy(profile) == StrategyType.fasttext


def test_select_strategy_layout_aware(router):
    profile = make_profile(
        char_density=0.001, whitespace_ratio=0.4, layout_complexity="multi_column"
    )
    assert router.select_strategy(profile) == StrategyType.layout_aware


def test_select_strategy_vision_augmented_low_density(router):
    profile = make_profile(char_density=0.0004, whitespace_ratio=0.2)
    assert router.select_strategy(profile) == StrategyType.vision_augmented


def test_select_strategy_vision_augmented_high_whitespace(router):
    profile = make_profile(char_density=0.002, whitespace_ratio=0.7)
    assert router.select_strategy(profile) == StrategyType.vision_augmented


def test_escalate_strategy(router):
    assert router.escalate_strategy(StrategyType.fasttext) == StrategyType.layout_aware
    assert (
        router.escalate_strategy(StrategyType.layout_aware)
        == StrategyType.vision_augmented
    )
    assert (
        router.escalate_strategy(StrategyType.vision_augmented)
        == StrategyType.vision_augmented
    )


@pytest.mark.skip(
    reason="VisionExtractor requires cv2 (OpenCV) which may not be installed"
)
def test_get_extractor_returns_correct_class(router):
    # These imports are inside extractor.py, so we can check class names
    from src.strategies.fasttext_extractor import FastTextExtractor
    from src.strategies.layout_extractor import LayoutExtractor
    from src.strategies.vision_extractor import VisionExtractor

    assert isinstance(
        router._get_extractor(StrategyType.fasttext, "dummy.pdf"), FastTextExtractor
    )
    assert isinstance(
        router._get_extractor(StrategyType.layout_aware, "dummy.pdf"), LayoutExtractor
    )
    assert isinstance(
        router._get_extractor(StrategyType.vision_augmented, "dummy.pdf"),
        VisionExtractor,
    )

    with pytest.raises(ValueError):
        router._get_extractor("unknown", "dummy.pdf")
