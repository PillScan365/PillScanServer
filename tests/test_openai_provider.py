from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from pillscan_server.errors import VisionProviderError
from pillscan_server.models import (
    ImageQuality,
    PillVisualAnalysis,
    SubjectType,
    VisibleIdentifiers,
    VisualEvidence,
)
from pillscan_server.openai_provider import SYSTEM_PROMPT, OpenAIPillVisionAnalyzer
from pillscan_server.protocols import PreparedImage


def visual_analysis() -> PillVisualAnalysis:
    return PillVisualAnalysis(
        subject_type=SubjectType.PACKAGE,
        state="direct_identifiers_visible",
        image_quality=ImageQuality(
            sufficient_for_analysis=True,
            blur="none",
            glare="none",
            subject_fills_frame=True,
            text_readability="clear",
        ),
        visible_identifiers=VisibleIdentifiers(
            product_name="Example tablets",
            strength="500 mg",
            permit_number="",
            manufacturer="Example Pharma",
            other_text=[],
            confidence="high",
        ),
        evidence=VisualEvidence(
            dosage_form="tablet",
            colors=["white"],
            shape="round",
            score_marks=[],
            symbols_or_logos=[],
            imprints=[],
            package_text=[],
            distinctive_features=[],
        ),
        candidate_hypotheses=[],
        uncertainty_reasons=[],
        next_actions=["Verify against an authoritative catalog."],
    )


class FakeResponses:
    def __init__(self, parsed: PillVisualAnalysis | None, *, include_usage: bool = True) -> None:
        self.parsed = parsed
        self.include_usage = include_usage
        self.request: dict[str, Any] = {}

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.request = kwargs
        usage = SimpleNamespace(
            input_tokens=1234,
            input_tokens_details=SimpleNamespace(cached_tokens=34),
            output_tokens=456,
            output_tokens_details=SimpleNamespace(reasoning_tokens=56),
            total_tokens=1690,
        )
        return SimpleNamespace(
            output_parsed=self.parsed,
            **({"usage": usage} if self.include_usage else {}),
        )


class FakeClient:
    def __init__(self, parsed: PillVisualAnalysis | None, *, include_usage: bool = True) -> None:
        self.responses = FakeResponses(parsed, include_usage=include_usage)


@pytest.mark.asyncio
async def test_provider_builds_structured_single_image_request() -> None:
    fake_client = FakeClient(visual_analysis())
    analyzer = OpenAIPillVisionAnalyzer(
        cast(AsyncOpenAI, fake_client),
        model="test-vision",
        image_detail="high",
    )
    image = PreparedImage("image/jpeg", b"capture", 100, 100)

    result = await analyzer.analyze(image, market="TW", context=None)

    assert result.analysis.subject_type == SubjectType.PACKAGE
    assert result.analysis.state == "direct_identifiers_visible"
    assert result.usage.input_tokens == 1234
    assert result.usage.cached_input_tokens == 34
    assert result.usage.reasoning_tokens == 56
    assert analyzer.provider_name == "openai"
    assert analyzer.model_name == "test-vision"
    assert fake_client.responses.request["text_format"] is PillVisualAnalysis
    assert fake_client.responses.request["store"] is False
    input_content = fake_client.responses.request["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in input_content) == 1
    assert all(
        item.get("detail") == "high" for item in input_content if item["type"] == "input_image"
    )
    assert "Use pill for a loose tablet" in SYSTEM_PROMPT
    assert "Use package for a" in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_provider_rejects_missing_structured_output() -> None:
    fake_client = FakeClient(None)
    analyzer = OpenAIPillVisionAnalyzer(
        cast(AsyncOpenAI, fake_client),
        model="test-vision",
        image_detail="high",
    )

    with pytest.raises(VisionProviderError):
        await analyzer.analyze(
            PreparedImage("image/jpeg", b"capture", 100, 100),
            market="TW",
            context="package unavailable",
        )


@pytest.mark.asyncio
async def test_provider_defaults_missing_usage_to_zero() -> None:
    analyzer = OpenAIPillVisionAnalyzer(
        cast(AsyncOpenAI, FakeClient(visual_analysis(), include_usage=False)),
        model="test-vision",
        image_detail="high",
    )

    result = await analyzer.analyze(
        PreparedImage("image/jpeg", b"capture", 100, 100),
        market="TW",
        context=None,
    )

    assert result.usage.total_tokens == 0
