from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pydantic import SecretStr

from pillscan_server.config import Settings
from pillscan_server.models import (
    ImageQuality,
    ModelUsage,
    PillVisualAnalysis,
    SubjectType,
    VisibleIdentifiers,
    VisualEvidence,
)
from pillscan_server.protocols import PreparedImage, VisionAnalysisResult


class FakeAnalyzer:
    provider_name = "fake"
    model_name = "fake-vision"

    def __init__(self) -> None:
        self.received_image: PreparedImage | None = None

    async def analyze(
        self,
        image: PreparedImage,
        *,
        market: str,
        context: str | None,
    ) -> VisionAnalysisResult:
        self.received_image = image
        return VisionAnalysisResult(
            analysis=PillVisualAnalysis(
                subject_type=SubjectType.PILL,
                state="visual_evidence_only",
                image_quality=ImageQuality(
                    sufficient_for_analysis=True,
                    blur="none",
                    glare="none",
                    subject_fills_frame=True,
                    text_readability="clear",
                ),
                visible_identifiers=VisibleIdentifiers(
                    product_name="",
                    strength="",
                    permit_number="",
                    manufacturer="",
                    other_text=[],
                    confidence="low",
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
                uncertainty_reasons=["Authoritative catalog verification has not run."],
                next_actions=["Verify the visible evidence against the market catalog."],
            ),
            usage=ModelUsage.empty(),
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        openai_api_key=SecretStr("test-openai-key"),
        max_upload_bytes=1024 * 1024,
        log_level="critical",
        analyses_per_minute=1000,
        tfda_catalog_required=False,
        tfda_catalog_path=Path("/nonexistent/pillscan-test-catalog.sqlite3"),
    )


@pytest.fixture
def fake_analyzer() -> FakeAnalyzer:
    return FakeAnalyzer()


@pytest.fixture
def jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (128, 96), "white").save(output, format="JPEG")
    return output.getvalue()
