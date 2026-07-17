from dataclasses import dataclass
from typing import Protocol

from pillscan_server.models import (
    DrugResolution,
    ExtractedMedication,
    ImageQuality,
    MedicationImageAnalysis,
    MedicationSubjectType,
    ModelUsage,
    PillVisualAnalysis,
)


@dataclass(frozen=True, slots=True)
class PreparedImage:
    media_type: str
    data: bytes
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VisionAnalysisResult:
    analysis: PillVisualAnalysis
    usage: ModelUsage


@dataclass(frozen=True, slots=True)
class MedicationVisionAnalysisResult:
    analysis: MedicationImageAnalysis
    usage: ModelUsage


class PillVisionAnalyzer(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def analyze(
        self,
        image: PreparedImage,
        *,
        market: str,
        context: str | None,
    ) -> VisionAnalysisResult: ...

    async def analyze_medications(
        self,
        image: PreparedImage,
        *,
        market: str,
        context: str | None,
    ) -> MedicationVisionAnalysisResult: ...


class DrugCatalogResolver(Protocol):
    async def resolve(
        self,
        analysis: PillVisualAnalysis,
        *,
        market: str,
    ) -> DrugResolution: ...

    async def resolve_medication(
        self,
        item: ExtractedMedication,
        *,
        image_quality: ImageQuality,
        subject_type: MedicationSubjectType,
        market: str,
    ) -> DrugResolution: ...
