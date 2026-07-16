from dataclasses import dataclass
from typing import Protocol

from pillscan_server.models import DrugResolution, PillVisualAnalysis


@dataclass(frozen=True, slots=True)
class PreparedImage:
    media_type: str
    data: bytes
    width: int
    height: int


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
    ) -> PillVisualAnalysis: ...


class DrugCatalogResolver(Protocol):
    async def resolve(
        self,
        analysis: PillVisualAnalysis,
        *,
        market: str,
    ) -> DrugResolution: ...
