from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SubjectType(StrEnum):
    PILL = "pill"
    PACKAGE = "package"
    UNKNOWN = "unknown"


class MedicationSubjectType(StrEnum):
    PILL = "pill"
    PACKAGE = "package"
    MEDICATION_DOCUMENT = "medication_document"
    UNKNOWN = "unknown"


class MedicationDocumentType(StrEnum):
    NONE = "none"
    PRESCRIPTION = "prescription"
    MEDICATION_LIST = "medication_list"
    MEDICATION_BAG = "medication_bag"
    DISPENSING_LABEL = "dispensing_label"
    UNKNOWN = "unknown"


class ObservedImprint(StrictModel):
    text: str = Field(description="Exact visible imprint text; use an empty string when unreadable")
    alternatives: list[str] = Field(description="Plausible alternate readings, most likely first")
    confidence: Literal["low", "medium", "high"]


class ImageQuality(StrictModel):
    sufficient_for_analysis: bool
    blur: Literal["none", "mild", "severe"]
    glare: Literal["none", "mild", "severe"]
    subject_fills_frame: bool
    text_readability: Literal["none", "partial", "clear"]


class VisibleIdentifiers(StrictModel):
    product_name: str = Field(description="Exact visible product name; empty when not visible")
    strength: str = Field(description="Exact visible strength text; empty when not visible")
    permit_number: str = Field(
        description="Exact visible Taiwan drug permit number; empty when not visible"
    )
    manufacturer: str = Field(description="Exact visible manufacturer; empty when not visible")
    other_text: list[str] = Field(description="Other useful text transcribed from the package")
    confidence: Literal["low", "medium", "high"]


class VisualEvidence(StrictModel):
    dosage_form: Literal["tablet", "capsule", "softgel", "other", "unknown"]
    colors: list[str]
    shape: str
    score_marks: list[str]
    symbols_or_logos: list[str]
    imprints: list[ObservedImprint]
    package_text: list[str]
    distinctive_features: list[str]


class CandidateHypothesis(StrictModel):
    product_name: str
    strength: str
    manufacturer: str
    market: str
    matching_evidence: list[str]
    conflicting_evidence: list[str]
    confidence: Literal["low", "medium", "high"]


class PillVisualAnalysis(StrictModel):
    subject_type: SubjectType
    state: Literal[
        "needs_better_image",
        "direct_identifiers_visible",
        "visual_evidence_only",
        "no_visual_match",
    ]
    image_quality: ImageQuality
    visible_identifiers: VisibleIdentifiers
    evidence: VisualEvidence
    candidate_hypotheses: list[CandidateHypothesis]
    uncertainty_reasons: list[str]
    next_actions: list[str]


class MedicationDirections(StrictModel):
    dose: str = Field(description="Visible dose per administration; empty when absent")
    frequency: str = Field(description="Visible administration frequency; empty when absent")
    route: str = Field(description="Visible administration route; empty when absent")
    duration: str = Field(description="Visible treatment duration; empty when absent")
    quantity: str = Field(description="Visible dispensed quantity; empty when absent")
    instructions: list[str] = Field(description="Other visible directions, copied without advice")


class ExtractedMedication(StrictModel):
    product_name: str = Field(description="Exact visible brand or product name; empty when absent")
    generic_name: str = Field(description="Exact visible generic name; empty when absent")
    strength: str = Field(description="Exact visible strength; empty when absent")
    dosage_form: str = Field(description="Exact visible dosage form; empty when absent")
    permit_number: str = Field(description="Exact visible Taiwan permit number; empty when absent")
    nhi_code: str = Field(description="Exact visible Taiwan NHI drug code; empty when absent")
    manufacturer: str = Field(description="Exact visible manufacturer; empty when absent")
    directions: MedicationDirections
    source_text: list[str] = Field(description="Visible text supporting this medication row")
    confidence: Literal["low", "medium", "high"]
    evidence: VisualEvidence


class MedicationImageAnalysis(StrictModel):
    """Provider output for a single image that may contain several medications."""

    subject_type: MedicationSubjectType
    document_type: MedicationDocumentType
    image_quality: ImageQuality
    items: list[ExtractedMedication]
    unresolved_text: list[str]
    uncertainty_reasons: list[str]
    next_actions: list[str]

    @model_validator(mode="after")
    def validate_subject_shape(self) -> Self:
        if self.subject_type is MedicationSubjectType.MEDICATION_DOCUMENT:
            if self.document_type is MedicationDocumentType.NONE:
                raise ValueError("medication_document requires a document type")
        elif self.document_type is not MedicationDocumentType.NONE:
            raise ValueError("non-document subjects require document_type=none")
        if self.subject_type is MedicationSubjectType.UNKNOWN and self.items:
            raise ValueError("unknown subjects cannot contain medication items")
        return self


class MedicationAnalysisSummary(StrictModel):
    subject_type: MedicationSubjectType
    document_type: MedicationDocumentType
    image_quality: ImageQuality
    unresolved_text: list[str]
    uncertainty_reasons: list[str]
    next_actions: list[str]


class ResolutionStatus(StrEnum):
    NEEDS_BETTER_IMAGE = "needs_better_image"
    EVIDENCE_EXTRACTED = "evidence_extracted"
    NOT_MEDICATION_IMAGE = "not_medication_image"
    CATALOG_CANDIDATES = "catalog_candidates"
    CATALOG_EXACT = "catalog_exact"
    CATALOG_NO_MATCH = "catalog_no_match"


class ResolutionSource(StrEnum):
    NOT_QUERIED = "not_queried"
    TFDA = "tfda"
    TFDA_NHI = "tfda_nhi"


class ProductIdentifiers(StrictModel):
    tfda_permit_number: str | None = Field(
        description="Complete TFDA permit number, including its official prefix",
    )
    tfda_ingredient_codes: list[str]
    nhi_code: str | None = Field(description="Taiwan NHI medication code")
    gtins: list[str] = Field(description="Package-level GS1 GTINs")


class DrugIngredient(StrictModel):
    official_name: str = Field(description="Official ingredient name from the source catalog")
    normalized_generic_name: str | None = Field(
        description="Normalized generic name; exact salt form is preserved when relevant",
    )
    tfda_ingredient_code: str | None
    prescription_label: str | None
    amount_description: str | None
    amount: str | None
    unit: str | None


class DrugProduct(StrictModel):
    identifiers: ProductIdentifiers
    brand_name_zh: str | None
    brand_name_en: str | None
    generic_display_name: str | None
    ingredients: list[DrugIngredient]
    dosage_form: str | None
    manufacturer: str | None
    applicant: str | None
    indications: str | None
    source_urls: list[str]


class CatalogCandidate(StrictModel):
    product: DrugProduct
    score: float = Field(ge=0.0, le=1.0)
    matching_evidence: list[str]
    conflicting_evidence: list[str]


class DrugResolution(StrictModel):
    status: ResolutionStatus
    source: ResolutionSource
    product: DrugProduct | None
    candidates: list[CatalogCandidate]
    catalog_version: str | None

    @model_validator(mode="after")
    def validate_resolution_state(self) -> Self:
        if self.status is ResolutionStatus.CATALOG_EXACT:
            if self.source is ResolutionSource.NOT_QUERIED or self.product is None:
                raise ValueError("catalog_exact requires a queried source and product")
            if self.product.identifiers.tfda_permit_number is None:
                raise ValueError("catalog_exact requires a TFDA permit number")
            if self.candidates:
                raise ValueError("catalog_exact must not include candidates")
        elif self.status is ResolutionStatus.CATALOG_CANDIDATES:
            if self.source is ResolutionSource.NOT_QUERIED or not self.candidates:
                raise ValueError("catalog_candidates requires a queried source and candidates")
            if self.product is not None:
                raise ValueError("catalog_candidates must not include an exact product")
        elif self.status is ResolutionStatus.CATALOG_NO_MATCH:
            if self.source is ResolutionSource.NOT_QUERIED:
                raise ValueError("catalog_no_match requires a queried source")
            if self.product is not None or self.candidates:
                raise ValueError("catalog_no_match must not include products")
        elif (
            self.source is not ResolutionSource.NOT_QUERIED
            or self.product is not None
            or self.candidates
        ):
            raise ValueError("pre-catalog states cannot include catalog results")
        return self


IDENTIFICATION_DISCLAIMER = (
    "Visual candidates are not a confirmed medication identity. Verify against an "
    "authoritative drug catalog and a pharmacist before use."
)


class PipelineTimings(StrictModel):
    """Successful request stages in milliseconds, measured with a monotonic clock."""

    upload_read_ms: float = Field(ge=0)
    image_normalization_ms: float = Field(ge=0)
    rate_limit_wait_ms: float = Field(ge=0)
    concurrency_wait_ms: float = Field(ge=0)
    vision_analysis_ms: float = Field(ge=0)
    catalog_resolution_ms: float = Field(ge=0)
    pipeline_total_ms: float = Field(ge=0)


class ModelUsage(StrictModel):
    """Billable token usage reported by the vision provider."""

    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @classmethod
    def empty(cls) -> "ModelUsage":
        return cls(
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
        )


class PillAnalysisResponse(StrictModel):
    schema_version: Literal["1.2"]
    analysis_id: UUID
    request_id: str
    provider: str
    model: str
    timings: PipelineTimings
    usage: ModelUsage
    analysis: PillVisualAnalysis
    resolution: DrugResolution
    disclaimer: str


class MedicationResultItem(StrictModel):
    index: int = Field(ge=0)
    extracted: ExtractedMedication
    resolution: DrugResolution


class MedicationAnalysisResponse(StrictModel):
    schema_version: Literal["2.0"]
    analysis_id: UUID
    request_id: str
    provider: str
    model: str
    timings: PipelineTimings
    usage: ModelUsage
    analysis: MedicationAnalysisSummary
    items: list[MedicationResultItem]
    disclaimer: str


class HealthResponse(StrictModel):
    status: Literal["ok", "ready"]
    service: str
    version: str


class ErrorResponse(StrictModel):
    code: str
    message: str
    request_id: str
