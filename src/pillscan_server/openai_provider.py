import base64
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from pillscan_server.errors import VisionProviderError
from pillscan_server.models import MedicationImageAnalysis, ModelUsage, PillVisualAnalysis
from pillscan_server.protocols import (
    MedicationVisionAnalysisResult,
    PreparedImage,
    VisionAnalysisResult,
)

SYSTEM_PROMPT = """You are a cautious visual evidence extractor for medication recognition.
Inspect the single supplied image and produce structured evidence that can later be verified
against an authoritative drug catalog.

Rules:
- First classify subject_type. Use pill for a loose tablet, capsule, or softgel. Use package for a
  medicine box, blister, bottle label, medication bag, or other labeled medicine container. Use
  unknown when neither class is supported by the image.
- For a package, transcribe only identifiers that are visibly present: product name, strength,
  Taiwan permit number, manufacturer, and other useful text. Do not infer missing text.
- For product_name, join the visible brand with adjacent variant, symptom, release-form, and dosage
  form words that distinguish the exact marketed product. For example, if "普拿疼" and
  "止痛加強錠" are both visible, return "普拿疼止痛加強錠", not only "普拿疼". Keep package
  counts and promotional text in other_text instead.
- Set state to direct_identifiers_visible only when a permit number is clear, or when both a clear
  product name and strength are visible. Otherwise use visual_evidence_only when the image is
  usable, needs_better_image when quality prevents reliable extraction, or no_visual_match when
  the subject is unknown.
- For a pill, leave package-only visible identifier fields empty and extract its visible imprint,
  dosage form, color, shape, score marks, symbols, and distinctive features.
- Never present a medication identity as confirmed from images alone.
- Read imprints character by character. Preserve spaces, scores, and uncertain alternatives.
- Candidate hypotheses are unverified search leads only. Include them only when imprint and
  appearance jointly support them; otherwise return an empty list.
- Do not invent a strength, manufacturer, or market. Use an empty string when unknown.
- If glare, blur, distance, or rotation prevents a reliable reading, set state to
  needs_better_image and give concrete retake instructions.
- Do not provide dosing, treatment, or other medical advice.
"""

MEDICATION_SYSTEM_PROMPT = """You are a cautious structured extractor for medication images.
Inspect the one supplied image and return every distinct medication item visibly supported by it.

Classification rules:
- Use pill for a loose tablet, capsule, or softgel.
- Use package for a medicine box, blister, bottle, or container representing one or more marketed
  medication products.
- Use medication_document for a prescription, medication list, medication bag, dispensing label,
  or similar document whose rows may name several medications.
- Use unknown when the image does not visibly support any medication class.
- document_type must be none for non-document subjects. For a medication document, choose the most
  specific document type, or unknown when its exact kind is unclear.

Extraction rules:
- Preserve item order from top to bottom, then left to right. Return one item per distinct
  medication row or product. Return an empty list when no medication item can be read reliably.
- Copy product name, generic name, strength, dosage form, Taiwan permit number, NHI drug code,
  manufacturer, and directions only when visibly present. Use empty strings or empty lists when
  absent; never infer missing identifiers or directions.
- product_name must combine the visible brand with adjacent variant, symptom, release-form, and
  dosage-form words that distinguish the marketed product. For example, when both "普拿疼" and
  "止痛加強錠" are visible, return "普拿疼止痛加強錠", not only "普拿疼". Keep package counts
  and promotional claims out of product_name.
- source_text must contain the exact visible text that supports that item. Put readable text that
  cannot be assigned safely to one item in unresolved_text.
- Directions are transcription only, not medical advice. Do not expand abbreviations unless the
  expanded wording is also visible.
- For Taiwanese medication tables, follow the printed column headers exactly. Map 數量 to
  directions.dose (the amount per administration), 天 to directions.duration, 總量 to
  directions.quantity (the total dispensed amount), 途徑 to directions.route, and 服用方法 to
  directions.frequency. Do not shift a value into an adjacent column.
- For a fixed daily schedule with plain numeric values, cross-check dose x days x administrations
  per day against total quantity. Use this only to re-read potentially confused characters such as
  二 versus 三; never invent a direction that is not visibly printed. If the visible text remains
  ambiguous, preserve it in source_text and lower confidence instead of guessing.
- For a loose pill, extract imprint, color, shape, score marks, symbols, and other visual evidence.
  Leave package/document-only fields empty unless they are visible in the same image.
- For packages and documents, place useful visible row or package text in evidence.package_text.
- Read identifiers character by character. If a character is uncertain, do not produce a permit
  number or NHI code; preserve the uncertain text in source_text or unresolved_text instead.
- Set confidence for the whole extracted item. Use high only when its identifying text or pill
  evidence is clear and internally consistent.
- If blur, glare, perspective, distance, or obstruction prevents reliable extraction, report it in
  image_quality and uncertainty_reasons and provide concrete retake guidance in next_actions.
- Never claim that visual extraction alone confirms medication identity.
- Do not provide diagnosis, treatment, dosing recommendations, or other medical advice.
"""


class OpenAIPillVisionAnalyzer:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        image_detail: str,
    ) -> None:
        self._client = client
        self._model = model
        self._image_detail = image_detail

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def analyze(
        self,
        image: PreparedImage,
        *,
        market: str,
        context: str | None,
    ) -> VisionAnalysisResult:
        content = _image_content(
            image,
            image_detail=self._image_detail,
            request_text=_request_text(market=market, context=context),
        )

        try:
            input_payload: Any = [{"role": "user", "content": content}]
            response = await self._client.responses.parse(
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input=input_payload,
                text_format=PillVisualAnalysis,
                max_output_tokens=2000,
                store=False,
            )
        except OpenAIError as exc:
            raise VisionProviderError from exc

        parsed = response.output_parsed
        if not isinstance(parsed, PillVisualAnalysis):
            raise VisionProviderError
        return VisionAnalysisResult(
            analysis=parsed,
            usage=_model_usage(response),
        )

    async def analyze_medications(
        self,
        image: PreparedImage,
        *,
        market: str,
        context: str | None,
    ) -> MedicationVisionAnalysisResult:
        content = _image_content(
            image,
            image_detail=self._image_detail,
            request_text=_medication_request_text(market=market, context=context),
        )
        try:
            input_payload: Any = [{"role": "user", "content": content}]
            response = await self._client.responses.parse(
                model=self._model,
                instructions=MEDICATION_SYSTEM_PROMPT,
                input=input_payload,
                text_format=MedicationImageAnalysis,
                max_output_tokens=4000,
                store=False,
            )
        except OpenAIError as exc:
            raise VisionProviderError from exc

        parsed = response.output_parsed
        if not isinstance(parsed, MedicationImageAnalysis):
            raise VisionProviderError
        return MedicationVisionAnalysisResult(
            analysis=parsed,
            usage=_model_usage(response),
        )


def _request_text(*, market: str, context: str | None) -> str:
    context_text = context if context else "No additional context supplied."
    return (
        "Classify the image as a loose pill, medication package, or unknown, then extract only "
        "visible evidence for catalog lookup. "
        f"Expected market: {market}. Additional user context: {context_text}"
    )


def _medication_request_text(*, market: str, context: str | None) -> str:
    context_text = context if context else "No additional context supplied."
    return (
        "Classify this single image and extract every visibly supported medication item in source "
        "order. A prescription or medication list can contain several items; a pill or package "
        "usually contains one. "
        f"Expected market: {market}. Additional user context: {context_text}"
    )


def _image_content(
    image: PreparedImage,
    *,
    image_detail: str,
    request_text: str,
) -> list[dict[str, Any]]:
    encoded = base64.b64encode(image.data).decode("ascii")
    return [
        {"type": "input_text", "text": request_text},
        {
            "type": "input_text",
            "text": f"Normalized image dimensions: {image.width}x{image.height}.",
        },
        {
            "type": "input_image",
            "image_url": f"data:{image.media_type};base64,{encoded}",
            "detail": image_detail,
        },
    ]


def _model_usage(response: Any) -> ModelUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return ModelUsage.empty()

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return ModelUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0)),
        cached_input_tokens=int(getattr(input_details, "cached_tokens", 0)),
        output_tokens=int(getattr(usage, "output_tokens", 0)),
        reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0)),
        total_tokens=int(getattr(usage, "total_tokens", 0)),
    )
