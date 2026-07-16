import base64
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from pillscan_server.errors import VisionProviderError
from pillscan_server.models import PillVisualAnalysis
from pillscan_server.protocols import PreparedImage

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
    ) -> PillVisualAnalysis:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": _request_text(market=market, context=context),
            }
        ]
        encoded = base64.b64encode(image.data).decode("ascii")
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": f"Normalized image dimensions: {image.width}x{image.height}.",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:{image.media_type};base64,{encoded}",
                    "detail": self._image_detail,
                },
            ]
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
        return parsed


def _request_text(*, market: str, context: str | None) -> str:
    context_text = context if context else "No additional context supplied."
    return (
        "Classify the image as a loose pill, medication package, or unknown, then extract only "
        "visible evidence for catalog lookup. "
        f"Expected market: {market}. Additional user context: {context_text}"
    )
