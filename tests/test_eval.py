import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from pillscan_server.eval import (
    EvalRecord,
    EvalSample,
    Pricing,
    RunnerConfig,
    SampleScore,
    build_summary,
    estimate_cost,
    load_samples,
    percentiles,
    render_summary_markdown,
    run_evaluation,
    score_response,
    wilson_interval,
)
from pillscan_server.models import (
    DrugResolution,
    ImageQuality,
    ModelUsage,
    PillAnalysisResponse,
    PillVisualAnalysis,
    PipelineTimings,
    ResolutionSource,
    ResolutionStatus,
    SubjectType,
    VisibleIdentifiers,
    VisualEvidence,
)
from tests.test_models import product

PERMIT = "衛署藥製字第012345號"


def response(
    *,
    subject: SubjectType = SubjectType.PACKAGE,
    permit: str | None = PERMIT,
) -> PillAnalysisResponse:
    resolution = (
        DrugResolution(
            status=ResolutionStatus.CATALOG_EXACT,
            source=ResolutionSource.TFDA_NHI,
            product=product().model_copy(
                update={
                    "identifiers": product().identifiers.model_copy(
                        update={"tfda_permit_number": permit}
                    )
                }
            ),
            candidates=[],
            catalog_version="2026-07-17",
        )
        if permit
        else DrugResolution(
            status=ResolutionStatus.CATALOG_NO_MATCH,
            source=ResolutionSource.TFDA,
            product=None,
            candidates=[],
            catalog_version="2026-07-17",
        )
    )
    return PillAnalysisResponse(
        schema_version="1.2",
        analysis_id=uuid4(),
        request_id="eval-test",
        provider="openai",
        model="gpt-5.4-mini",
        timings=PipelineTimings(
            upload_read_ms=1,
            image_normalization_ms=2,
            rate_limit_wait_ms=3,
            concurrency_wait_ms=4,
            vision_analysis_ms=4000,
            catalog_resolution_ms=5,
            pipeline_total_ms=4015,
        ),
        usage=ModelUsage(
            input_tokens=2000,
            cached_input_tokens=200,
            output_tokens=500,
            reasoning_tokens=50,
            total_tokens=2500,
        ),
        analysis=PillVisualAnalysis(
            subject_type=subject,
            state="direct_identifiers_visible",
            image_quality=ImageQuality(
                sufficient_for_analysis=True,
                blur="none",
                glare="none",
                subject_fills_frame=True,
                text_readability="clear",
            ),
            visible_identifiers=VisibleIdentifiers(
                product_name="範例錠",
                strength="500 mg",
                permit_number=permit or "",
                manufacturer="範例藥廠",
                other_text=[],
                confidence="high",
            ),
            evidence=VisualEvidence(
                dosage_form="tablet",
                colors=[],
                shape="",
                score_marks=[],
                symbols_or_logos=[],
                imprints=[],
                package_text=["範例錠"],
                distinctive_features=[],
            ),
            candidate_hypotheses=[],
            uncertainty_reasons=[],
            next_actions=[],
        ),
        resolution=resolution,
        disclaimer="test",
    )


def sample(tmp_path: Path, *, must_abstain: bool = False) -> EvalSample:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    return EvalSample(
        id="package:sample" if not must_abstain else "pill:sample",
        image_path=image,
        expected_subject_type=SubjectType.PILL if must_abstain else SubjectType.PACKAGE,
        expected_permit_number=PERMIT,
        must_abstain=must_abstain,
        group="strong" if not must_abstain else "ambiguous",
        source_manifest=tmp_path / "manifest.json",
    )


def record(tmp_path: Path, parsed: PillAnalysisResponse) -> EvalRecord:
    eval_sample = sample(tmp_path)
    return EvalRecord(
        sample=eval_sample,
        completed_at=datetime.now(UTC),
        image_sha256="a" * 64,
        request_success=True,
        status_code=200,
        attempts=1,
        scheduler_wait_ms=10,
        client_latency_ms=4100,
        estimated_cost=estimate_cost(parsed, Pricing()),
        score=score_response(eval_sample, parsed),
        response=parsed,
        error=None,
    )


def test_loads_tfda_package_and_pill_manifests(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "same.jpg").write_bytes(b"image")
    package_manifest = tmp_path / "package.json"
    package_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "same",
                        "permit_number": PERMIT,
                        "primary_image": "images/same.jpg",
                        "visual_category": "package_photo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pill_manifest = tmp_path / "pill.json"
    pill_manifest.write_text(
        json.dumps(
            {
                "dataset_type": "pill_appearance",
                "records": [
                    {
                        "id": "same",
                        "permit_number": PERMIT,
                        "primary_image": "images/same.jpg",
                        "matchability": "ambiguous",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    samples = load_samples([package_manifest, pill_manifest])

    assert [item.id for item in samples] == ["package:same", "pill:same"]
    assert samples[0].group == "package_photo"
    assert samples[1].must_abstain is True


def test_loads_generic_manifest_and_rejects_missing_image(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    manifest = tmp_path / "eval.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "custom-1",
                        "image": "image.png",
                        "expected_subject_type": "package",
                        "expected_permit_number": PERMIT,
                        "group": "phone_photo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_samples([manifest])[0].image_path == image

    image.unlink()
    with pytest.raises(ValueError, match="missing eval image"):
        load_samples([manifest])


def test_rejects_empty_and_duplicate_manifests(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text('{"samples": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no eval samples"):
        load_samples([empty])

    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "same",
                        "image": "image.jpg",
                        "expected_subject_type": "package",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate eval sample id"):
        load_samples([duplicate, duplicate])


def test_must_abstain_only_accepts_pills(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="loose-pill"):
        EvalSample(
            id="invalid",
            image_path=tmp_path / "image.jpg",
            expected_subject_type=SubjectType.PACKAGE,
            must_abstain=True,
            group="invalid",
            source_manifest=tmp_path / "manifest.json",
        )


def test_runner_config_rejects_non_positive_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        RunnerConfig(
            endpoint="https://pillscan.test",
            output_dir=tmp_path,
            market="TW",
            context=None,
            api_token=None,
            concurrency=0,
            requests_per_minute=20,
            timeout_seconds=5,
            max_attempts=1,
            pricing=Pricing(),
        )


def test_scores_exact_match_and_unsafe_pill_identification(tmp_path: Path) -> None:
    exact_score = score_response(sample(tmp_path), response())
    unsafe_score = score_response(
        sample(tmp_path, must_abstain=True),
        response(subject=SubjectType.PILL),
    )

    assert exact_score.passed is True
    assert exact_score.expected_permit_present is True
    assert exact_score.exact_identification_correct is True
    assert unsafe_score.passed is False
    assert unsafe_score.safe_abstention is False
    assert unsafe_score.unsafe_exact_identification is True


def test_scores_negative_control_without_an_exact_id(tmp_path: Path) -> None:
    eval_sample = EvalSample(
        id="negative:empty-frame",
        image_path=tmp_path / "empty.jpg",
        expected_subject_type=SubjectType.UNKNOWN,
        expected_permit_number=None,
        must_abstain=False,
        group="negative_control",
        source_manifest=tmp_path / "negative.json",
    )

    score = score_response(eval_sample, response(subject=SubjectType.UNKNOWN, permit=None))

    assert score.subject_correct is True
    assert score.exact_identification_correct is None
    assert score.unsafe_exact_identification is False
    assert score.passed is True


def test_cost_percentiles_and_summary(tmp_path: Path) -> None:
    parsed = response()
    pricing = Pricing()
    eval_record = record(tmp_path, parsed)

    assert estimate_cost(parsed, pricing) == pytest.approx(0.003615)
    assert percentiles([1, 2, 3, 100]) == {
        "count": 4,
        "min": 1,
        "p50": 2.5,
        "p95": 85.45,
        "p99": 97.09,
        "max": 100,
    }
    assert percentiles([])["p95"] is None
    assert wilson_interval(0, 0) is None
    assert wilson_interval(95, 100) == {"low": 0.888248, "high": 0.978457}

    summary = build_summary([eval_record], pricing, processed_this_run=1, run_wall_seconds=4.1)
    markdown = render_summary_markdown(summary)
    assert summary["subject_accuracy"] == 1
    assert summary["exact_identification_accuracy"] == 1
    assert summary["latency_ms"]["vision_analysis_ms"]["p99"] == 4000
    assert summary["usage"]["total_tokens"] == 2500
    assert summary["last_run"]["throughput_per_minute"] == pytest.approx(14.634)
    assert summary["confidence_intervals_95"]["subject_accuracy"]["high"] == 1
    assert "Unsafe exact identifications: 0" in markdown


@pytest.mark.asyncio
async def test_runner_writes_jsonl_summary_and_resumes(tmp_path: Path) -> None:
    calls = 0
    parsed = response()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=parsed.model_dump(mode="json"))

    output_dir = tmp_path / "run"
    config = RunnerConfig(
        endpoint="https://pillscan.test",
        output_dir=output_dir,
        market="TW",
        context=None,
        api_token="test-token",
        concurrency=2,
        requests_per_minute=60_000,
        timeout_seconds=5,
        max_attempts=2,
        pricing=Pricing(),
    )
    eval_sample = sample(tmp_path)
    transport = httpx.MockTransport(handler)

    first = await run_evaluation([eval_sample], config, transport=transport)
    second = await run_evaluation([eval_sample], config, transport=transport)

    assert calls == 1
    assert first["sample_count"] == 1
    assert second["overall_pass_rate"] == 1
    assert len((output_dir / "records.jsonl").read_text().splitlines()) == 1
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.md").is_file()
    assert json.loads((output_dir / "run.json").read_text())["remaining_count"] == 0


@pytest.mark.asyncio
async def test_runner_records_non_retryable_failure(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(400, json={"message": "invalid image"})
    )
    config = RunnerConfig(
        endpoint="https://pillscan.test/v1/pills/analyze",
        output_dir=tmp_path / "failed-run",
        market="TW",
        context="test",
        api_token=None,
        concurrency=1,
        requests_per_minute=60_000,
        timeout_seconds=5,
        max_attempts=2,
        pricing=Pricing(),
    )

    summary = await run_evaluation([sample(tmp_path)], config, transport=transport)
    saved = EvalRecord.model_validate_json(
        (config.output_dir / "records.jsonl").read_text().strip()
    )

    assert summary["request_success_rate"] == 0
    assert saved.status_code == 400
    assert saved.attempts == 1
    assert saved.error == "invalid image"
    assert saved.score == SampleScore(
        subject_correct=False,
        expected_permit_present=None,
        exact_identification_correct=None,
        safe_abstention=None,
        unsafe_exact_identification=False,
        passed=False,
    )


@pytest.mark.asyncio
async def test_runner_retries_rate_limit_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            httpx.Response(429, json={"message": "slow down"}),
            httpx.Response(200, json=response().model_dump(mode="json")),
        ]
    )
    transport = httpx.MockTransport(lambda _: next(responses))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("pillscan_server.eval.asyncio.sleep", no_sleep)
    config = RunnerConfig(
        endpoint="https://pillscan.test",
        output_dir=tmp_path / "retry-run",
        market="TW",
        context=None,
        api_token=None,
        concurrency=1,
        requests_per_minute=60_000,
        timeout_seconds=5,
        max_attempts=2,
        pricing=Pricing(),
    )

    summary = await run_evaluation([sample(tmp_path)], config, transport=transport)
    saved = EvalRecord.model_validate_json(
        (config.output_dir / "records.jsonl").read_text().strip()
    )

    assert summary["retry_count"] == 1
    assert saved.request_success is True
    assert saved.attempts == 2
