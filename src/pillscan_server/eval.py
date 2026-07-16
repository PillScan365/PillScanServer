import argparse
import asyncio
import hashlib
import json
import math
import mimetypes
import os
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Self

import httpx
from pydantic import Field, model_validator

from pillscan_server.models import (
    PillAnalysisResponse,
    ResolutionStatus,
    StrictModel,
    SubjectType,
)

DEFAULT_INPUT_PRICE = 0.75
DEFAULT_CACHED_INPUT_PRICE = 0.075
DEFAULT_OUTPUT_PRICE = 4.50
TIMING_FIELDS = (
    "upload_read_ms",
    "image_normalization_ms",
    "rate_limit_wait_ms",
    "concurrency_wait_ms",
    "vision_analysis_ms",
    "catalog_resolution_ms",
    "pipeline_total_ms",
)


class EvalSample(StrictModel):
    id: str
    image_path: Path
    expected_subject_type: SubjectType
    expected_permit_number: str | None = None
    must_abstain: bool = False
    group: str
    source_manifest: Path

    @model_validator(mode="after")
    def validate_expectation(self) -> Self:
        if self.must_abstain and self.expected_subject_type is not SubjectType.PILL:
            raise ValueError("must_abstain is only valid for loose-pill samples")
        return self


class Pricing(StrictModel):
    input_per_million: float = Field(default=DEFAULT_INPUT_PRICE, ge=0)
    cached_input_per_million: float = Field(default=DEFAULT_CACHED_INPUT_PRICE, ge=0)
    output_per_million: float = Field(default=DEFAULT_OUTPUT_PRICE, ge=0)
    currency: str = "USD"
    source: str = "https://developers.openai.com/api/docs/pricing"
    as_of: str = "2026-07-17"


class SampleScore(StrictModel):
    subject_correct: bool
    expected_permit_present: bool | None
    exact_identification_correct: bool | None
    safe_abstention: bool | None
    unsafe_exact_identification: bool
    passed: bool


class EvalRecord(StrictModel):
    sample: EvalSample
    completed_at: datetime
    image_sha256: str
    request_success: bool
    status_code: int | None
    attempts: int = Field(ge=1)
    scheduler_wait_ms: float = Field(ge=0)
    client_latency_ms: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    score: SampleScore
    response: PillAnalysisResponse | None
    error: str | None


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    endpoint: str
    output_dir: Path
    market: str
    context: str | None
    api_token: str | None
    concurrency: int
    requests_per_minute: float
    timeout_seconds: float
    max_attempts: int
    pricing: Pricing

    def __post_init__(self) -> None:
        positive = {
            "concurrency": self.concurrency,
            "requests_per_minute": self.requests_per_minute,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"runner values must be positive: {', '.join(invalid)}")


class PacedRateLimiter:
    """Space request starts evenly to avoid the burst behavior of token buckets."""

    def __init__(self, requests_per_minute: float) -> None:
        self._interval = 60.0 / requests_per_minute
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            scheduled = max(now, self._next_start)
            self._next_start = scheduled + self._interval
        await asyncio.sleep(max(0.0, scheduled - now))


def load_samples(manifest_paths: Sequence[Path]) -> list[EvalSample]:
    samples: list[EvalSample] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaded = (
            _load_generic_manifest(manifest_path, payload)
            if "samples" in payload
            else _load_tfda_manifest(manifest_path, payload)
        )
        for sample in loaded:
            if sample.id in seen:
                raise ValueError(f"duplicate eval sample id: {sample.id}")
            if not sample.image_path.is_file():
                raise ValueError(f"missing eval image: {sample.image_path}")
            seen.add(sample.id)
            samples.append(sample)
    if not samples:
        raise ValueError("no eval samples found")
    return samples


def _load_generic_manifest(path: Path, payload: dict[str, Any]) -> list[EvalSample]:
    result: list[EvalSample] = []
    for item in payload["samples"]:
        image_path = _resolve_image(path, item["image"])
        result.append(
            EvalSample(
                id=str(item["id"]),
                image_path=image_path,
                expected_subject_type=item["expected_subject_type"],
                expected_permit_number=item.get("expected_permit_number"),
                must_abstain=bool(item.get("must_abstain", False)),
                group=str(item.get("group", "ungrouped")),
                source_manifest=path.resolve(),
            )
        )
    return result


def _load_tfda_manifest(path: Path, payload: dict[str, Any]) -> list[EvalSample]:
    is_pill = payload.get("dataset_type") == "pill_appearance"
    subject = SubjectType.PILL if is_pill else SubjectType.PACKAGE
    result: list[EvalSample] = []
    for item in payload.get("records", []):
        image = item.get("primary_image")
        if not image:
            continue
        matchability = str(item.get("matchability", "unknown"))
        group = matchability if is_pill else str(item.get("visual_category", "package"))
        raw_id = str(item.get("id") or item["permit_number"])
        result.append(
            EvalSample(
                id=f"{subject.value}:{raw_id}",
                image_path=_resolve_image(path, image),
                expected_subject_type=subject,
                expected_permit_number=item.get("permit_number"),
                must_abstain=is_pill and matchability != "strong",
                group=group,
                source_manifest=path.resolve(),
            )
        )
    return result


def _resolve_image(manifest_path: Path, value: str) -> Path:
    image_path = Path(value)
    return (
        image_path.resolve()
        if image_path.is_absolute()
        else (manifest_path.parent / image_path).resolve()
    )


async def run_evaluation(
    samples: Sequence[EvalSample],
    config: RunnerConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = config.output_dir / "records.jsonl"
    completed = _completed_ids(records_path)
    remaining = [sample for sample in samples if sample.id not in completed]
    _write_run_config(config, samples, remaining)
    run_started_at = perf_counter()

    limiter = PacedRateLimiter(config.requests_per_minute)
    semaphore = asyncio.Semaphore(config.concurrency)
    timeout = httpx.Timeout(config.timeout_seconds)
    headers = {"Authorization": f"Bearer {config.api_token}"} if config.api_token else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, transport=transport) as client:
        queue: asyncio.Queue[EvalSample] = asyncio.Queue()
        for sample in remaining:
            queue.put_nowait(sample)
        output_lock = asyncio.Lock()
        completed_count = 0

        async def worker() -> None:
            nonlocal completed_count
            while True:
                try:
                    sample = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await _run_sample(client, sample, config, limiter, semaphore)
                    async with output_lock:
                        _append_jsonl(records_path, record.model_dump(mode="json"))
                        completed_count += 1
                        if (
                            completed_count == 1
                            or completed_count % 10 == 0
                            or completed_count == len(remaining)
                        ):
                            print(
                                f"completed {completed_count}/{len(remaining)} "
                                f"sample={record.sample.id} passed={record.score.passed}"
                            )
                finally:
                    queue.task_done()

        async with asyncio.TaskGroup() as task_group:
            for _ in range(min(config.concurrency, len(remaining))):
                task_group.create_task(worker())

    records = _read_records(records_path)
    run_wall_seconds = perf_counter() - run_started_at
    summary = build_summary(
        records,
        config.pricing,
        processed_this_run=len(remaining),
        run_wall_seconds=run_wall_seconds,
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


async def _run_sample(
    client: httpx.AsyncClient,
    sample: EvalSample,
    config: RunnerConfig,
    limiter: PacedRateLimiter,
    semaphore: asyncio.Semaphore,
) -> EvalRecord:
    image_bytes = await asyncio.to_thread(sample.image_path.read_bytes)
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    media_type = mimetypes.guess_type(sample.image_path.name)[0] or "application/octet-stream"
    endpoint = _analysis_endpoint(config.endpoint)
    job_started = perf_counter()
    request_started: float | None = None
    response: httpx.Response | None = None
    error: str | None = None

    for attempt in range(1, config.max_attempts + 1):
        await limiter.wait()
        try:
            async with semaphore:
                if request_started is None:
                    request_started = perf_counter()
                response = await client.post(
                    endpoint,
                    files={"image": (sample.image_path.name, image_bytes, media_type)},
                    data={
                        "market": config.market,
                        **({"context": config.context} if config.context else {}),
                    },
                    headers={
                        "X-Request-ID": (
                            f"eval-{hashlib.sha256(sample.id.encode()).hexdigest()[:16]}"
                        )
                    },
                )
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < config.max_attempts:
                await asyncio.sleep(2 ** (attempt - 1))
                continue
            return _failed_record(
                sample,
                image_hash,
                attempt,
                job_started,
                request_started,
                None,
                error,
            )

        if response.status_code == 429 or response.status_code >= 500:
            error = f"HTTP {response.status_code}: {_response_error(response)}"
            if attempt < config.max_attempts:
                await asyncio.sleep(2 ** (attempt - 1))
                continue
        break

    if response is None:
        return _failed_record(
            sample,
            image_hash,
            config.max_attempts,
            job_started,
            request_started,
            None,
            error or "request completed without a response",
        )
    now = perf_counter()
    scheduler_wait_ms = round(((request_started or now) - job_started) * 1000, 2)
    latency_ms = round((now - (request_started or job_started)) * 1000, 2)
    if not response.is_success:
        return _failed_record(
            sample,
            image_hash,
            attempt,
            job_started,
            request_started,
            response.status_code,
            error or _response_error(response),
        )
    try:
        parsed = PillAnalysisResponse.model_validate(response.json())
    except (ValueError, json.JSONDecodeError) as exc:
        return _failed_record(
            sample,
            image_hash,
            attempt,
            job_started,
            request_started,
            response.status_code,
            f"invalid response contract: {exc}",
        )

    score = score_response(sample, parsed)
    return EvalRecord(
        sample=sample,
        completed_at=datetime.now(UTC),
        image_sha256=image_hash,
        request_success=True,
        status_code=response.status_code,
        attempts=attempt,
        scheduler_wait_ms=scheduler_wait_ms,
        client_latency_ms=latency_ms,
        estimated_cost=estimate_cost(parsed, config.pricing),
        score=score,
        response=parsed,
        error=None,
    )


def _failed_record(
    sample: EvalSample,
    image_hash: str,
    attempts: int,
    job_started: float,
    request_started: float | None,
    status_code: int | None,
    error: str,
) -> EvalRecord:
    now = perf_counter()
    return EvalRecord(
        sample=sample,
        completed_at=datetime.now(UTC),
        image_sha256=image_hash,
        request_success=False,
        status_code=status_code,
        attempts=attempts,
        scheduler_wait_ms=round(((request_started or now) - job_started) * 1000, 2),
        client_latency_ms=round((now - (request_started or job_started)) * 1000, 2),
        estimated_cost=0,
        score=SampleScore(
            subject_correct=False,
            expected_permit_present=None,
            exact_identification_correct=None,
            safe_abstention=None,
            unsafe_exact_identification=False,
            passed=False,
        ),
        response=None,
        error=error,
    )


def score_response(sample: EvalSample, response: PillAnalysisResponse) -> SampleScore:
    subject_correct = response.analysis.subject_type is sample.expected_subject_type
    resolution = response.resolution
    exact_permit = (
        resolution.product.identifiers.tfda_permit_number
        if resolution.status is ResolutionStatus.CATALOG_EXACT and resolution.product
        else None
    )
    candidate_permits = {
        candidate.product.identifiers.tfda_permit_number for candidate in resolution.candidates
    }
    expected = sample.expected_permit_number
    expected_present = expected in ({exact_permit} | candidate_permits) if expected else None
    exact_correct = exact_permit == expected if expected and not sample.must_abstain else None
    unsafe_exact = exact_permit is not None and (sample.must_abstain or exact_permit != expected)
    safe_abstention = exact_permit is None if sample.must_abstain else None
    if sample.must_abstain:
        identification_ok = safe_abstention
    elif expected:
        identification_ok = exact_correct
    else:
        identification_ok = exact_permit is None
    return SampleScore(
        subject_correct=subject_correct,
        expected_permit_present=expected_present,
        exact_identification_correct=exact_correct,
        safe_abstention=safe_abstention,
        unsafe_exact_identification=unsafe_exact,
        passed=subject_correct and identification_ok is True and not unsafe_exact,
    )


def estimate_cost(response: PillAnalysisResponse, pricing: Pricing) -> float:
    usage = response.usage
    uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
    cost = (
        uncached * pricing.input_per_million
        + usage.cached_input_tokens * pricing.cached_input_per_million
        + usage.output_tokens * pricing.output_per_million
    ) / 1_000_000
    return round(cost, 8)


def build_summary(
    records: Sequence[EvalRecord],
    pricing: Pricing,
    *,
    processed_this_run: int | None = None,
    run_wall_seconds: float | None = None,
) -> dict[str, Any]:
    total = len(records)
    successful = [record for record in records if record.request_success]
    exact_expected = [
        record
        for record in records
        if not record.sample.must_abstain and record.sample.expected_permit_number is not None
    ]
    abstention_expected = [record for record in records if record.sample.must_abstain]
    predicted_subjects = [
        record.response.analysis.subject_type.value if record.response else "request_error"
        for record in records
    ]
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record, predicted in zip(records, predicted_subjects, strict=True):
        confusion[record.sample.expected_subject_type.value][predicted] += 1

    timing_values: dict[str, list[float]] = {
        "scheduler_wait_ms": [],
        "client_latency_ms": [],
    }
    timing_values.update({field: [] for field in TIMING_FIELDS})
    for record in successful:
        timing_values["scheduler_wait_ms"].append(record.scheduler_wait_ms)
        timing_values["client_latency_ms"].append(record.client_latency_ms)
        if record.response is None:
            continue
        for field in TIMING_FIELDS:
            timing_values[field].append(float(getattr(record.response.timings, field)))

    token_totals = {
        "input_tokens": sum(
            record.response.usage.input_tokens for record in successful if record.response
        ),
        "cached_input_tokens": sum(
            record.response.usage.cached_input_tokens for record in successful if record.response
        ),
        "output_tokens": sum(
            record.response.usage.output_tokens for record in successful if record.response
        ),
        "reasoning_tokens": sum(
            record.response.usage.reasoning_tokens for record in successful if record.response
        ),
        "total_tokens": sum(
            record.response.usage.total_tokens for record in successful if record.response
        ),
    }
    status_counts = Counter(
        record.response.resolution.status.value if record.response else "request_error"
        for record in records
    )
    groups: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        groups[record.sample.group].append(record)

    request_successes = len(successful)
    passes = sum(record.score.passed for record in records)
    subject_correct = sum(record.score.subject_correct for record in records)
    exact_correct = sum(
        record.score.exact_identification_correct is True for record in exact_expected
    )
    abstention_correct = sum(record.score.safe_abstention is True for record in abstention_expected)
    expected_permit_count = sum(
        record.sample.expected_permit_number is not None for record in records
    )
    expected_permit_present = sum(
        record.score.expected_permit_present is True for record in records
    )
    quality_counts = {
        "request_success_rate": (request_successes, total),
        "overall_pass_rate": (passes, total),
        "subject_accuracy": (subject_correct, total),
        "exact_identification_accuracy": (exact_correct, len(exact_expected)),
        "safe_abstention_rate": (abstention_correct, len(abstention_expected)),
        "candidate_recall": (expected_permit_present, expected_permit_count),
    }
    throughput = (
        processed_this_run / run_wall_seconds * 60
        if processed_this_run is not None and run_wall_seconds and run_wall_seconds > 0
        else None
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_count": total,
        "request_success_count": request_successes,
        "request_success_rate": _rate(*quality_counts["request_success_rate"]),
        "overall_pass_count": passes,
        "overall_pass_rate": _rate(*quality_counts["overall_pass_rate"]),
        "subject_accuracy": _rate(*quality_counts["subject_accuracy"]),
        "exact_identification_accuracy": _rate(*quality_counts["exact_identification_accuracy"]),
        "safe_abstention_rate": _rate(*quality_counts["safe_abstention_rate"]),
        "candidate_recall": _rate(*quality_counts["candidate_recall"]),
        "confidence_intervals_95": {
            name: wilson_interval(*counts) for name, counts in quality_counts.items()
        },
        "unsafe_exact_identification_count": sum(
            record.score.unsafe_exact_identification for record in records
        ),
        "retry_count": sum(record.attempts - 1 for record in records),
        "resolution_status_counts": dict(sorted(status_counts.items())),
        "subject_confusion_matrix": {
            expected: dict(sorted(counts.items())) for expected, counts in sorted(confusion.items())
        },
        "latency_ms": {name: percentiles(values) for name, values in timing_values.items()},
        "last_run": {
            "processed_count": processed_this_run,
            "wall_seconds": round(run_wall_seconds, 3) if run_wall_seconds is not None else None,
            "throughput_per_minute": round(throughput, 3) if throughput is not None else None,
        },
        "usage": token_totals,
        "pricing": pricing.model_dump(mode="json"),
        "estimated_cost": {
            "currency": pricing.currency,
            "total": round(sum(record.estimated_cost for record in records), 6),
            "per_successful_request": round(
                sum(record.estimated_cost for record in records) / len(successful), 8
            )
            if successful
            else 0,
        },
        "groups": {
            group: {
                "count": len(items),
                "pass_rate": _rate(sum(item.score.passed for item in items), len(items)),
                "subject_accuracy": _rate(
                    sum(item.score.subject_correct for item in items), len(items)
                ),
                "unsafe_exact_identification_count": sum(
                    item.score.unsafe_exact_identification for item in items
                ),
            }
            for group, items in sorted(groups.items())
        },
    }


def percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 2),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": round(ordered[-1], 2),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float] | None:
    if total == 0:
        return None
    observed = successes / total
    denominator = 1 + z**2 / total
    center = (observed + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(observed * (1 - observed) / total + z**2 / (4 * total**2)) / denominator
    return {"low": round(max(0, center - margin), 6), "high": round(min(1, center + margin), 6)}


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def render_summary_markdown(summary: dict[str, Any]) -> str:
    client = summary["latency_ms"]["client_latency_ms"]
    pipeline = summary["latency_ms"]["pipeline_total_ms"]
    vision = summary["latency_ms"]["vision_analysis_ms"]
    cost = summary["estimated_cost"]
    lines = [
        "# PillScan evaluation summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Quality",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Request success: {_pct(summary['request_success_rate'])}",
        f"- Subject accuracy: {_pct(summary['subject_accuracy'])}",
        f"- Exact identification accuracy: {_pct(summary['exact_identification_accuracy'])}",
        f"- Safe abstention rate: {_pct(summary['safe_abstention_rate'])}",
        f"- Candidate recall: {_pct(summary['candidate_recall'])}",
        f"- Unsafe exact identifications: {summary['unsafe_exact_identification_count']}",
        "",
        "## Latency",
        "",
        "| Metric | p50 | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        _latency_row("Client end-to-end", client),
        _latency_row("Pipeline total", pipeline),
        _latency_row("Vision analysis", vision),
        "",
        f"- Last-run throughput: {summary['last_run']['throughput_per_minute']} requests/minute",
        "",
        "## Usage and estimated cost",
        "",
        f"- Total tokens: {summary['usage']['total_tokens']:,}",
        f"- Output tokens: {summary['usage']['output_tokens']:,}",
        f"- Estimated total: {cost['currency']} {cost['total']:.6f}",
        (
            f"- Estimated per successful request: {cost['currency']} "
            f"{cost['per_successful_request']:.8f}"
        ),
        "",
        "The cost excludes provider-side usage from failed attempts that return no usage object.",
        "",
    ]
    return "\n".join(lines)


def _latency_row(label: str, values: dict[str, Any]) -> str:
    return (
        f"| {label} | {_ms(values['p50'])} | {_ms(values['p95'])} | "
        f"{_ms(values['p99'])} | {_ms(values['max'])} |"
    )


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} ms"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _analysis_endpoint(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    return (
        normalized if normalized.endswith("/v1/pills/analyze") else f"{normalized}/v1/pills/analyze"
    )


def _response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("message") or payload.get("detail") or payload)
    except (ValueError, json.JSONDecodeError):
        return response.text[:500]


def _completed_ids(path: Path) -> set[str]:
    return {record.sample.id for record in _read_records(path)} if path.is_file() else set()


def _read_records(path: Path) -> list[EvalRecord]:
    if not path.is_file():
        return []
    records: list[EvalRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(EvalRecord.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}: {exc}") from exc
    return records


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()


def _write_run_config(
    config: RunnerConfig,
    samples: Sequence[EvalSample],
    remaining: Sequence[EvalSample],
) -> None:
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": config.endpoint,
        "market": config.market,
        "context": config.context,
        "concurrency": config.concurrency,
        "requests_per_minute": config.requests_per_minute,
        "timeout_seconds": config.timeout_seconds,
        "max_attempts": config.max_attempts,
        "pricing": config.pricing.model_dump(mode="json"),
        "sample_count": len(samples),
        "remaining_count": len(remaining),
        "manifests": sorted({str(sample.source_manifest) for sample in samples}),
    }
    (config.output_dir / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _select_samples(samples: list[EvalSample], limit: int | None, seed: int) -> list[EvalSample]:
    selected = list(samples)
    random.Random(seed).shuffle(selected)  # noqa: S311 - deterministic eval sampling
    return selected[:limit] if limit is not None else selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible PillScan HTTP evaluation")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--market", default="TW")
    parser.add_argument("--context")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests-per-minute", type=float, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--api-token-env", default="PILLSCAN_API_TOKEN")
    parser.add_argument("--input-price", type=float, default=DEFAULT_INPUT_PRICE)
    parser.add_argument("--cached-input-price", type=float, default=DEFAULT_CACHED_INPUT_PRICE)
    parser.add_argument("--output-price", type=float, default=DEFAULT_OUTPUT_PRICE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    for name in ("limit", "concurrency", "requests_per_minute", "timeout_seconds", "max_attempts"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero")

    samples = _select_samples(load_samples(args.manifests), args.limit, args.seed)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("eval-results") / timestamp
    pricing = Pricing(
        input_per_million=args.input_price,
        cached_input_per_million=args.cached_input_price,
        output_per_million=args.output_price,
    )
    config = RunnerConfig(
        endpoint=args.endpoint,
        output_dir=output_dir,
        market=args.market,
        context=args.context,
        api_token=os.getenv(args.api_token_env),
        concurrency=args.concurrency,
        requests_per_minute=args.requests_per_minute,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        pricing=pricing,
    )
    summary = asyncio.run(run_evaluation(samples, config))
    print(f"summary: {output_dir / 'summary.md'}")
    print(
        f"pass_rate={_pct(summary['overall_pass_rate'])} "
        f"p95={_ms(summary['latency_ms']['client_latency_ms']['p95'])} "
        f"cost={pricing.currency} {summary['estimated_cost']['total']:.6f}"
    )


if __name__ == "__main__":
    main()
