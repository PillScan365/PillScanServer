# Evaluation guide

The evaluation runner exercises the complete HTTP path: upload validation, normalization, OpenAI
vision, Pydantic Structured Output, TFDA/NHIA resolution, and response serialization. It is not a
mock-model accuracy test.

## Recommended sample sizes

Use three gates rather than spending the full budget after every code change:

| Gate | Samples | Purpose |
| --- | ---: | --- |
| Smoke | 50 | Validate the runner, schema, credentials, labels, and obvious regressions |
| Development | 200–300 | Tune prompts and catalog thresholds; compare candidate approaches |
| Frozen benchmark | 1,000 | Report final accuracy, safety, latency, and cost |

The current curated TFDA data contains 100 package images and 100 loose-pill images, which is a
useful development set. It is too clean and too small to support a production accuracy claim.

For the frozen 1,000-image benchmark, target:

- 500 packages: 200 official or clean labels, 200 real phone photos, and 100 hard cases with glare,
  perspective, rotation, blur, obstruction, or small text.
- 400 loose pills: 200 strong and distinctive imprints, 100 ambiguous or colliding appearances,
  and 100 unreadable or otherwise unsafe-to-identify cases.
- 100 negative controls: non-medication objects, empty frames, irrelevant packaging, and severely
  unusable images.

Keep at least 20% as a hidden holdout. Split by TFDA permit number, not by image, so multiple photos
of the same marketed product cannot appear in both development and holdout sets. Freeze the model
snapshot, prompt, catalog version, pricing snapshot, seed, and manifest before a reportable run.

## Start the server

The eval client defaults to 20 evenly spaced request starts per minute and concurrency 2, matching
the server defaults without causing a token-bucket burst.

```bash
uv run pillscan-bootstrap
```

In another terminal:

```bash
uv run pillscan-eval \
  /path/to/package_sample/manifest.json \
  /path/to/pill_sample/manifest.json \
  --output-dir eval-results/dev-200 \
  --requests-per-minute 20 \
  --concurrency 2
```

If production authentication is enabled, keep the bearer token in `PILLSCAN_API_TOKEN`. The runner
reads that environment variable by name and never accepts or records the token value in its output.
Use `--api-token-env NAME` to select a different environment variable.

Useful options:

```text
--limit 50                  deterministic subset for a smoke run
--seed 20260717             reproducible sample order
--max-attempts 3            retry network errors, HTTP 429, and HTTP 5xx
--timeout-seconds 120       per-attempt HTTP timeout
--endpoint URL              remote or local server base URL
--input-price 0.75          USD per one million uncached input tokens
--cached-input-price 0.075  USD per one million cached input tokens
--output-price 4.50         USD per one million output tokens
```

Re-run the same command and output directory to resume. Completed sample IDs in `records.jsonl` are
skipped. Use a new output directory when the model, prompt, catalog, pricing, or evaluation labels
change.

## Manifest formats

The runner directly understands the curated TFDA package and pill manifests. A generic manifest is
also supported:

```json
{
  "schema_version": "eval.v1",
  "samples": [
    {
      "id": "phone-package-0001",
      "image": "images/phone-package-0001.jpg",
      "expected_subject_type": "package",
      "expected_permit_number": "衛部藥製字第012345號",
      "must_abstain": false,
      "group": "package_phone_photo"
    },
    {
      "id": "ambiguous-pill-0001",
      "image": "images/ambiguous-pill-0001.jpg",
      "expected_subject_type": "pill",
      "expected_permit_number": "衛部藥製字第054321號",
      "must_abstain": true,
      "group": "pill_ambiguous"
    }
  ]
}
```

`must_abstain` means an exact identity would be unsafe for that image. The expected permit number is
still useful for measuring whether the correct product appears among non-authoritative candidates.

## Output artifacts

Each run directory contains:

- `records.jsonl`: append-only per-image input hash, expectation, raw API response, score, retries,
  scheduler wait, client latency, pipeline timings, provider usage, and estimated cost.
- `run.json`: non-secret execution settings, manifests, sample count, and pricing snapshot.
- `summary.json`: machine-readable aggregate quality, 95% Wilson intervals, confusion matrix,
  status distribution, subgroup results, token totals, cost, throughput, and latency percentiles.
- `summary.md`: compact human-readable report.

The main quality metrics are deliberately separate:

- **Subject accuracy:** pill, package, or unknown classification.
- **Exact identification accuracy:** correct TFDA permit number when an exact result is expected.
- **Safe abstention rate:** no exact result when the image is marked unsafe to identify.
- **Candidate recall:** expected permit appears either as the exact result or among candidates.
- **Unsafe exact identification:** any exact result on a must-abstain sample, or an exact permit that
  differs from ground truth. Treat this as the highest-severity failure.

Latency includes separate scheduler wait, client end-to-end, upload, normalization, rate-limit wait,
concurrency wait, vision analysis, catalog resolution, and pipeline total distributions. The report
uses linear-interpolated p50, p95, and p99 values; do not use a mean as the latency target.

Cost uses the successful response's provider-reported tokens. It excludes usage that an upstream
provider may have charged for failed attempts without returning a usage object. Update the pricing
flags whenever OpenAI prices or the processing region change.
