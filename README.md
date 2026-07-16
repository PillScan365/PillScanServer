# PillScan Server

An async FastAPI service that accepts one photo of either a loose pill or medication package,
normalizes it, automatically classifies the subject with an OpenAI vision model, and returns
schema-validated visual evidence resolved against a local TFDA/NHIA medication catalog.

The vision model never creates official identifiers. The server uses the full TFDA permit number
as the canonical Taiwan product ID, joins official ingredients and NHIA codes, and abstains or
returns ranked candidates when the catalog evidence is not unique.

## Runtime choices

- **Raspberry Pi:** run the multi-architecture container. OpenAI performs VLM inference remotely.
- **Linux 5090 workstation:** run directly with Pixi; no system Python or Docker installation is
  required.
- **Development:** use uv and the committed `uv.lock`.

## Configuration

The server reads `.env` followed by `.env.local`; real env vars take precedence. Copy
`.env.example` to `.env.local` for a new deployment. Never commit `.env.local`.

Required:

- `OPENAI_API_KEY`
- `PILLSCAN_TFDA_CATALOG_PATH`: local SQLite catalog; defaults to
  `.data/tfda/catalog.sqlite3`

Production additionally requires:

- `PILLSCAN_API_TOKEN`: bearer token required by `/v1/pills/analyze`
- `PILLSCAN_TRUSTED_HOSTS`: JSON array of accepted Host header values

## Run with uv

```bash
uv sync --locked
uv run pillscan-bootstrap
```

`pillscan-bootstrap` downloads the current TFDA and NHIA open datasets on first use, builds the
SQLite catalog atomically, then starts FastAPI. Later starts reuse the existing catalog. To sync
or rebuild explicitly:

```bash
uv run pillscan-catalog-sync
uv run pillscan-catalog-sync --force
```

Development with reload:

```bash
uv run uvicorn pillscan_server.main:app --host 0.0.0.0 --port 8000 --reload
```

## Run with Pixi

Pixi reads the same `pyproject.toml`, resolves Python and all PyPI dependencies, and provides a
one-command task suitable for the 5090 workstation:

```bash
pixi run serve
```

Quality gate:

```bash
pixi run check
```

## Run on Raspberry Pi with a container

Set production values in `.env.local`, then:

```bash
docker compose up --build -d
```

The image is based on `python:3.12-slim`, runs as an unprivileged user, has a read-only root
filesystem in Compose, drops Linux capabilities, and supports both ARM64 and AMD64 builders.
The catalog is persisted in the `pillscan-data` volume; the container downloads it only when the
volume is new.

## Medication catalog

The sync command joins these Taiwan government open datasets:

- TFDA dataset 37: active product permits, names, dosage forms, manufacturers, packaging and the
  international-barcode field
- TFDA dataset 39: package and leaflet document URLs
- TFDA dataset 42: pill shape, color, score marks, imprints and official reference images
- TFDA dataset 43: official ingredients, ingredient codes, amounts and units
- NHIA medication items: current NHI drug codes and ATC codes

The generated SQLite database is ignored by Git. The 2026-07-16 snapshot produces 22,347 unique
active products, 38,272 ingredient rows, 6,129 appearance records and 32,833 active NHI-code
links. TFDA currently leaves its international-barcode values empty in that snapshot, so `gtins`
correctly remains `[]` until official values appear.

See [docs/data-sources.md](docs/data-sources.md) for the database selection and licensing notes.

## API

Health checks:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

Analyze one medication image:

```bash
curl -X POST http://localhost:8000/v1/pills/analyze \
  -H "Authorization: Bearer $PILLSCAN_API_TOKEN" \
  -F "image=@capture.jpg" \
  -F "market=TW"
```

Interactive API documentation is available at `/docs` outside production.

### Stable response contract

Every successful analysis returns `schema_version: "1.2"` with four deliberately separate layers:

- `analysis` contains only image-derived observations from the vision provider.
- `resolution` contains authoritative catalog results. It remains `evidence_extracted` with
  `source: not_queried` only when the market is not Taiwan or image quality prevents lookup.
- `timings` reports upload reading, image normalization, rate-limit wait, concurrency wait,
  vision analysis, catalog resolution, and tracked pipeline total in milliseconds. The HTTP
  `Server-Timing` header also exposes these stages for client and browser profiling.
- `usage` reports provider input, cached input, output, reasoning, and total tokens without
  exposing prompts or image data.

Every documented response field is required. Unavailable scalar values are returned as `null`,
and unavailable collections as `[]`, so generated Swift and other API clients receive one stable
shape.

An exact catalog result has this shape:

```json
{
  "status": "catalog_exact",
  "source": "tfda_nhi",
  "catalog_version": "2026-07-16",
  "product": {
    "identifiers": {
      "tfda_permit_number": "衛署藥製字第012345號",
      "tfda_ingredient_codes": ["A001234"],
      "nhi_code": "AC12345100",
      "gtins": ["04712345678901"]
    },
    "brand_name_zh": "範例錠",
    "brand_name_en": "EXAMPLE TABLETS",
    "generic_display_name": "ACETAMINOPHEN 500 MG",
    "ingredients": [
      {
        "official_name": "ACETAMINOPHEN",
        "normalized_generic_name": "acetaminophen",
        "tfda_ingredient_code": "A001234",
        "prescription_label": null,
        "amount_description": null,
        "amount": "500",
        "unit": "MG"
      }
    ],
    "dosage_form": "錠劑",
    "manufacturer": "範例藥廠",
    "applicant": null,
    "indications": null,
    "source_urls": []
  },
  "candidates": []
}
```

`catalog_exact` is schema-invalid without a complete TFDA permit number. Combination medicines
use multiple `ingredients` entries rather than flattening their generic names into one string.

## Architecture

- Pydantic Settings validates configuration without exposing secrets.
- FastAPI lifespan owns one shared `AsyncOpenAI` client and closes it cleanly.
- FastAPI lifespan opens one read-only `aiosqlite` catalog connection and refuses to start when
  the required catalog is missing.
- `PillVisionAnalyzer` is a protocol, so an OpenAI provider can later be replaced by a local VLM
  worker on the 5090.
- Images are size-limited, decoded defensively, EXIF-rotated, metadata-stripped, and normalized
  off the event loop.
- Loguru emits request-scoped console logs in development and JSON logs in the production
  container without recording headers, bodies, or image data.
- `aiolimiter` bounds VLM calls per minute, while an `asyncio.Semaphore` caps concurrent analyses
  to protect Raspberry Pi memory and upstream API capacity.
- OpenAI Responses Structured Outputs are parsed directly into Pydantic models.
- The default `gpt-5.4-mini` high-detail path targets sub-five-second clear-package responses;
  images are capped at 2048 pixels before upload, matching the model's high-detail limit.
- The API accepts exactly one image and classifies it as `pill`, `package`, or `unknown`.
- Package includes medicine boxes, blisters, bottle labels, and medication bags. Clear visible
  product names, strengths, permit numbers, and manufacturers are resolved against TFDA.
- A package may resolve exactly from an official permit number or one unique high-confidence name
  and strength match. A loose pill additionally requires a high-confidence imprint, at least two
  matching appearance discriminators, a high score and a safe margin over the next candidate.
- Ambiguous cases return `catalog_candidates`; missing catalog matches return `catalog_no_match`.

## Evaluation

`pillscan-eval` runs a paced, resumable HTTP benchmark against the real FastAPI endpoint. It writes
one JSONL record per image immediately, plus JSON and Markdown summaries containing quality,
safety, token cost, throughput, and p50/p95/p99 latency metrics.

```bash
uv run pillscan-eval \
  /path/to/package_sample/manifest.json \
  /path/to/pill_sample/manifest.json \
  --output-dir eval-results/development-200
```

See [docs/evaluation.md](docs/evaluation.md) for dataset design, ground-truth format, metric
definitions, pricing overrides, and the recommended 50 → 200 → 1,000 sample progression.
