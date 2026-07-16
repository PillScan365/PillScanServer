# Medication data-source decision

PillScan identifies Taiwanese medication products with Taiwan government open data. International
databases can enrich ingredients later, but they must not issue the canonical product identity for
a Taiwan-market package or loose pill.

## Selected primary sources

| Need | Source | Identifier or fields | Update | Role |
| --- | --- | --- | --- | --- |
| Taiwan product identity | [TFDA active permits](https://data.gov.tw/dataset/9123) | Full `許可證字號`, names, form, manufacturer, packaging, international-barcode field | Weekly | Canonical product record |
| Loose-pill appearance | [TFDA drug appearance](https://data.gov.tw/dataset/9120) | Shape, color, score mark, imprint, size, image URL | Weekly | Candidate retrieval and comparison |
| Generic ingredients | [TFDA detailed ingredients](https://data.gov.tw/dataset/9121) | Ingredient name/code, label, amount and unit | Weekly | Official generic and strength data |
| Package evidence | [TFDA leaflet/box data](https://data.gov.tw/dataset/9117) | Box and leaflet URLs | Weekly | Source traceability |
| Reimbursement identity | [NHIA medication items](https://data.gov.tw/dataset/23715) | NHI drug code, ingredient, manufacturer, form and ATC code | Monthly | NHI-code enrichment |

All five selected datasets are free and published under Taiwan's Government Open Data License,
version 1.0. The catalog builder records source hashes and a catalog version, collapses TFDA's
duplicate manufacturing-process rows by permit number, and writes a new SQLite file atomically.

## Sources evaluated but not used for Taiwan product identity

| Source | Appropriate use | Why it is not the canonical PillScan ID source |
| --- | --- | --- |
| [RxNorm / RxNav](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html) | Normalizing US clinical-drug concepts and ingredient vocabulary | RxCUI describes the US RxNorm concept system, not a Taiwan TFDA licensed product |
| [DailyMed](https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm) | US SPL labels, NDCs, RxCUIs and packaging | US-market labels do not uniquely identify Taiwan products |
| [openFDA NDC](https://open.fda.gov/apis/drug/ndc/how-to-use-the-endpoint/) | US NDC product lookup | NDC is a US identifier |
| [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) | Molecular identifiers and chemical properties | A chemical compound is not a packaged medication product |
| NLM Pillbox archive | Historical research only | NLM retired the API and explicitly says the static data should not be used for pill identification |
| DrugBank and commercial pill-identification datasets | Potential licensed enrichment | They are not unrestricted open-data dependencies and cannot replace TFDA identity |

## Identity policy

- `tfda_permit_number` is the canonical Taiwan product ID.
- `nhi_code` is a reimbursement code and may be absent or change over time; it is not the primary
  identity.
- `tfda_ingredient_codes` identify ingredients, not the finished product.
- `gtins` identify package trade items when TFDA publishes values, but a product can have multiple
  packages and the current snapshot contains no populated values.
- The VLM extracts evidence only. All returned official codes come from the local catalog.
- Exact package resolution requires a visible official permit or one unique high-confidence name
  and strength match.
- Exact loose-pill resolution requires a high-confidence imprint plus appearance corroboration and
  a safe candidate margin. Otherwise the API returns ranked candidates.
