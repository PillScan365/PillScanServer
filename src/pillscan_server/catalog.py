from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import aiosqlite

from pillscan_server.catalog_sync import CATALOG_SCHEMA_VERSION, normalize_search, normalize_text
from pillscan_server.models import (
    CatalogCandidate,
    DrugIngredient,
    DrugProduct,
    DrugResolution,
    PillVisualAnalysis,
    ProductIdentifiers,
    ResolutionSource,
    ResolutionStatus,
    SubjectType,
)

MAX_CANDIDATE_POOL = 250
MIN_CANDIDATE_SCORE = 0.25
SERIAL_PATTERN = re.compile(r"(\d{6})(?!\d)")


@dataclass(frozen=True, slots=True)
class SearchRecord:
    permit_number: str
    name_zh: str | None
    name_en: str | None
    dosage_form: str | None
    manufacturer: str | None
    applicant: str | None
    indications: str | None
    package_description: str | None
    ingredient_text: str
    shapes: tuple[str, ...]
    colors: tuple[str, ...]
    score_marks: tuple[str, ...]
    imprints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    record: SearchRecord
    score: float
    matching_evidence: tuple[str, ...]
    conflicting_evidence: tuple[str, ...]


class TfdaCatalog:
    def __init__(
        self,
        connection: aiosqlite.Connection,
        *,
        catalog_version: str,
        record_count: int,
    ) -> None:
        self._connection = connection
        self.catalog_version = catalog_version
        self.record_count = record_count

    @classmethod
    async def open(cls, path: Path) -> TfdaCatalog:
        if not await asyncio.to_thread(path.is_file):
            raise FileNotFoundError(f"TFDA catalog not found: {path}")
        connection = await aiosqlite.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            await connection.execute("PRAGMA query_only = ON")
            cursor = await connection.execute("SELECT key, value FROM metadata")
            metadata = {row["key"]: row["value"] for row in await cursor.fetchall()}
            if metadata.get("schema_version") != CATALOG_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported TFDA catalog schema: {metadata.get('schema_version', 'missing')}"
                )
            catalog_version = metadata["catalog_version"]
            record_count = int(metadata["product_count"])
        except BaseException:
            await connection.close()
            raise
        return cls(
            connection,
            catalog_version=catalog_version,
            record_count=record_count,
        )

    async def close(self) -> None:
        await self._connection.close()

    async def resolve(self, analysis: PillVisualAnalysis, *, market: str) -> DrugResolution:
        if market.strip().upper() not in {"TW", "TAIWAN"}:
            return _not_queried_resolution(analysis)
        if analysis.state == "needs_better_image":
            return _not_queried_resolution(analysis)
        if analysis.subject_type is SubjectType.UNKNOWN or analysis.state == "no_visual_match":
            return _not_queried_resolution(analysis)

        direct = await self._resolve_permit(analysis)
        if direct is not None:
            return direct
        if analysis.subject_type is SubjectType.PACKAGE:
            return await self._resolve_package(analysis)
        if analysis.subject_type is SubjectType.PILL:
            return await self._resolve_pill(analysis)

    async def _resolve_permit(self, analysis: PillVisualAnalysis) -> DrugResolution | None:
        visible = analysis.visible_identifiers.permit_number
        if not visible:
            for text in analysis.evidence.package_text:
                if "字第" in text and "號" in text:
                    visible = text
                    break
        if not visible:
            return None

        normalized = normalize_search(visible)
        cursor = await self._connection.execute(
            "SELECT permit_number FROM products WHERE normalized_permit = ?",
            (normalized,),
        )
        row = await cursor.fetchone()
        if row is None:
            serials = SERIAL_PATTERN.findall(visible)
            if serials:
                cursor = await self._connection.execute(
                    "SELECT permit_number FROM products WHERE serial_number = ? LIMIT 2",
                    (serials[-1],),
                )
                rows = list(await cursor.fetchall())
                if len(rows) == 1:
                    row = rows[0]
        if row is None:
            return None
        product = await self._load_product(row["permit_number"])
        return DrugResolution(
            status=ResolutionStatus.CATALOG_EXACT,
            source=_source_for_products([product]),
            product=product,
            candidates=[],
            catalog_version=self.catalog_version,
        )

    async def _resolve_package(self, analysis: PillVisualAnalysis) -> DrugResolution:
        visible = analysis.visible_identifiers
        query_terms = [visible.product_name, *analysis.evidence.package_text, *visible.other_text]
        normalized_terms = list(
            dict.fromkeys(
                term for value in query_terms if len(term := normalize_search(value)) >= 3
            )
        )[:12]
        if not normalized_terms:
            return self._no_match()

        permits: set[str] = set()
        for term in normalized_terms:
            cursor = await self._connection.execute(
                """
                SELECT DISTINCT permit_number
                FROM search_terms
                WHERE normalized_term = ?
                   OR normalized_term LIKE '%' || ? || '%'
                   OR ? LIKE '%' || normalized_term || '%'
                LIMIT ?
                """,
                (term, term, term, MAX_CANDIDATE_POOL),
            )
            permits.update(row["permit_number"] for row in await cursor.fetchall())
            if len(permits) >= MAX_CANDIDATE_POOL:
                break
        if not permits:
            return self._no_match()

        records = await self._load_search_records(permits)
        scored = sorted(
            (self._score_package(record, analysis) for record in records),
            key=lambda item: (-item.score, item.record.permit_number),
        )
        scored = [item for item in scored if item.score >= MIN_CANDIDATE_SCORE]
        if not scored:
            return self._no_match()

        strong_matches = [
            item
            for item in scored
            if _strong_package_match(
                visible.product_name,
                visible.strength,
                item,
            )
        ]
        if visible.confidence == "high" and len(strong_matches) == 1:
            product = await self._load_product(strong_matches[0].record.permit_number)
            return DrugResolution(
                status=ResolutionStatus.CATALOG_EXACT,
                source=_source_for_products([product]),
                product=product,
                candidates=[],
                catalog_version=self.catalog_version,
            )
        return await self._candidate_resolution(scored)

    def _score_package(
        self,
        record: SearchRecord,
        analysis: PillVisualAnalysis,
    ) -> ScoredRecord:
        visible = analysis.visible_identifiers
        product_name = normalize_search(visible.product_name)
        names = [normalize_search(record.name_zh), normalize_search(record.name_en)]
        matching: list[str] = []
        conflicting: list[str] = []
        score = 0.0

        if product_name:
            if product_name in names:
                score += 0.7
                matching.append("visible product name exactly matches the TFDA product name")
            elif any(product_name in name or name in product_name for name in names if name):
                score += 0.52
                matching.append("visible product name is contained in the TFDA product name")
            else:
                similarity = max(
                    (SequenceMatcher(None, product_name, name).ratio() for name in names if name),
                    default=0.0,
                )
                score += similarity * 0.4
                if similarity >= 0.65:
                    matching.append("visible product name is textually similar")
                else:
                    conflicting.append("visible product name differs from the TFDA product name")

        combined = normalize_search(
            " ".join(
                filter(
                    None,
                    [
                        record.name_zh,
                        record.name_en,
                        record.ingredient_text,
                        record.package_description,
                    ],
                )
            )
        )
        strength = normalize_search(visible.strength)
        if strength:
            if strength in combined:
                score += 0.15
                matching.append("visible strength matches TFDA product or ingredient text")
            else:
                conflicting.append("visible strength was not found in TFDA product text")

        manufacturer = normalize_search(visible.manufacturer)
        candidate_manufacturer = normalize_search(record.manufacturer)
        if manufacturer:
            if manufacturer in candidate_manufacturer or candidate_manufacturer in manufacturer:
                score += 0.15
                matching.append("visible manufacturer matches TFDA")
            elif candidate_manufacturer:
                conflicting.append("visible manufacturer differs from TFDA")

        package_terms = {
            normalize_search(term)
            for term in [*analysis.evidence.package_text, *visible.other_text]
            if len(normalize_search(term)) >= 3
        }
        overlap_count = sum(
            term in combined or combined in term for term in package_terms if combined and term
        )
        if overlap_count:
            score += min(0.1, overlap_count * 0.04)
            matching.append("additional package text supports the catalog record")
        return ScoredRecord(
            record=record,
            score=min(1.0, score),
            matching_evidence=tuple(matching),
            conflicting_evidence=tuple(conflicting),
        )

    async def _resolve_pill(self, analysis: PillVisualAnalysis) -> DrugResolution:
        primary_imprints = {
            normalize_search(imprint.text): imprint.confidence
            for imprint in analysis.evidence.imprints
            if normalize_search(imprint.text)
        }
        alternative_imprints = {
            normalize_search(alternative)
            for imprint in analysis.evidence.imprints
            for alternative in imprint.alternatives
            if normalize_search(alternative)
        }
        all_imprints = set(primary_imprints) | alternative_imprints

        if all_imprints:
            placeholders = ",".join("?" for _ in all_imprints)
            cursor = await self._connection.execute(
                f"""
                SELECT DISTINCT permit_number
                FROM imprint_index
                WHERE normalized_imprint IN ({placeholders})
                LIMIT ?
                """,  # noqa: S608 - placeholders are generated, values remain parameterized
                (*sorted(all_imprints), MAX_CANDIDATE_POOL),
            )
        else:
            cursor = await self._connection.execute(
                "SELECT permit_number FROM appearances LIMIT ?",
                (MAX_CANDIDATE_POOL,),
            )
        permits = {row["permit_number"] for row in await cursor.fetchall()}
        if not permits:
            return self._no_match()

        records = await self._load_search_records(permits)
        scored = sorted(
            (self._score_pill(record, analysis) for record in records),
            key=lambda item: (-item.score, item.record.permit_number),
        )
        scored = [item for item in scored if item.score >= MIN_CANDIDATE_SCORE]
        if not scored:
            return self._no_match()

        top = scored[0]
        second_score = scored[1].score if len(scored) > 1 else 0.0
        high_imprint_match = any(
            confidence == "high"
            and imprint in {normalize_search(item) for item in top.record.imprints}
            for imprint, confidence in primary_imprints.items()
        )
        visible_discriminators = sum(
            bool(values)
            for values in (
                analysis.evidence.colors,
                [analysis.evidence.shape] if analysis.evidence.shape else [],
                analysis.evidence.score_marks,
            )
        )
        if (
            analysis.image_quality.sufficient_for_analysis
            and high_imprint_match
            and visible_discriminators >= 2
            and top.score >= 0.88
            and top.score - second_score >= 0.12
        ):
            product = await self._load_product(top.record.permit_number)
            return DrugResolution(
                status=ResolutionStatus.CATALOG_EXACT,
                source=_source_for_products([product]),
                product=product,
                candidates=[],
                catalog_version=self.catalog_version,
            )
        return await self._candidate_resolution(scored)

    def _score_pill(
        self,
        record: SearchRecord,
        analysis: PillVisualAnalysis,
    ) -> ScoredRecord:
        candidate_imprints = {normalize_search(item) for item in record.imprints}
        matching: list[str] = []
        conflicting: list[str] = []
        score = 0.0

        best_imprint_score = 0.0
        for imprint in analysis.evidence.imprints:
            primary = normalize_search(imprint.text)
            if primary and primary in candidate_imprints:
                best_imprint_score = max(
                    best_imprint_score,
                    {"high": 0.7, "medium": 0.62, "low": 0.5}[imprint.confidence],
                )
            elif any(
                normalize_search(alternative) in candidate_imprints
                for alternative in imprint.alternatives
            ):
                best_imprint_score = max(best_imprint_score, 0.45)
        if best_imprint_score:
            score += best_imprint_score
            matching.append("visible imprint matches TFDA appearance data")
        elif analysis.evidence.imprints:
            conflicting.append("visible imprint does not match TFDA appearance data")

        if _feature_matches(analysis.evidence.shape, record.shapes, feature="shape"):
            score += 0.1
            matching.append("shape matches TFDA appearance data")
        elif analysis.evidence.shape and analysis.evidence.shape != "unknown":
            conflicting.append("shape differs from TFDA appearance data")

        if any(
            _feature_matches(color, record.colors, feature="color")
            for color in analysis.evidence.colors
        ):
            score += 0.1
            matching.append("color matches TFDA appearance data")
        elif analysis.evidence.colors:
            conflicting.append("color differs from TFDA appearance data")

        if any(
            _feature_matches(mark, record.score_marks, feature="score")
            for mark in analysis.evidence.score_marks
        ):
            score += 0.05
            matching.append("score mark matches TFDA appearance data")
        dosage_form = normalize_search(analysis.evidence.dosage_form)
        if (
            dosage_form
            and dosage_form != "unknown"
            and _dosage_form_matches(dosage_form, record.dosage_form)
        ):
            score += 0.05
            matching.append("dosage form matches TFDA")

        return ScoredRecord(
            record=record,
            score=min(1.0, score),
            matching_evidence=tuple(matching),
            conflicting_evidence=tuple(conflicting),
        )

    async def _load_search_records(self, permits: set[str]) -> list[SearchRecord]:
        if not permits:
            return []
        placeholders = ",".join("?" for _ in permits)
        cursor = await self._connection.execute(
            f"""
            SELECT
                p.*,
                COALESCE((
                    SELECT group_concat(
                        i.official_name || ' ' || COALESCE(i.amount, '') || ' '
                            || COALESCE(i.unit, ''),
                        ' | '
                    )
                    FROM ingredients i
                    WHERE i.permit_number = p.permit_number
                ), '') AS ingredient_text,
                a.shapes_json,
                a.colors_json,
                a.score_marks_json,
                a.imprints_json
            FROM products p
            LEFT JOIN appearances a ON a.permit_number = p.permit_number
            WHERE p.permit_number IN ({placeholders})
            """,  # noqa: S608 - placeholders are generated, values remain parameterized
            tuple(sorted(permits)),
        )
        rows = await cursor.fetchall()
        return [
            SearchRecord(
                permit_number=row["permit_number"],
                name_zh=row["name_zh"],
                name_en=row["name_en"],
                dosage_form=row["dosage_form"],
                manufacturer=row["manufacturer"],
                applicant=row["applicant"],
                indications=row["indications"],
                package_description=row["package_description"],
                ingredient_text=row["ingredient_text"],
                shapes=_json_tuple(row["shapes_json"]),
                colors=_json_tuple(row["colors_json"]),
                score_marks=_json_tuple(row["score_marks_json"]),
                imprints=_json_tuple(row["imprints_json"]),
            )
            for row in rows
        ]

    async def _load_product(self, permit_number: str) -> DrugProduct:
        cursor = await self._connection.execute(
            "SELECT * FROM products WHERE permit_number = ?", (permit_number,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise LookupError(f"Catalog product disappeared: {permit_number}")

        cursor = await self._connection.execute(
            "SELECT * FROM ingredients WHERE permit_number = ? ORDER BY id", (permit_number,)
        )
        ingredient_rows = await cursor.fetchall()
        ingredients = [
            DrugIngredient(
                official_name=item["official_name"],
                normalized_generic_name=(normalize_text(item["official_name"]) or "").casefold()
                or None,
                tfda_ingredient_code=item["ingredient_code"],
                prescription_label=item["prescription_label"],
                amount_description=item["amount_description"],
                amount=item["amount"],
                unit=item["unit"],
            )
            for item in ingredient_rows
        ]
        cursor = await self._connection.execute(
            "SELECT code_type, code FROM product_codes WHERE permit_number = ? ORDER BY code",
            (permit_number,),
        )
        codes: dict[str, list[str]] = {"nhi": [], "gtin": [], "atc": []}
        for item in await cursor.fetchall():
            codes[item["code_type"]].append(item["code"])
        cursor = await self._connection.execute(
            "SELECT url FROM source_urls WHERE permit_number = ? ORDER BY url", (permit_number,)
        )
        source_urls = [item["url"] for item in await cursor.fetchall()]
        ingredient_codes = list(
            dict.fromkeys(
                ingredient.tfda_ingredient_code
                for ingredient in ingredients
                if ingredient.tfda_ingredient_code
            )
        )
        generic_display_name = (
            " + ".join(
                " ".join(
                    filter(None, [ingredient.official_name, ingredient.amount, ingredient.unit])
                )
                for ingredient in ingredients
            )
            or None
        )
        return DrugProduct(
            identifiers=ProductIdentifiers(
                tfda_permit_number=permit_number,
                tfda_ingredient_codes=ingredient_codes,
                nhi_code=codes["nhi"][0] if codes["nhi"] else None,
                gtins=codes["gtin"],
            ),
            brand_name_zh=row["name_zh"],
            brand_name_en=row["name_en"],
            generic_display_name=generic_display_name,
            ingredients=ingredients,
            dosage_form=row["dosage_form"],
            manufacturer=row["manufacturer"],
            applicant=row["applicant"],
            indications=row["indications"],
            source_urls=source_urls,
        )

    async def _candidate_resolution(self, scored: list[ScoredRecord]) -> DrugResolution:
        selected = scored[:5]
        products = [await self._load_product(item.record.permit_number) for item in selected]
        candidates = [
            CatalogCandidate(
                product=product,
                score=round(item.score, 4),
                matching_evidence=list(item.matching_evidence),
                conflicting_evidence=list(item.conflicting_evidence),
            )
            for item, product in zip(selected, products, strict=True)
        ]
        return DrugResolution(
            status=ResolutionStatus.CATALOG_CANDIDATES,
            source=_source_for_products(products),
            product=None,
            candidates=candidates,
            catalog_version=self.catalog_version,
        )

    def _no_match(self) -> DrugResolution:
        return DrugResolution(
            status=ResolutionStatus.CATALOG_NO_MATCH,
            source=ResolutionSource.TFDA,
            product=None,
            candidates=[],
            catalog_version=self.catalog_version,
        )


def _not_queried_resolution(analysis: PillVisualAnalysis) -> DrugResolution:
    if analysis.state == "needs_better_image":
        status = ResolutionStatus.NEEDS_BETTER_IMAGE
    elif analysis.state == "no_visual_match" or analysis.subject_type is SubjectType.UNKNOWN:
        status = ResolutionStatus.NOT_MEDICATION_IMAGE
    else:
        status = ResolutionStatus.EVIDENCE_EXTRACTED
    return DrugResolution(
        status=status,
        source=ResolutionSource.NOT_QUERIED,
        product=None,
        candidates=[],
        catalog_version=None,
    )


def _source_for_products(products: list[DrugProduct]) -> ResolutionSource:
    if any(product.identifiers.nhi_code for product in products):
        return ResolutionSource.TFDA_NHI
    return ResolutionSource.TFDA


def _exact_name_match(value: str, record: SearchRecord) -> bool:
    normalized = normalize_search(value)
    return bool(normalized) and normalized in {
        normalize_search(record.name_zh),
        normalize_search(record.name_en),
    }


def _strong_package_match(
    product_name: str,
    strength: str,
    candidate: ScoredRecord,
) -> bool:
    observed_name = normalize_search(product_name)
    catalog_names = [
        normalize_search(candidate.record.name_zh),
        normalize_search(candidate.record.name_en),
    ]
    exact_name = _exact_name_match(product_name, candidate.record)
    contained_name = bool(observed_name) and any(
        observed_name in name or name in observed_name for name in catalog_names if name
    )
    if not exact_name and not contained_name:
        return False
    observed_strength = normalize_search(strength)
    if not observed_strength:
        return exact_name
    catalog_text = normalize_search(
        " ".join(
            filter(
                None,
                [
                    candidate.record.name_zh,
                    candidate.record.name_en,
                    candidate.record.ingredient_text,
                    candidate.record.package_description,
                ],
            )
        )
    )
    return observed_strength in catalog_text and candidate.score >= 0.65


def _json_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    payload = json.loads(value)
    if not isinstance(payload, list):
        return ()
    return tuple(str(item) for item in payload)


FEATURE_ALIASES = {
    "shape": {
        "round": ("round", "圓", "圓形"),
        "oval": ("oval", "橢圓", "橢圓形"),
        "oblong": ("oblong", "長圓", "長橢圓"),
        "capsule": ("capsule", "膠囊", "長圓"),
        "triangle": ("triangle", "三角", "三角形"),
        "square": ("square", "方", "四方", "正方形"),
        "diamond": ("diamond", "菱形"),
    },
    "color": {
        "white": ("white", "白", "白色"),
        "yellow": ("yellow", "黃", "黃色"),
        "red": ("red", "紅", "紅色"),
        "pink": ("pink", "粉紅", "粉色"),
        "blue": ("blue", "藍", "藍色"),
        "green": ("green", "綠", "綠色"),
        "orange": ("orange", "橙", "橘", "橘色"),
        "brown": ("brown", "棕", "褐", "咖啡"),
        "gray": ("gray", "grey", "灰", "灰色"),
        "black": ("black", "黑", "黑色"),
        "purple": ("purple", "紫", "紫色"),
        "transparent": ("transparent", "透明"),
    },
    "score": {
        "line": ("line", "singleline", "直線", "一字"),
        "cross": ("cross", "十字", "十字線"),
        "none": ("none", "無", "無刻痕"),
    },
}


def _feature_matches(observed: str, candidates: tuple[str, ...], *, feature: str) -> bool:
    observed_normalized = normalize_search(observed)
    candidate_normalized = {normalize_search(candidate) for candidate in candidates}
    if not observed_normalized or not candidate_normalized:
        return False
    if any(
        observed_normalized in candidate or candidate in observed_normalized
        for candidate in candidate_normalized
    ):
        return True
    for aliases in FEATURE_ALIASES[feature].values():
        normalized_aliases = {normalize_search(alias) for alias in aliases}
        if observed_normalized in normalized_aliases and candidate_normalized & normalized_aliases:
            return True
    return False


def _dosage_form_matches(observed: str, candidate: str | None) -> bool:
    normalized_candidate = normalize_search(candidate)
    if observed == "tablet":
        return "錠" in (candidate or "") or "tablet" in normalized_candidate
    if observed in {"capsule", "softgel"}:
        return "膠囊" in (candidate or "") or "capsule" in normalized_candidate
    return observed in normalized_candidate or normalized_candidate in observed
