import csv
import json
import sqlite3
import zipfile
from datetime import date
from pathlib import Path

import pytest

from pillscan_server.catalog import TfdaCatalog
from pillscan_server.catalog_sync import (
    build_catalog,
    extract_gtins,
    normalize_search,
    normalize_text,
    split_multivalue,
    split_urls,
    sync_catalog,
)
from pillscan_server.models import (
    ImageQuality,
    ObservedImprint,
    PillVisualAnalysis,
    SubjectType,
    VisibleIdentifiers,
    VisualEvidence,
)

PERMIT_PACKAGE = "衛部藥製字第058256號"
PERMIT_PILL = "衛部藥製字第060001號"


def write_zip(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.json", json.dumps(rows, ensure_ascii=False))


def make_catalog(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    write_zip(
        raw / "licenses_active.json.zip",
        [
            {
                "許可證字號": PERMIT_PACKAGE,
                "中文品名": "百樂行膜衣錠20毫克",
                "英文品名": "Paroxin F.C. Tablets 20mg",
                "劑型": "膜衣錠",
                "適應症": "憂鬱症",
                "包裝": "鋁箔盒裝",
                "申請商名稱": "新瑞生物科技股份有限公司",
                "製造商名稱": "瑞士藥廠股份有限公司新市廠",
                "包裝與國際條碼": "盒裝 4006381333931",
            },
            {
                "許可證字號": PERMIT_PILL,
                "中文品名": "測試錠10毫克",
                "英文品名": "TEST TABLETS 10MG",
                "劑型": "錠劑",
                "適應症": "測試用途",
                "包裝": "盒裝",
                "申請商名稱": "測試申請商",
                "製造商名稱": "測試藥廠",
                "包裝與國際條碼": "",
            },
            {
                "許可證字號": "衛部藥製字第060002號",
                "中文品名": "無刻痕測試錠",
                "英文品名": "PLAIN TEST TABLETS",
                "劑型": "錠劑",
                "適應症": "測試用途",
                "包裝": "盒裝",
                "申請商名稱": "另一申請商",
                "製造商名稱": "另一藥廠",
                "包裝與國際條碼": "12345670",
            },
            {
                "許可證字號": PERMIT_PILL,
                "中文品名": "測試錠10毫克",
                "英文品名": "TEST TABLETS 10MG",
                "劑型": "錠劑",
                "適應症": "測試用途",
                "包裝": "盒裝",
                "申請商名稱": "測試申請商",
                "製造商名稱": "第二製造廠",
                "包裝與國際條碼": "",
            },
        ],
    )
    write_zip(
        raw / "appearance.json.zip",
        [
            {
                "許可證字號": PERMIT_PACKAGE,
                "形狀": "圓形",
                "顏色": "白色",
                "刻痕": "直線",
                "標註一": "PX",
                "標註二": "20",
                "外觀圖檔連結": "https://example.test/paroxin.jpg",
            },
            {
                "許可證字號": PERMIT_PILL,
                "形狀": "圓形",
                "顏色": "白色",
                "刻痕": "直線",
                "標註一": "C9",
                "標註二": "AB",
                "外觀圖檔連結": "https://example.test/c9.jpg",
            },
        ],
    )
    write_zip(
        raw / "ingredients.json.zip",
        [
            {
                "許可證字號": PERMIT_PACKAGE,
                "處方標示": "EACH TABLET CONTAINS",
                "成分名稱": "PAROXETINE HYDROCHLORIDE",
                "成分代碼": "PARO001",
                "含量描述": "相當於PAROXETINE 20MG",
                "含量": "22.8",
                "含量單位": "MG",
            },
            {
                "許可證字號": PERMIT_PILL,
                "處方標示": "EACH TABLET CONTAINS",
                "成分名稱": "TESTOL",
                "成分代碼": "TEST001",
                "含量描述": None,
                "含量": "10",
                "含量單位": "MG",
            },
        ],
    )
    write_zip(
        raw / "inserts.json.zip",
        [
            {
                "許可證字號": PERMIT_PACKAGE,
                "仿單圖檔連結": "https://example.test/leaflet.pdf",
                "外盒圖檔連結": "https://example.test/box.pdf",
            }
        ],
    )
    nhia = tmp_path / "nhia.csv"
    fields = [
        "藥品代號",
        "藥品英文名稱",
        "藥品中文名稱",
        "有效起日",
        "有效迄日",
        "ATC代碼",
        "藥品代碼超連結",
    ]
    with nhia.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "藥品代號": "OLD5825610",
                    "藥品英文名稱": "Paroxin F.C. Tablets 20mg",
                    "藥品中文名稱": "百樂行膜衣錠20毫克",
                    "有效起日": "1100101",
                    "有效迄日": "1111231",
                    "ATC代碼": "N06AB05",
                    "藥品代碼超連結": "https://example.test/?licId=51058256",
                },
                {
                    "藥品代號": "AC58256100",
                    "藥品英文名稱": "Paroxin F.C. Tablets 20mg",
                    "藥品中文名稱": "百樂行膜衣錠20毫克",
                    "有效起日": "1130401",
                    "有效迄日": "9991231",
                    "ATC代碼": "N06AB05",
                    "藥品代碼超連結": "https://example.test/?licId=51058256",
                },
            ]
        )
    catalog_path = tmp_path / "catalog.sqlite3"
    report = build_catalog(
        raw,
        catalog_path,
        nhia_csv=nhia,
        catalog_version="2026-07-16",
        as_of=date(2026, 7, 16),
    )
    assert report.product_count == 3
    assert report.ingredient_count == 2
    assert report.appearance_count == 2
    assert report.nhi_code_count == 1
    assert report.gtin_count == 2
    return catalog_path


def analysis(
    *,
    subject_type: SubjectType,
    product_name: str = "",
    strength: str = "",
    permit_number: str = "",
    manufacturer: str = "",
    imprints: list[ObservedImprint] | None = None,
    colors: list[str] | None = None,
    shape: str = "unknown",
    score_marks: list[str] | None = None,
    other_text: list[str] | None = None,
    package_text: list[str] | None = None,
    confidence: str = "high",
) -> PillVisualAnalysis:
    return PillVisualAnalysis(
        subject_type=subject_type,
        state="direct_identifiers_visible"
        if subject_type is SubjectType.PACKAGE
        else "visual_evidence_only",
        image_quality=ImageQuality(
            sufficient_for_analysis=True,
            blur="none",
            glare="none",
            subject_fills_frame=True,
            text_readability="clear",
        ),
        visible_identifiers=VisibleIdentifiers(
            product_name=product_name,
            strength=strength,
            permit_number=permit_number,
            manufacturer=manufacturer,
            other_text=other_text or [],
            confidence=confidence,
        ),
        evidence=VisualEvidence(
            dosage_form="tablet",
            colors=colors or [],
            shape=shape,
            score_marks=score_marks or [],
            symbols_or_logos=[],
            imprints=imprints or [],
            package_text=package_text if package_text is not None else [product_name, strength],
            distinctive_features=[],
        ),
        candidate_hypotheses=[],
        uncertainty_reasons=[],
        next_actions=[],
    )


def test_normalization_and_gtin_validation() -> None:
    assert normalize_search(" Ｐaroxin F.C. ") == "paroxinfc"  # noqa: RUF001
    assert normalize_text("  Ａ  B\n C  ") == "A B C"  # noqa: RUF001
    assert normalize_text(None) is None
    assert normalize_text("  ") is None
    assert split_multivalue("白;;;黃色;;; ") == ["白", "黃色"]
    assert split_urls("https://a.test/1;https://a.test/2;;;https://a.test/3") == [
        "https://a.test/1",
        "https://a.test/2",
        "https://a.test/3",
    ]
    assert extract_gtins("盒裝 4006381333931;;;錯誤 1234567890123") == ["4006381333931"]


def test_sync_reuses_existing_catalog_and_can_build_from_cached_sources(tmp_path: Path) -> None:
    catalog_path = make_catalog(tmp_path)
    assert (
        sync_catalog(
            raw_dir=tmp_path / "raw",
            nhia_csv=tmp_path / "nhia.csv",
            catalog_path=catalog_path,
            force=False,
            skip_download=True,
        )
        is None
    )
    rebuilt = tmp_path / "rebuilt.sqlite3"
    report = sync_catalog(
        raw_dir=tmp_path / "raw",
        nhia_csv=tmp_path / "nhia.csv",
        catalog_path=rebuilt,
        force=False,
        skip_download=True,
    )
    assert report is not None
    assert report.product_count == 3


def test_catalog_build_rejects_missing_raw_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing TFDA datasets"):
        build_catalog(tmp_path / "missing", tmp_path / "catalog.sqlite3")


@pytest.mark.asyncio
async def test_package_name_resolves_to_official_tfda_product(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PACKAGE,
                product_name="百樂行膜衣錠20毫克",
                strength="20mg",
                manufacturer="瑞士藥廠股份有限公司新市廠",
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.source == "tfda_nhi"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PACKAGE
    assert resolution.product.identifiers.nhi_code == "AC58256100"
    assert resolution.product.identifiers.gtins == ["4006381333931"]
    assert resolution.product.ingredients[0].official_name == "PAROXETINE HYDROCHLORIDE"
    assert resolution.product.ingredients[0].tfda_ingredient_code == "PARO001"
    assert resolution.catalog_version == "2026-07-16"


@pytest.mark.asyncio
async def test_lower_confidence_package_returns_candidate_not_exact(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    observed = analysis(
        subject_type=SubjectType.PACKAGE,
        product_name="百樂行",
        strength="999mg",
        manufacturer="不同藥廠",
    )
    observed = observed.model_copy(
        update={
            "visible_identifiers": observed.visible_identifiers.model_copy(
                update={"confidence": "medium"}
            )
        }
    )
    try:
        resolution = await catalog.resolve(observed, market="TW")
    finally:
        await catalog.close()

    assert resolution.status == "catalog_candidates"
    assert resolution.candidates[0].product.identifiers.tfda_permit_number == PERMIT_PACKAGE
    assert "visible strength was not found" in resolution.candidates[0].conflicting_evidence[0]
    assert any(
        "manufacturer differs" in item for item in resolution.candidates[0].conflicting_evidence
    )


@pytest.mark.asyncio
async def test_unique_package_name_plus_separate_strength_resolves_exactly(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PACKAGE,
                product_name="Paroxin F.C. Tablets",
                strength="20mg",
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PACKAGE


@pytest.mark.asyncio
async def test_split_visible_package_name_is_reconstructed_exactly(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PACKAGE,
                product_name="百樂行",
                other_text=["膜衣錠20毫克", "30錠"],
                package_text=["百樂行", "膜衣錠20毫克", "30錠"],
                confidence="medium",
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PACKAGE


@pytest.mark.asyncio
async def test_package_text_name_falls_back_when_product_name_field_is_empty(
    tmp_path: Path,
) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PACKAGE,
                product_name="",
                package_text=["百樂行膜衣錠20毫克"],
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PACKAGE


@pytest.mark.asyncio
async def test_visible_permit_number_is_an_authoritative_exact_match(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(subject_type=SubjectType.PACKAGE, permit_number=PERMIT_PILL),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PILL


@pytest.mark.asyncio
async def test_six_digit_permit_serial_can_resolve_when_unique(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(subject_type=SubjectType.PACKAGE, permit_number="060001"),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PILL


@pytest.mark.asyncio
async def test_permit_transcribed_in_package_text_is_used(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    observed = analysis(subject_type=SubjectType.PACKAGE)
    observed = observed.model_copy(
        update={
            "evidence": observed.evidence.model_copy(
                update={"package_text": [f"核准字號 {PERMIT_PACKAGE}"]}
            )
        }
    )
    try:
        resolution = await catalog.resolve(observed, market="TW")
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PACKAGE


@pytest.mark.asyncio
async def test_invalid_visible_permit_falls_back_to_product_name_search(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PACKAGE,
                product_name="百樂行膜衣錠20毫克",
                strength="20mg",
                permit_number="衛部藥製字第999999號",
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PACKAGE


@pytest.mark.asyncio
async def test_unique_high_confidence_pill_signature_can_resolve_exactly(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PILL,
                imprints=[ObservedImprint(text="C9", alternatives=[], confidence="high")],
                colors=["white"],
                shape="round",
                score_marks=["line"],
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_exact"
    assert resolution.product is not None
    assert resolution.product.identifiers.tfda_permit_number == PERMIT_PILL
    assert resolution.product.generic_display_name == "TESTOL 10 MG"


@pytest.mark.asyncio
async def test_lower_confidence_pill_match_returns_ranked_candidates(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PILL,
                imprints=[ObservedImprint(text="C9", alternatives=["G9"], confidence="medium")],
                colors=["white"],
                shape="round",
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_candidates"
    assert resolution.product is None
    assert resolution.candidates[0].product.identifiers.tfda_permit_number == PERMIT_PILL
    assert resolution.candidates[0].score > 0.8
    assert "visible imprint matches" in resolution.candidates[0].matching_evidence[0]


@pytest.mark.asyncio
async def test_pill_without_imprint_never_resolves_exactly(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(
                subject_type=SubjectType.PILL,
                colors=["white"],
                shape="round",
                score_marks=["line"],
            ),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_candidates"
    assert len(resolution.candidates) == 2


@pytest.mark.asyncio
async def test_low_quality_and_unknown_images_skip_catalog_lookup(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    poor = analysis(subject_type=SubjectType.PILL)
    poor = poor.model_copy(
        update={
            "state": "needs_better_image",
            "image_quality": poor.image_quality.model_copy(
                update={"sufficient_for_analysis": False, "blur": "severe"}
            ),
        }
    )
    unknown = analysis(subject_type=SubjectType.PILL).model_copy(
        update={"subject_type": SubjectType.UNKNOWN, "state": "no_visual_match"}
    )
    try:
        poor_resolution = await catalog.resolve(poor, market="TW")
        unknown_resolution = await catalog.resolve(unknown, market="TW")
    finally:
        await catalog.close()

    assert poor_resolution.status == "needs_better_image"
    assert poor_resolution.source == "not_queried"
    assert unknown_resolution.status == "not_medication_image"


@pytest.mark.asyncio
async def test_unmatched_package_returns_catalog_no_match(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(subject_type=SubjectType.PACKAGE, product_name="完全不存在的藥"),
            market="TW",
        )
    finally:
        await catalog.close()

    assert resolution.status == "catalog_no_match"
    assert resolution.source == "tfda"
    assert resolution.product is None
    assert resolution.candidates == []


@pytest.mark.asyncio
async def test_non_taiwan_market_does_not_query_tfda(tmp_path: Path) -> None:
    catalog = await TfdaCatalog.open(make_catalog(tmp_path))
    try:
        resolution = await catalog.resolve(
            analysis(subject_type=SubjectType.PACKAGE, product_name="百樂行膜衣錠20毫克"),
            market="US",
        )
    finally:
        await catalog.close()

    assert resolution.status == "evidence_extracted"
    assert resolution.source == "not_queried"


@pytest.mark.asyncio
async def test_catalog_open_validates_file_and_schema(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await TfdaCatalog.open(tmp_path / "missing.sqlite3")

    invalid = tmp_path / "invalid.sqlite3"
    connection = sqlite3.connect(invalid)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO metadata VALUES ('schema_version', '999')")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="Unsupported TFDA catalog schema"):
        await TfdaCatalog.open(invalid)
