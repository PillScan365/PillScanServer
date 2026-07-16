from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

CATALOG_SCHEMA_VERSION = "1"
TFDA_BASE_URL = "https://data.fda.gov.tw/data/opendata/export/{dataset_id}/json"
TFDA_DATASETS = {
    "appearance": 42,
    "licenses_active": 37,
    "ingredients": 43,
    "inserts": 39,
}
NHIA_DATASET_URL = "https://info.nhi.gov.tw/api/iode0000s01/Dataset?rId=A21030000I-E41001-001"
TFDA_SOURCE_URLS = {
    "licenses_active": "https://data.gov.tw/dataset/9123",
    "appearance": "https://data.gov.tw/dataset/9120",
    "ingredients": "https://data.gov.tw/dataset/9121",
    "inserts": "https://data.gov.tw/dataset/9117",
}
NHIA_SOURCE_URL = "https://data.gov.tw/dataset/23715"
MULTIVALUE_SEPARATOR = ";;;"
LICENSE_ID_PATTERN = re.compile(r"[?&]licId=(\d+)", re.IGNORECASE)
SERIAL_PATTERN = re.compile(r"(\d{6})(?!\d)")
GTIN_PATTERN = re.compile(r"(?<!\d)(\d{8}|\d{12,14})(?!\d)")


@dataclass(frozen=True, slots=True)
class CatalogBuildReport:
    product_count: int
    ingredient_count: int
    appearance_count: int
    nhi_code_count: int
    gtin_count: int


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"[\t\r\n ]+", " ", text)
    return text or None


def normalize_search(value: object) -> str:
    text = normalize_text(value)
    if text is None:
        return ""
    return "".join(character.casefold() for character in text if character.isalnum())


def split_multivalue(value: object) -> list[str]:
    text = normalize_text(value)
    if text is None:
        return []
    return [item for raw in text.split(MULTIVALUE_SEPARATOR) if (item := normalize_text(raw))]


def split_urls(value: object) -> list[str]:
    urls: list[str] = []
    for group in split_multivalue(value):
        urls.extend(part for part in re.split(r";(?=https?://)", group) if part)
    return urls


def extract_gtins(value: object) -> list[str]:
    text = normalize_text(value) or ""
    return list(
        dict.fromkeys(
            match.group(1) for match in GTIN_PATTERN.finditer(text) if _valid_gtin(match.group(1))
        )
    )


def _valid_gtin(value: str) -> bool:
    body = value[:-1]
    weighted_sum = sum(
        int(digit) * (3 if index % 2 == 0 else 1) for index, digit in enumerate(reversed(body))
    )
    return (10 - weighted_sum % 10) % 10 == int(value[-1])


def _unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _permit_serial(permit_number: str) -> str | None:
    matches = SERIAL_PATTERN.findall(permit_number)
    return matches[-1] if matches else None


def _load_zipped_json(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        json_files = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if len(json_files) != 1:
            raise ValueError(f"Expected one JSON file in {path}, found {json_files}")
        with archive.open(json_files[0]) as source:
            payload = json.load(source)
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return payload


def _group_by_permit(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        permit = normalize_text(row.get("許可證字號"))
        if permit:
            grouped[permit].append(row)
    return grouped


def _deduplicate_licenses(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse TFDA's per-manufacturing-process rows into one product permit record."""
    merged: dict[str, dict[str, Any]] = {}
    manufacturers: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        permit = normalize_text(row.get("許可證字號"))
        if permit is None:
            continue
        target = merged.setdefault(permit, dict(row))
        for key, value in row.items():
            if not normalize_text(target.get(key)) and normalize_text(value):
                target[key] = value
        manufacturer = normalize_text(row.get("製造商名稱"))
        if manufacturer and manufacturer not in manufacturers[permit]:
            manufacturers[permit].append(manufacturer)
    for permit, values in manufacturers.items():
        merged[permit]["製造商名稱"] = ";".join(values)
    return list(merged.values())


def _roc_date(value: str | None) -> date | None:
    if not value or not value.isdigit() or len(value) != 7:
        return None
    year = int(value[:3]) + 1911
    try:
        return date(year, int(value[3:5]), int(value[5:7]))
    except ValueError:
        return None


def _load_nhia_codes(
    path: Path | None,
    licenses: Sequence[Mapping[str, Any]],
    *,
    as_of: date,
) -> dict[str, list[tuple[str, str | None]]]:
    if path is None or not path.exists():
        return {}

    permits_by_serial: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for license_row in licenses:
        permit = normalize_text(license_row.get("許可證字號"))
        if permit and (serial := _permit_serial(permit)):
            permits_by_serial[serial].append(license_row)

    latest_by_code: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            code = normalize_text(row.get("藥品代號"))
            license_url = normalize_text(row.get("藥品代碼超連結")) or ""
            match = LICENSE_ID_PATTERN.search(license_url)
            if not code or not match:
                continue
            end_text = normalize_text(row.get("有效迄日"))
            end_date = _roc_date(end_text)
            if end_text != "9991231" and end_date is not None and end_date < as_of:
                continue
            current = latest_by_code.get(code)
            start_text = normalize_text(row.get("有效起日")) or ""
            if current is None or start_text > (current.get("有效起日") or ""):
                latest_by_code[code] = row

    result: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for code, row in latest_by_code.items():
        license_url = normalize_text(row.get("藥品代碼超連結")) or ""
        match = LICENSE_ID_PATTERN.search(license_url)
        if match is None:
            continue
        serial = match.group(1)[-6:]
        candidates = permits_by_serial.get(serial, [])
        if len(candidates) > 1:
            name_zh = normalize_search(row.get("藥品中文名稱"))
            name_en = normalize_search(row.get("藥品英文名稱"))
            matched = [
                candidate
                for candidate in candidates
                if name_zh == normalize_search(candidate.get("中文品名"))
                or name_en == normalize_search(candidate.get("英文品名"))
            ]
            candidates = matched
        if len(candidates) != 1:
            continue
        permit = normalize_text(candidates[0].get("許可證字號"))
        if permit:
            result[permit].append((code, normalize_text(row.get("ATC代碼"))))

    return {permit: list(dict.fromkeys(values)) for permit, values in result.items()}


def build_catalog(
    raw_dir: Path,
    destination: Path,
    *,
    nhia_csv: Path | None = None,
    catalog_version: str | None = None,
    as_of: date | None = None,
) -> CatalogBuildReport:
    paths = {name: raw_dir / f"{name}.json.zip" for name in TFDA_DATASETS}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TFDA datasets: {', '.join(missing)}")

    licenses = _deduplicate_licenses(_load_zipped_json(paths["licenses_active"]))
    appearances = _group_by_permit(_load_zipped_json(paths["appearance"]))
    ingredients = _group_by_permit(_load_zipped_json(paths["ingredients"]))
    inserts = _group_by_permit(_load_zipped_json(paths["inserts"]))
    effective_date = as_of or date.today()
    version = catalog_version or effective_date.isoformat()
    nhia_codes = _load_nhia_codes(nhia_csv, licenses, as_of=effective_date)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    temporary_path.unlink()

    connection = sqlite3.connect(temporary_path)
    try:
        _create_schema(connection)
        ingredient_count = 0
        appearance_count = 0
        nhi_code_count = 0
        gtin_count = 0
        product_count = 0

        for license_row in licenses:
            permit = normalize_text(license_row.get("許可證字號"))
            if permit is None:
                continue
            product_count += 1
            name_zh = normalize_text(license_row.get("中文品名"))
            name_en = normalize_text(license_row.get("英文品名"))
            manufacturer = normalize_text(license_row.get("製造商名稱"))
            applicant = normalize_text(license_row.get("申請商名稱"))
            package_description = normalize_text(license_row.get("包裝"))
            connection.execute(
                """
                INSERT INTO products (
                    permit_number, normalized_permit, serial_number, name_zh, name_en,
                    normalized_name_zh, normalized_name_en, dosage_form, manufacturer,
                    normalized_manufacturer, applicant, indications, package_description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    permit,
                    normalize_search(permit),
                    _permit_serial(permit),
                    name_zh,
                    name_en,
                    normalize_search(name_zh),
                    normalize_search(name_en),
                    normalize_text(license_row.get("劑型")),
                    manufacturer,
                    normalize_search(manufacturer),
                    applicant,
                    normalize_text(license_row.get("適應症")),
                    package_description,
                ),
            )

            ingredient_names: list[str] = []
            for ingredient in ingredients.get(permit, []):
                official_name = normalize_text(ingredient.get("成分名稱"))
                if official_name is None:
                    continue
                ingredient_count += 1
                ingredient_names.append(official_name)
                connection.execute(
                    """
                    INSERT INTO ingredients (
                        permit_number, official_name, normalized_name, ingredient_code,
                        prescription_label, amount_description, amount, unit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        permit,
                        official_name,
                        normalize_search(official_name),
                        normalize_text(ingredient.get("成分代碼")),
                        normalize_text(ingredient.get("處方標示")),
                        normalize_text(ingredient.get("含量描述")),
                        normalize_text(ingredient.get("含量")),
                        normalize_text(ingredient.get("含量單位")),
                    ),
                )

            appearance_rows = appearances.get(permit, [])
            shapes = _unique(
                item for row in appearance_rows for item in split_multivalue(row.get("形狀"))
            )
            colors = _unique(
                item for row in appearance_rows for item in split_multivalue(row.get("顏色"))
            )
            score_marks = _unique(
                item for row in appearance_rows for item in split_multivalue(row.get("刻痕"))
            )
            imprints = _unique(
                item
                for row in appearance_rows
                for field in ("標註一", "標註二")
                for item in split_multivalue(row.get(field))
            )
            image_urls = _unique(
                item
                for row in appearance_rows
                for item in split_multivalue(row.get("外觀圖檔連結"))
            )
            if appearance_rows:
                appearance_count += 1
                connection.execute(
                    """
                    INSERT INTO appearances (
                        permit_number, shapes_json, colors_json, score_marks_json,
                        imprints_json, image_urls_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    tuple(
                        [permit]
                        + [
                            json.dumps(values, ensure_ascii=False)
                            for values in (shapes, colors, score_marks, imprints, image_urls)
                        ]
                    ),
                )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO imprint_index (
                        normalized_imprint, imprint, permit_number
                    ) VALUES (?, ?, ?)
                    """,
                    [(normalize_search(imprint), imprint, permit) for imprint in imprints],
                )

            gtins = extract_gtins(license_row.get("包裝與國際條碼"))
            gtin_count += len(gtins)
            connection.executemany(
                """
                INSERT OR IGNORE INTO product_codes (
                    permit_number, code_type, code
                ) VALUES (?, 'gtin', ?)
                """,
                [(permit, gtin) for gtin in gtins],
            )
            for nhi_code, atc_code in nhia_codes.get(permit, []):
                nhi_code_count += 1
                connection.execute(
                    """
                    INSERT OR IGNORE INTO product_codes (
                        permit_number, code_type, code
                    ) VALUES (?, 'nhi', ?)
                    """,
                    (permit, nhi_code),
                )
                if atc_code:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO product_codes (
                            permit_number, code_type, code
                        ) VALUES (?, 'atc', ?)
                        """,
                        (permit, atc_code),
                    )

            document_rows = inserts.get(permit, [])
            source_urls = _unique(
                [*TFDA_SOURCE_URLS.values(), NHIA_SOURCE_URL]
                + image_urls
                + [
                    url
                    for row in document_rows
                    for field in ("仿單圖檔連結", "外盒圖檔連結")
                    for url in split_urls(row.get(field))
                ]
            )
            connection.executemany(
                "INSERT OR IGNORE INTO source_urls (permit_number, url) VALUES (?, ?)",
                [(permit, url) for url in source_urls],
            )
            search_terms = _unique([name_zh, name_en, manufacturer, applicant, *ingredient_names])
            connection.executemany(
                """
                INSERT OR IGNORE INTO search_terms (
                    normalized_term, display_term, permit_number
                ) VALUES (?, ?, ?)
                """,
                [
                    (normalize_search(term), term, permit)
                    for term in search_terms
                    if normalize_search(term)
                ],
            )

        metadata = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "catalog_version": version,
            "generated_at": datetime.now(UTC).isoformat(),
            "product_count": str(product_count),
            "tfda_dataset_ids": json.dumps(TFDA_DATASETS, sort_keys=True),
            "tfda_sha256": json.dumps(
                {name: _sha256(path) for name, path in paths.items()}, sort_keys=True
            ),
            "nhia_included": str(bool(nhia_csv and nhia_csv.exists())).lower(),
        }
        connection.executemany("INSERT INTO metadata (key, value) VALUES (?, ?)", metadata.items())
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError(f"Catalog integrity check failed: {integrity}")
    except BaseException:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()
        temporary_path.replace(destination)

    return CatalogBuildReport(
        product_count=product_count,
        ingredient_count=ingredient_count,
        appearance_count=appearance_count,
        nhi_code_count=nhi_code_count,
        gtin_count=gtin_count,
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE products (
            permit_number TEXT PRIMARY KEY,
            normalized_permit TEXT NOT NULL UNIQUE,
            serial_number TEXT,
            name_zh TEXT,
            name_en TEXT,
            normalized_name_zh TEXT NOT NULL,
            normalized_name_en TEXT NOT NULL,
            dosage_form TEXT,
            manufacturer TEXT,
            normalized_manufacturer TEXT NOT NULL,
            applicant TEXT,
            indications TEXT,
            package_description TEXT
        );
        CREATE INDEX products_serial_idx ON products(serial_number);
        CREATE INDEX products_name_zh_idx ON products(normalized_name_zh);
        CREATE INDEX products_name_en_idx ON products(normalized_name_en);

        CREATE TABLE ingredients (
            id INTEGER PRIMARY KEY,
            permit_number TEXT NOT NULL REFERENCES products(permit_number) ON DELETE CASCADE,
            official_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            ingredient_code TEXT,
            prescription_label TEXT,
            amount_description TEXT,
            amount TEXT,
            unit TEXT
        );
        CREATE INDEX ingredients_permit_idx ON ingredients(permit_number);
        CREATE INDEX ingredients_name_idx ON ingredients(normalized_name);

        CREATE TABLE appearances (
            permit_number TEXT PRIMARY KEY REFERENCES products(permit_number) ON DELETE CASCADE,
            shapes_json TEXT NOT NULL,
            colors_json TEXT NOT NULL,
            score_marks_json TEXT NOT NULL,
            imprints_json TEXT NOT NULL,
            image_urls_json TEXT NOT NULL
        );
        CREATE TABLE imprint_index (
            normalized_imprint TEXT NOT NULL,
            imprint TEXT NOT NULL,
            permit_number TEXT NOT NULL REFERENCES products(permit_number) ON DELETE CASCADE,
            PRIMARY KEY (normalized_imprint, permit_number)
        );
        CREATE INDEX imprint_lookup_idx ON imprint_index(normalized_imprint);

        CREATE TABLE product_codes (
            permit_number TEXT NOT NULL REFERENCES products(permit_number) ON DELETE CASCADE,
            code_type TEXT NOT NULL CHECK (code_type IN ('nhi', 'atc', 'gtin')),
            code TEXT NOT NULL,
            PRIMARY KEY (permit_number, code_type, code)
        );
        CREATE INDEX product_codes_lookup_idx ON product_codes(code_type, code);

        CREATE TABLE source_urls (
            permit_number TEXT NOT NULL REFERENCES products(permit_number) ON DELETE CASCADE,
            url TEXT NOT NULL,
            PRIMARY KEY (permit_number, url)
        );
        CREATE TABLE search_terms (
            normalized_term TEXT NOT NULL,
            display_term TEXT NOT NULL,
            permit_number TEXT NOT NULL REFERENCES products(permit_number) ON DELETE CASCADE,
            PRIMARY KEY (normalized_term, permit_number)
        );
        CREATE INDEX search_terms_lookup_idx ON search_terms(normalized_term);
        """
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, expect_zip: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(  # noqa: S310 - callers only pass fixed HTTPS source URLs
        url, headers={"User-Agent": "PillScan/0.1"}
    )
    with (
        urllib.request.urlopen(request, timeout=180) as response,  # noqa: S310
        tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary,
    ):
        shutil.copyfileobj(response, temporary)
        temporary_path = Path(temporary.name)
    try:
        if temporary_path.stat().st_size == 0:
            raise ValueError(f"Empty response from {url}")
        if expect_zip and not zipfile.is_zipfile(temporary_path):
            raise ValueError(f"Expected a ZIP response from {url}")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def sync_catalog(
    *,
    raw_dir: Path,
    nhia_csv: Path,
    catalog_path: Path,
    force: bool,
    skip_download: bool,
) -> CatalogBuildReport | None:
    if catalog_path.exists() and not force:
        return None
    if not skip_download:
        for name, dataset_id in TFDA_DATASETS.items():
            destination = raw_dir / f"{name}.json.zip"
            if force or not destination.exists():
                _download(
                    TFDA_BASE_URL.format(dataset_id=dataset_id),
                    destination,
                    expect_zip=True,
                )
        if force or not nhia_csv.exists():
            _download(NHIA_DATASET_URL, nhia_csv, expect_zip=False)
    return build_catalog(raw_dir, catalog_path, nhia_csv=nhia_csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TFDA/NHIA open data and build the local PillScan SQLite catalog"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path(".data/tfda/raw"))
    parser.add_argument("--nhia-csv", type=Path, default=Path(".data/nhia/drugs.csv"))
    parser.add_argument("--catalog-path", type=Path, default=Path(".data/tfda/catalog.sqlite3"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = sync_catalog(
        raw_dir=args.raw_dir,
        nhia_csv=args.nhia_csv,
        catalog_path=args.catalog_path,
        force=args.force,
        skip_download=args.skip_download,
    )
    if report is None:
        print(f"catalog already exists: {args.catalog_path}")
        return
    print(
        f"catalog ready: {args.catalog_path} "
        f"({report.product_count} products, {report.ingredient_count} ingredients, "
        f"{report.appearance_count} appearances, {report.nhi_code_count} NHI codes, "
        f"{report.gtin_count} GTINs)"
    )


if __name__ == "__main__":
    main()
