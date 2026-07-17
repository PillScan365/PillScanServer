from __future__ import annotations

import argparse
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pillscan_server.catalog_sync import CatalogBuildReport, sync_catalog

ENV_ASSIGNMENT = re.compile(r"^(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$")


@dataclass(frozen=True, slots=True)
class CatalogHealth:
    ready: bool
    catalog_version: str | None = None
    product_count: int | None = None
    reason: str | None = None


def initialize_local_env(env_file: Path, example_file: Path) -> bool:
    """Create a private local env file without replacing an existing deployment config."""
    if env_file.exists():
        return False
    content = example_file.read_text(encoding="utf-8")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        target.write(content)
    return True


def env_var_is_configured(name: str, env_files: tuple[Path, ...]) -> bool:
    """Check configuration presence without logging or returning a secret value."""
    if os.environ.get(name, "").strip():
        return True
    for path in env_files:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as source:
            for raw_line in source:
                match = ENV_ASSIGNMENT.match(raw_line.strip())
                if match is None or match.group("name") != name:
                    continue
                value = match.group("value").strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1].strip()
                if value:
                    return True
    return False


def inspect_catalog(path: Path) -> CatalogHealth:
    if not path.is_file():
        return CatalogHealth(ready=False, reason="catalog file is missing")
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
            product_count = connection.execute("SELECT COUNT(*) FROM products").fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return CatalogHealth(ready=False, reason="catalog is not a valid PillScan database")
    if integrity is None or integrity[0] != "ok":
        return CatalogHealth(ready=False, reason="catalog integrity check failed")
    if product_count is None or int(product_count[0]) <= 0:
        return CatalogHealth(ready=False, reason="catalog contains no products")
    return CatalogHealth(
        ready=True,
        catalog_version=metadata.get("catalog_version"),
        product_count=int(product_count[0]),
    )


def _print_catalog_result(report: CatalogBuildReport | None, path: Path) -> None:
    if report is None:
        print(f"[ok] Reusing catalog: {path}")
        return
    print(
        f"[ok] Built catalog: {report.product_count:,} products, "
        f"{report.ingredient_count:,} ingredients, {report.appearance_count:,} appearances, "
        f"{report.nhi_code_count:,} NHI codes"
    )


def run_setup(
    *,
    env_file: Path,
    env_example: Path,
    raw_dir: Path,
    nhia_csv: Path,
    catalog_path: Path,
    force_data: bool,
    skip_data: bool,
    check_only: bool,
) -> int:
    if not check_only:
        created = initialize_local_env(env_file, env_example)
        print(
            f"[ok] Created private config: {env_file}"
            if created
            else f"[ok] Reusing local config: {env_file}"
        )
        if not skip_data:
            print("[info] Syncing TFDA and NHIA open data...")
            report = sync_catalog(
                raw_dir=raw_dir,
                nhia_csv=nhia_csv,
                catalog_path=catalog_path,
                force=force_data,
                skip_download=False,
            )
            _print_catalog_result(report, catalog_path)

    problems: list[str] = []
    if env_var_is_configured("OPENAI_API_KEY", (Path(".env"), env_file)):
        print("[ok] OPENAI_API_KEY is configured")
    else:
        problems.append(f"set OPENAI_API_KEY in {env_file} or the process environment")

    health = inspect_catalog(catalog_path)
    if health.ready:
        version = health.catalog_version or "unknown"
        print(f"[ok] Catalog is healthy: version {version}, {health.product_count or 0:,} products")
    else:
        problems.append(f"catalog is not ready: {health.reason}")

    if problems:
        for problem in problems:
            print(f"[missing] {problem}")
        print("[action] Fix the missing item(s), then run: pixi run doctor")
        return 2
    print("[ready] PillScan is configured. Start it with: pixi run serve")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and verify a complete local PillScan deployment"
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--env-example", type=Path, default=Path(".env.example"))
    parser.add_argument("--raw-dir", type=Path, default=Path(".data/tfda/raw"))
    parser.add_argument("--nhia-csv", type=Path, default=Path(".data/nhia/drugs.csv"))
    parser.add_argument("--catalog-path", type=Path, default=Path(".data/tfda/catalog.sqlite3"))
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(
        run_setup(
            env_file=args.env_file,
            env_example=args.env_example,
            raw_dir=args.raw_dir,
            nhia_csv=args.nhia_csv,
            catalog_path=args.catalog_path,
            force_data=args.force_data,
            skip_data=args.skip_data,
            check_only=args.check_only,
        )
    )


if __name__ == "__main__":
    main()
