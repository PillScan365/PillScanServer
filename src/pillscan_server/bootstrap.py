from pillscan_server.catalog_sync import sync_catalog
from pillscan_server.cli import main as serve
from pillscan_server.config import get_settings


def main() -> None:
    settings = get_settings()
    sync_catalog(
        raw_dir=settings.tfda_raw_dir,
        nhia_csv=settings.nhia_drug_csv_path,
        catalog_path=settings.tfda_catalog_path,
        force=False,
        skip_download=False,
    )
    serve()


if __name__ == "__main__":
    main()
