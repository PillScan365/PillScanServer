from pathlib import Path
from types import SimpleNamespace

from pillscan_server import bootstrap


def test_bootstrap_syncs_catalog_before_starting_server(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, object]] = []
    settings = SimpleNamespace(
        tfda_raw_dir=Path("raw"),
        nhia_drug_csv_path=Path("nhia.csv"),
        tfda_catalog_path=Path("catalog.sqlite3"),
    )

    monkeypatch.setattr(bootstrap, "get_settings", lambda: settings)
    monkeypatch.setattr(
        bootstrap,
        "sync_catalog",
        lambda **kwargs: calls.append(("sync", kwargs)),
    )
    monkeypatch.setattr(bootstrap, "serve", lambda: calls.append(("serve", None)))

    bootstrap.main()

    assert calls[0][0] == "sync"
    assert calls[1] == ("serve", None)
