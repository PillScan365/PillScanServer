import os
import sqlite3
from pathlib import Path

from pillscan_server import setup
from pillscan_server.catalog_sync import CatalogBuildReport


def make_healthy_catalog(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE products (permit_number TEXT PRIMARY KEY);
        INSERT INTO metadata VALUES ('catalog_version', 'test-version');
        INSERT INTO products VALUES ('TEST-001');
        """
    )
    connection.commit()
    connection.close()


def test_initialize_local_env_creates_private_file_once(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    target = tmp_path / ".env.local"
    example.write_text("OPENAI_API_KEY=\n", encoding="utf-8")

    assert setup.initialize_local_env(target, example) is True
    assert setup.initialize_local_env(target, example) is False
    assert target.read_text(encoding="utf-8") == "OPENAI_API_KEY=\n"
    assert target.stat().st_mode & 0o777 == 0o600


def test_env_var_check_accepts_process_or_nonempty_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    env_file = tmp_path / ".env.local"
    env_file.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert setup.env_var_is_configured("OPENAI_API_KEY", (env_file,)) is False

    env_file.write_text('export OPENAI_API_KEY="configured"\n', encoding="utf-8")
    assert setup.env_var_is_configured("OPENAI_API_KEY", (env_file,)) is True

    env_file.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    assert setup.env_var_is_configured("OPENAI_API_KEY", (env_file,)) is True


def test_inspect_catalog_reports_missing_invalid_empty_and_healthy(tmp_path: Path) -> None:
    missing = setup.inspect_catalog(tmp_path / "missing.sqlite3")
    assert missing.ready is False
    assert missing.reason == "catalog file is missing"

    invalid_path = tmp_path / "invalid.sqlite3"
    invalid_path.write_text("not sqlite", encoding="utf-8")
    assert setup.inspect_catalog(invalid_path).ready is False

    empty_path = tmp_path / "empty.sqlite3"
    connection = sqlite3.connect(empty_path)
    connection.executescript(
        "CREATE TABLE metadata (key TEXT, value TEXT); CREATE TABLE products (id TEXT);"
    )
    connection.commit()
    connection.close()
    assert setup.inspect_catalog(empty_path).reason == "catalog contains no products"

    healthy_path = tmp_path / "catalog.sqlite3"
    make_healthy_catalog(healthy_path)
    health = setup.inspect_catalog(healthy_path)
    assert health.ready is True
    assert health.catalog_version == "test-version"
    assert health.product_count == 1


def test_run_setup_initializes_data_and_reports_ready(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    example = tmp_path / ".env.example"
    env_file = tmp_path / ".env.local"
    catalog = tmp_path / "catalog.sqlite3"
    example.write_text("OPENAI_API_KEY=local-test-key\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_sync(**kwargs: object) -> CatalogBuildReport:
        calls.append(kwargs)
        make_healthy_catalog(catalog)
        return CatalogBuildReport(1, 2, 3, 4, 0)

    monkeypatch.setattr(setup, "sync_catalog", fake_sync)
    result = setup.run_setup(
        env_file=env_file,
        env_example=example,
        raw_dir=tmp_path / "raw",
        nhia_csv=tmp_path / "nhia.csv",
        catalog_path=catalog,
        force_data=True,
        skip_data=False,
        check_only=False,
    )

    assert result == 0
    assert calls[0]["force"] is True
    output = capsys.readouterr().out
    assert "[ready]" in output
    assert "local-test-key" not in output


def test_check_only_reports_every_missing_requirement(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = setup.run_setup(
        env_file=tmp_path / ".env.local",
        env_example=tmp_path / ".env.example",
        raw_dir=tmp_path / "raw",
        nhia_csv=tmp_path / "nhia.csv",
        catalog_path=tmp_path / "catalog.sqlite3",
        force_data=False,
        skip_data=False,
        check_only=True,
    )

    assert result == 2
    output = capsys.readouterr().out
    assert "OPENAI_API_KEY" in output
    assert "catalog file is missing" in output


def test_parser_defaults_and_main_exit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    args = setup.build_parser().parse_args(["--check-only"])
    assert args.check_only is True

    monkeypatch.setattr(setup, "run_setup", lambda **kwargs: 0)
    monkeypatch.setattr("sys.argv", ["pillscan-setup", "--check-only"])
    try:
        setup.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("setup.main() must terminate with the setup status")


def test_env_assignment_pattern_rejects_comments() -> None:
    assert setup.ENV_ASSIGNMENT.match("# OPENAI_API_KEY=hidden") is None
    assert os.environ.get("THIS_VARIABLE_SHOULD_NOT_EXIST_FOR_PILLSCAN") is None
