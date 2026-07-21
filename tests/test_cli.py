from __future__ import annotations

import subprocess

from actant import cli


def test_server_start_runs_attached_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    recorded: dict[str, object] = {}

    def fake_run(args, *, env=None, **kwargs):  # type: ignore[no-untyped-def]
        recorded["args"] = list(args)
        return 0

    monkeypatch.setattr(cli, "_run_compose", fake_run)

    assert cli.main(["server", "start"]) == 0
    assert recorded["args"] == [
        "up",
        "--remove-orphans",
        "temporal-postgres",
        "temporal",
        "temporal-ui",
    ]


def test_server_start_detached_forwards_ports(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    recorded: dict[str, object] = {}

    def fake_run(args, *, env=None, **kwargs):  # type: ignore[no-untyped-def]
        recorded["args"] = args
        recorded["env"] = env
        recorded["options"] = kwargs
        return 0

    monkeypatch.setattr(cli, "_run_compose", fake_run)

    result = cli.main(["server", "start", "--detach", "--port", "17233", "--ui-port", "18233"])

    assert result == 0
    assert recorded == {
        "args": [
            "up",
            "--detach",
            "--wait",
            "--wait-timeout",
            "60",
            "--remove-orphans",
            "temporal-postgres",
            "temporal",
            "temporal-ui",
        ],
        "env": {
            "ACTANT_TEMPORAL_PORT": "17233",
            "ACTANT_TEMPORAL_UI_PORT": "18233",
        },
        "options": {
            "compose_file": None,
            "project_name": "actant-local",
            "compose_command": "docker compose",
        },
    }
    assert capsys.readouterr().out == (
        "Temporal: localhost:17233\nTemporal UI: http://localhost:18233\n"
    )


def test_server_lifecycle_commands(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli,
        "_run_compose",
        lambda args, **kwargs: calls.append(list(args)) or 0,
    )

    assert cli.main(["server", "status"]) == 0
    assert cli.main(["server", "logs", "--follow"]) == 0
    assert cli.main(["server", "stop"]) == 0
    assert cli.main(["server", "reset"]) == 0

    assert calls == [
        ["ps"],
        ["logs", "--follow", "temporal"],
        ["down"],
        ["down", "--volumes"],
    ]


def test_missing_docker_has_actionable_error(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    def missing(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing)

    assert cli._run_compose(["ps"]) == 127
    assert "requires Docker" in capsys.readouterr().err


def test_packaged_compose_file_exists() -> None:
    assert cli._compose_file().is_file()


def test_server_overrides_and_no_ui(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    recorded: dict[str, object] = {}

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        recorded["args"] = list(args)
        recorded.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_run_compose", fake_run)

    result = cli.main(
        [
            "server",
            "start",
            "--compose-file",
            "/tmp/custom.yml",
            "--project-name",
            "custom-project",
            "--compose-command",
            "podman compose",
            "--no-ui",
        ]
    )

    assert result == 0
    assert recorded["args"] == [
        "up",
        "--remove-orphans",
        "temporal-postgres",
        "temporal",
    ]
    assert recorded["compose_file"] == "/tmp/custom.yml"
    assert recorded["project_name"] == "custom-project"
    assert recorded["compose_command"] == "podman compose"
    assert "Temporal UI" not in capsys.readouterr().out
