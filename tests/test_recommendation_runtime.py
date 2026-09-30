"""Tests for the cross-platform recommendation process transport."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from openbiliclaw import recommendation_runtime
from openbiliclaw.recommendation_runtime import (
    DEFAULT_RECOMMENDATION_PORT,
    RECOMMENDATION_PORT_ENV,
    RECOMMENDATION_SOCK_ENV,
    UNIX_SOCKET_SUN_PATH_BYTES,
    ensure_recommendation_transport_env,
    recommendation_sock_from_data_path,
    recommendation_transport_enabled,
    unix_socket_path_too_long,
)

if TYPE_CHECKING:
    from pathlib import Path


def _deep_data_path(tmp_path: Path) -> Path:
    """Return a data path whose derived Unix socket path exceeds sun_path."""
    candidate = tmp_path
    while not unix_socket_path_too_long(recommendation_sock_from_data_path(candidate)):
        candidate = candidate / "deeper-data-directory"
    return candidate


def test_recommendation_sock_from_data_path(tmp_path: Path) -> None:
    assert recommendation_sock_from_data_path(tmp_path) == str(
        tmp_path / "runtime" / "recommendation.sock"
    )


def test_ensure_recommendation_transport_env_posix(monkeypatch, tmp_path: Path) -> None:
    # macOS pytest tmp_path is itself deep enough to exceed sun_path; use a
    # short path so this test exercises the Unix-socket branch.
    sock = "/tmp/obc-test-recommendation.sock"
    monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, sock)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    monkeypatch.setattr(recommendation_runtime.os, "name", "posix")

    description = ensure_recommendation_transport_env(tmp_path)

    assert description == f"Unix socket {sock}"
    assert RECOMMENDATION_SOCK_ENV in recommendation_runtime.os.environ
    assert RECOMMENDATION_PORT_ENV not in recommendation_runtime.os.environ
    assert recommendation_transport_enabled() is True


def test_ensure_recommendation_transport_env_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.setenv(RECOMMENDATION_PORT_ENV, str(DEFAULT_RECOMMENDATION_PORT))
    monkeypatch.setattr(recommendation_runtime.os, "name", "nt")

    description = ensure_recommendation_transport_env(tmp_path)

    assert description == f"TCP 127.0.0.1:{DEFAULT_RECOMMENDATION_PORT}"
    assert RECOMMENDATION_PORT_ENV in recommendation_runtime.os.environ
    assert RECOMMENDATION_SOCK_ENV not in recommendation_runtime.os.environ
    assert recommendation_transport_enabled() is True


def test_recommendation_transport_enabled_false_without_env(monkeypatch) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)

    assert recommendation_transport_enabled() is False


def test_ensure_recommendation_transport_env_posix_with_explicit_port(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.setenv(RECOMMENDATION_PORT_ENV, "18424")
    monkeypatch.setattr(recommendation_runtime.os, "name", "posix")

    description = ensure_recommendation_transport_env(tmp_path)

    assert description == "TCP 127.0.0.1:18424"
    assert RECOMMENDATION_PORT_ENV in recommendation_runtime.os.environ
    assert RECOMMENDATION_SOCK_ENV not in recommendation_runtime.os.environ


def test_recommendation_server_windows_binds_tcp(monkeypatch) -> None:
    import openbiliclaw.recommendation_server as recommendation_server

    calls: list[dict[str, object]] = []
    fake_app = object()
    # Transport tests must not depend on the developer machine's on-disk LLM
    # config; the readiness gate itself is covered by test_worker_degraded_boot.
    monkeypatch.setattr(recommendation_server, "wait_for_buildable_llm", lambda: None)
    monkeypatch.setattr(recommendation_server.os, "name", "nt")
    monkeypatch.setattr(recommendation_server, "create_app", lambda: fake_app)
    monkeypatch.setattr(
        recommendation_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append({"app": args[0], **kwargs}),
    )
    monkeypatch.setenv(RECOMMENDATION_PORT_ENV, str(DEFAULT_RECOMMENDATION_PORT))

    recommendation_server.main()

    assert calls == [
        {
            "app": fake_app,
            "host": "127.0.0.1",
            "port": DEFAULT_RECOMMENDATION_PORT,
            "log_level": "info",
        }
    ]


@pytest.mark.skipif(recommendation_runtime.os.name == "nt", reason="POSIX Unix socket path")
def test_recommendation_server_posix_binds_unix_socket(monkeypatch, tmp_path: Path) -> None:
    import openbiliclaw.recommendation_server as recommendation_server

    calls: list[dict[str, object]] = []
    fake_app = object()
    sock = recommendation_sock_from_data_path(tmp_path)
    # Same as the Windows transport test: binding behavior is independent of
    # the on-disk LLM config readiness gate.
    monkeypatch.setattr(recommendation_server, "wait_for_buildable_llm", lambda: None)
    monkeypatch.setattr(recommendation_server, "create_app", lambda: fake_app)
    monkeypatch.setattr(
        recommendation_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append({"app": args[0], **kwargs}),
    )
    monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, sock)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)

    recommendation_server.main()

    assert calls == [{"app": fake_app, "uds": sock, "log_level": "info"}]


def test_unix_socket_path_too_long_boundary() -> None:
    fit = "a" * (UNIX_SOCKET_SUN_PATH_BYTES - 1)
    assert unix_socket_path_too_long(fit) is False
    assert unix_socket_path_too_long(fit + "a") is True


def test_ensure_recommendation_transport_env_posix_long_path_falls_back_to_tcp(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    monkeypatch.setattr(recommendation_runtime.os, "name", "posix")
    deep_data_path = _deep_data_path(tmp_path)
    sock = recommendation_sock_from_data_path(deep_data_path)

    with caplog.at_level(logging.WARNING, logger="openbiliclaw.recommendation_runtime"):
        description = ensure_recommendation_transport_env(deep_data_path)

    assert description == f"TCP 127.0.0.1:{DEFAULT_RECOMMENDATION_PORT}"
    assert recommendation_runtime.os.environ[RECOMMENDATION_PORT_ENV] == str(
        DEFAULT_RECOMMENDATION_PORT
    )
    assert RECOMMENDATION_SOCK_ENV not in recommendation_runtime.os.environ
    assert recommendation_transport_enabled() is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert str(len(sock.encode())) in message
    assert str(UNIX_SOCKET_SUN_PATH_BYTES) in message
    assert str(DEFAULT_RECOMMENDATION_PORT) in message


def test_ensure_recommendation_transport_env_posix_long_explicit_sock_falls_back_to_tcp(
    monkeypatch, tmp_path: Path
) -> None:
    long_sock = recommendation_sock_from_data_path(_deep_data_path(tmp_path))
    monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, long_sock)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    monkeypatch.setattr(recommendation_runtime.os, "name", "posix")

    description = ensure_recommendation_transport_env(tmp_path)

    assert description == f"TCP 127.0.0.1:{DEFAULT_RECOMMENDATION_PORT}"
    assert RECOMMENDATION_SOCK_ENV not in recommendation_runtime.os.environ
    assert recommendation_runtime.os.environ[RECOMMENDATION_PORT_ENV] == str(
        DEFAULT_RECOMMENDATION_PORT
    )


@pytest.mark.skipif(recommendation_runtime.os.name == "nt", reason="POSIX Unix socket path")
def test_recommendation_server_and_proxy_share_tcp_fallback_on_long_path(
    monkeypatch, tmp_path: Path
) -> None:
    """ensure_* picks TCP for a deep data dir and the child honors the same env."""
    import openbiliclaw.recommendation_server as recommendation_server

    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    deep_data_path = _deep_data_path(tmp_path)

    assert ensure_recommendation_transport_env(deep_data_path) == (
        f"TCP 127.0.0.1:{DEFAULT_RECOMMENDATION_PORT}"
    )

    calls: list[dict[str, object]] = []
    fake_app = object()
    monkeypatch.setattr(recommendation_server, "wait_for_buildable_llm", lambda: None)
    monkeypatch.setattr(recommendation_server, "create_app", lambda: fake_app)
    monkeypatch.setattr(
        recommendation_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append({"app": args[0], **kwargs}),
    )

    recommendation_server.main()

    assert calls == [
        {
            "app": fake_app,
            "host": "127.0.0.1",
            "port": DEFAULT_RECOMMENDATION_PORT,
            "log_level": "info",
        }
    ]


@pytest.mark.skipif(recommendation_runtime.os.name == "nt", reason="POSIX Unix socket path")
def test_recommendation_server_standalone_long_path_binds_tcp(monkeypatch, tmp_path: Path) -> None:
    """Without inherited env the server derives the path itself and must not crash."""
    import openbiliclaw.recommendation_server as recommendation_server

    calls: list[dict[str, object]] = []
    fake_app = object()
    deep_data_path = _deep_data_path(tmp_path)
    monkeypatch.setattr(recommendation_server, "wait_for_buildable_llm", lambda: None)
    monkeypatch.setattr(recommendation_server, "create_app", lambda: fake_app)
    monkeypatch.setattr(
        recommendation_server,
        "load_config",
        lambda: SimpleNamespace(data_path=deep_data_path),
    )
    monkeypatch.setattr(
        recommendation_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append({"app": args[0], **kwargs}),
    )
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)

    recommendation_server.main()

    assert calls == [
        {
            "app": fake_app,
            "host": "127.0.0.1",
            "port": DEFAULT_RECOMMENDATION_PORT,
            "log_level": "info",
        }
    ]
