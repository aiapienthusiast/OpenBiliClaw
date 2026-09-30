"""Shared runtime helpers for the dedicated recommendation API process.

On POSIX systems the recommendation process is exposed through a Unix domain
socket.  Windows asyncio/uvicorn does not implement ``create_unix_server``, so
on Windows the same process listens on a loopback TCP port instead.  POSIX also
falls back to loopback TCP when the socket path derived from the data
directory would exceed the platform ``sun_path`` limit (about 104 bytes on
macOS), because binding such a path crashes the child with ``OSError:
AF_UNIX path too long``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RECOMMENDATION_PORT = 8423
RECOMMENDATION_SOCK_ENV = "OPENBILICLAW_RECOMMENDATION_SOCK"
RECOMMENDATION_PORT_ENV = "OPENBILICLAW_RECOMMENDATION_PORT"

# Size of ``sockaddr_un.sun_path`` on the strictest supported POSIX platform
# (macOS/BSD: 104 bytes including the NUL terminator; Linux allows 108).  A
# path is only bindable when its encoded form plus the terminator fits.
UNIX_SOCKET_SUN_PATH_BYTES = 104


def recommendation_sock_from_data_path(data_path: Path) -> str:
    """Return the default Unix socket path for a data directory."""
    return str(Path(data_path) / "runtime" / "recommendation.sock")


def unix_socket_path_too_long(path: str) -> bool:
    """Whether a Unix socket path exceeds the portable ``sun_path`` limit."""
    return len(os.fsencode(path)) >= UNIX_SOCKET_SUN_PATH_BYTES


def ensure_recommendation_transport_env(data_path: Path) -> str:
    """Set the platform-appropriate recommendation transport environment.

    Returns a human-readable transport description for status/log output.
    """
    if os.name == "nt" or os.environ.get(RECOMMENDATION_PORT_ENV, "").strip():
        os.environ.setdefault(RECOMMENDATION_PORT_ENV, str(DEFAULT_RECOMMENDATION_PORT))
        os.environ.pop(RECOMMENDATION_SOCK_ENV, None)
        port = os.environ[RECOMMENDATION_PORT_ENV]
        return f"TCP 127.0.0.1:{port}"
    sock = os.environ.get(RECOMMENDATION_SOCK_ENV) or recommendation_sock_from_data_path(data_path)
    if unix_socket_path_too_long(sock):
        os.environ.setdefault(RECOMMENDATION_PORT_ENV, str(DEFAULT_RECOMMENDATION_PORT))
        os.environ.pop(RECOMMENDATION_SOCK_ENV, None)
        port = os.environ[RECOMMENDATION_PORT_ENV]
        logger.warning(
            "Recommendation Unix socket path %r is %d bytes, exceeding the "
            "%d-byte AF_UNIX sun_path limit; falling back to TCP 127.0.0.1:%s",
            sock,
            len(os.fsencode(sock)),
            UNIX_SOCKET_SUN_PATH_BYTES,
            port,
        )
        return f"TCP 127.0.0.1:{port}"
    os.environ[RECOMMENDATION_SOCK_ENV] = sock
    os.environ.pop(RECOMMENDATION_PORT_ENV, None)
    return f"Unix socket {os.environ[RECOMMENDATION_SOCK_ENV]}"


def recommendation_transport_enabled() -> bool:
    """Whether the main API is expected to proxy to a recommendation process."""
    return bool(
        os.environ.get(RECOMMENDATION_SOCK_ENV, "").strip()
        or os.environ.get(RECOMMENDATION_PORT_ENV, "").strip()
    )
