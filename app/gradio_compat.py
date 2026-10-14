
"""Runtime configuration for the local Gradio process."""

from __future__ import annotations

import os

LOCALHOST_BYPASS = ("127.0.0.1", "localhost", "::1")


def configure_localhost_proxy_bypass() -> None:
    """Ensure Gradio's self-check reaches its local server directly.

    HTTPX reads the Windows proxy setting but does not import its registry
    exceptions. Gradio uses HTTPX to request ``/gradio_api/startup-events``
    during launch, so localhost must be listed in the standard environment
    bypass variable before Gradio is imported.
    """
    configured_hosts = {
        host.strip().lower()
        for host in os.environ.get("NO_PROXY", "").split(",")
        if host.strip()
    }
    configured_hosts.update(LOCALHOST_BYPASS)
    value = ",".join(sorted(configured_hosts))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value
