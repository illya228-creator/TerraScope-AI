
"""Small, version-scoped workarounds for the local Gradio runtime."""

from __future__ import annotations

import asyncio

import gradio as gr


def configure_gradio_runtime() -> None:
    """Avoid Gradio creating an unused event loop during app construction.

    Gradio 6.30 creates a new event loop just to construct locks when no loop is
    running.  In the local Python 3.14 installation that reaches a hanging
    ``socket.socketpair()`` fallback on Windows.  ``asyncio`` primitives are
    safely created without a loop and bind lazily when the server runs.

    This patches only Gradio's two internal aliases; it does not alter the
    application event loop or inference behaviour.
    """
    import gradio.queueing as queueing
    import gradio.utils as utils

    queueing.safe_get_lock = asyncio.Lock
    utils.safe_get_lock = asyncio.Lock
    utils.safe_get_stop_event = asyncio.Event


def apply_gradio_compatibility(demo):
    """
    Compatibility workaround for Gradio versions whose theme.css
    endpoint expects the Blocks.theme_css attribute.
    """
    if not hasattr(demo, "theme_css"):
        demo.theme_css = ""

    return demo
