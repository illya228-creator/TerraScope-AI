import asyncio


def test_gradio_runtime_configuration_allows_blocks_construction():
    """The compatibility shim must not create a Windows event loop eagerly."""
    import gradio as gr

    from app.gradio_compat import configure_gradio_runtime

    configure_gradio_runtime()
    demo = gr.Blocks(analytics_enabled=False)

    assert isinstance(demo._queue.pending_message_lock, asyncio.Lock)
    assert isinstance(demo._queue.delete_lock, asyncio.Lock)
