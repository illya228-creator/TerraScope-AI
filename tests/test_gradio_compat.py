import os

from httpx._utils import get_environment_proxies

from app.gradio_compat import configure_localhost_proxy_bypass


def test_localhost_proxy_bypass_is_configured(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)

    configure_localhost_proxy_bypass()

    configured_hosts = set(os.environ["NO_PROXY"].split(","))
    assert {"127.0.0.1", "localhost", "::1"} <= configured_hosts
    assert os.environ["no_proxy"] == os.environ["NO_PROXY"]
    proxies = get_environment_proxies()
    assert proxies["all://127.0.0.1"] is None
    assert proxies["all://localhost"] is None
