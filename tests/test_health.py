from fastapi.testclient import TestClient

from vllm_proxy.server import app, log_messages


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_message_log_is_disabled_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MESSAGE_LOG_FILE", raising=False)

    log_messages("input", [{"role": "user", "content": "hello"}])

    assert not (tmp_path / "vllm-proxy.jsonl").exists()


def test_message_log_can_be_enabled_with_env(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "messages.jsonl"
    monkeypatch.setenv("MESSAGE_LOG_FILE", str(log_file))

    log_messages("input", [{"role": "user", "content": "hello"}])

    assert log_file.read_text(encoding="utf-8") == (
        '{"direction": "input", "message": {"role": "user", "content": "hello"}}\n'
    )
