# vllm-proxy

A thin vLLM wrapper that exposes a minimal OpenAI-compatible Responses API surface.

## Run from GitHub with uvx

This project is intended to run directly from the public GitHub repository,
without publishing to PyPI.

```bash
uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy
```

By default, the proxy listens on `127.0.0.1:8080` and forwards requests to
`http://127.0.0.1:8000/v1`.

You can configure the bind address and port with CLI options.

```bash
uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy \
  --host 0.0.0.0 \
  --port 8080
```

Message logging is disabled by default. To enable it, pass a JSONL file path
with `--message-log-file`.

```bash
uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy \
  --message-log-file ~/.local/state/vllm-proxy/messages.jsonl
```

Common log locations depend on the runtime environment.

- Local development: a temporary directory outside the repository, or `./logs/`
- Per-user CLI usage: `~/.local/state/vllm-proxy/` on Linux, or `~/Library/Logs/vllm-proxy/` on macOS
- Service deployment: stdout/stderr collected by journald or a logging backend, or `/var/log/vllm-proxy/`

Because message logs can contain prompts and responses, avoid writing them into
shared repositories.

## Configuration

The server can also be configured with environment variables.

| Variable | Default |
| --- | --- |
| `HOST` | `127.0.0.1` |
| `PORT` | `8080` |
| `RELOAD` | unset |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `VLLM_API_KEY` | `dummy` |
| `VLLM_MODEL` | `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` |
| `SYSTEM_LANGUAGE_INSTRUCTION` | `特に指定がない限り、日本語で簡潔に応答してください。` |
| `MESSAGE_LOG_FILE` | unset |

Example:

```bash
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
MESSAGE_LOG_FILE=~/.local/state/vllm-proxy/messages.jsonl \
  uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy
```

## Development

```bash
uv sync
make start
```
