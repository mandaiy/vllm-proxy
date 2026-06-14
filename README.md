# vllm-proxy

vLLM に対して、最小限の OpenAI 互換 Responses API を提供する薄いプロキシです。

## GitHub から uvx で実行する

このプロジェクトは PyPI には公開せず、public GitHub repository から直接実行する想定です。

```bash
uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy
```

デフォルトでは `127.0.0.1:8080` で listen し、リクエストを
`http://127.0.0.1:8000/v1` へ転送します。

listen するアドレスとポートは CLI オプションでも指定できます。

```bash
uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy \
  --host 0.0.0.0 \
  --port 8080
```

メッセージログはデフォルトでは出力しません。出力したい場合は
`--message-log-file` で JSONL ファイルの保存先を指定します。

```bash
uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy \
  --message-log-file ~/.local/state/vllm-proxy/messages.jsonl
```

ログファイルの一般的な保存先は実行環境によって異なります。

- ローカルの開発用途: プロジェクト外の一時ディレクトリや `./logs/`
- ユーザー単位の CLI 実行: Linux では `~/.local/state/vllm-proxy/`、macOS では `~/Library/Logs/vllm-proxy/`
- systemd などのサービス運用: stdout/stderr に流して journald やログ基盤で収集、または `/var/log/vllm-proxy/`

プロンプトや応答を含むため、共有リポジトリ内への保存は避けるのが無難です。

## 設定

サーバー設定は環境変数でも指定できます。

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

実行例:

```bash
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
MESSAGE_LOG_FILE=~/.local/state/vllm-proxy/messages.jsonl \
  uvx --from git+https://github.com/mandaiy/vllm-proxy.git vllm-proxy
```

## 開発

```bash
uv sync
make start
```
