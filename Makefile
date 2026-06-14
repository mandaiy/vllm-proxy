.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; BEGIN {print "\nUsage: make [command]\n"} {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: start
start: ## Starts server (uvicorn vllm_proxy.server:app --host 0.0.0.0 --port 8000)
	uv run uvicorn vllm_proxy.server:app --host 0.0.0.0 --port 8000

.PHONY: format
format:
	@uv run ruff format
	@uv run ruff check --fix

.PHONY: lint
lint:
	@uv run ruff format --check
	@uv run ruff check

.PHONY: type-check
type-check:
	@uv run pyright src/

.PHONY: test
test:
	@uv run pytest
