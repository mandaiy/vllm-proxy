import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
DEFAULT_MODEL = os.environ.get(
    "VLLM_MODEL",
    "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ",
)
SYSTEM_LANGUAGE_INSTRUCTION = os.environ.get(
    "SYSTEM_LANGUAGE_INSTRUCTION",
    "特に指定がない限り、日本語で簡潔に応答してください。",
)
MESSAGE_LOG_FILE = os.environ.get("MESSAGE_LOG_FILE", "vllm-proxy.jsonl")

app = FastAPI()


def write_message_log(entry: dict[str, Any]) -> bool:
    try:
        with open(MESSAGE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"failed to write message log: {exc}", file=sys.stderr)
        return False

    return True


def log_messages(direction: str, messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if not write_message_log({"direction": direction, "message": message}):
            return


def log_response_output(output: list[dict[str, Any]]) -> None:
    for item in output:
        if not write_message_log({"direction": "output", "output": item}):
            return


def ensure_object_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}

    normalized = dict(schema)
    normalized.setdefault("type", "object")

    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        normalized["properties"] = {}

    required = normalized.get("required")
    if not isinstance(required, list):
        normalized["required"] = []

    return normalized


def extract_tool_parameters(tool_like: dict[str, Any]) -> dict[str, Any]:
    return ensure_object_schema(tool_like.get("parameters") or tool_like.get("input_schema") or tool_like.get("schema"))


def normalize_tool_for_chat(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Convert various Responses/OpenAI-like tool schemas into Chat Completions tool format.

    Chat Completions expected shape:
    {
      "type": "function",
      "function": {
        "name": "...",
        "description": "...",
        "parameters": {...}
      }
    }
    """
    if not isinstance(tool, dict):
        return None

    tool_type = tool.get("type")

    # Already Chat Completions format
    if tool_type == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
        return {
            "type": "function",
            "function": {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": extract_tool_parameters(fn),
            },
        }

    # Responses-style function tool:
    # { "type": "function", "name": "...", "description": "...", "parameters": {...} }
    if "name" in tool and tool_type in {"function", "custom"}:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": extract_tool_parameters(tool),
            },
        }

    # Ignore built-in tools such as web_search, file_search, computer_use, etc.
    # vLLM/Qwen cannot execute OpenAI built-ins.
    return None


def normalize_tools_for_chat(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    result = []
    for tool in tools:
        converted = normalize_tool_for_chat(tool)
        if converted and converted["function"].get("name"):
            result.append(converted)
    return result


def extract_text_from_content(content: Any) -> str:
    """Responses API input content can be a string or a list of content parts.
    Convert common text parts into plain text.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Responses-style text parts may use input_text/output_text/text.
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "input_text" in item:
                    parts.append(str(item["input_text"]))
                elif item.get("type") in ("input_text", "output_text") and "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(parts)

    return str(content)


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}

        if isinstance(parsed, dict):
            return parsed

    return {}


def stringify_tool_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(parse_json_object(value), ensure_ascii=False)


def responses_input_to_chat_messages(input_value: Any) -> list[dict[str, Any]]:
    """Convert Responses API-ish input into Chat Completions messages.

    Supports:
    - input: "hello"
    - input: [{role: "user", content: "..."}]
    - function_call_output items as tool results
    """
    messages: list[dict[str, Any]] = []

    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if not isinstance(input_value, list):
        return [{"role": "user", "content": str(input_value)}]

    for item in input_value:
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue

        item_type = item.get("type")
        role = item.get("role")

        # Tool result returned by Codex after executing a function/tool call.
        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": stringify_tool_arguments(item.get("arguments", {})),
                            },
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "unknown_call",
                    "content": extract_text_from_content(item.get("output", "")),
                }
            )
            continue

        if role in ("system", "user", "assistant"):
            messages.append(
                {
                    "role": role,
                    "content": extract_text_from_content(item.get("content", "")),
                }
            )
            continue

        # Fallback
        content = item.get("content") or item.get("text") or item.get("input_text") or ""
        if content:
            messages.append({"role": "user", "content": extract_text_from_content(content)})

    if not messages:
        messages.append({"role": "user", "content": ""})

    return messages


def chat_tool_calls_to_responses_output(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Chat Completions tool_calls to Responses API-like function_call items."""
    output = []

    for tc in tool_calls:
        fn = tc.get("function", {}) or {}
        call_id = tc.get("id") or f"call_{uuid.uuid4().hex}"

        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
                "status": "completed",
            }
        )

    return output


def build_usage(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def build_responses_response(
    *,
    response_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    output_text_parts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                output_text_parts.append(part.get("text", ""))

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "model": model,
        "output": output,
        "output_text": "".join(output_text_parts),
        "usage": build_usage(usage),
    }


def chat_message_to_responses_response(
    *,
    model: str,
    chat_response: dict[str, Any],
) -> dict[str, Any]:
    """Convert a vLLM Chat Completions response to a minimal Responses API-like response."""
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    choice = (chat_response.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}

    output: list[dict[str, Any]] = []

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        output.extend(chat_tool_calls_to_responses_output(tool_calls))

    content = message.get("content")
    if content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )

    return build_responses_response(
        response_id=response_id,
        created_at=created_at,
        model=model,
        status="completed",
        output=output,
        usage=chat_response.get("usage") or {},
    )


def normalize_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice

    if not isinstance(tool_choice, dict):
        return "auto"

    if tool_choice.get("type") == "function":
        function = tool_choice.get("function") or {}
        name = function.get("name") or tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}

    name = tool_choice.get("name")
    if name:
        return {"type": "function", "function": {"name": name}}

    return "auto"


def build_chat_payload(req: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    model = req.get("model") or DEFAULT_MODEL

    messages = responses_input_to_chat_messages(req.get("input", ""))
    if SYSTEM_LANGUAGE_INSTRUCTION:
        messages.insert(
            0,
            {
                "role": "system",
                "content": SYSTEM_LANGUAGE_INSTRUCTION,
            },
        )
    tools = normalize_tools_for_chat(req.get("tools", []))

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # max_output_tokens in Responses API -> max_tokens in Chat Completions
    if "max_output_tokens" in req:
        payload["max_tokens"] = req["max_output_tokens"]
    elif "max_completion_tokens" in req:
        payload["max_tokens"] = req["max_completion_tokens"]

    if "temperature" in req:
        payload["temperature"] = req["temperature"]

    if "top_p" in req:
        payload["top_p"] = req["top_p"]

    if tools:
        payload["tools"] = tools

        # Codex/vLLM often sends/requires auto tool choice.
        # vLLM needs --enable-auto-tool-choice and --tool-call-parser for this.
        payload["tool_choice"] = normalize_tool_choice(req.get("tool_choice", "auto"))

    return payload


def sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def iter_sse_json(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue

        if line == "":
            if not data_lines:
                continue

            raw = "\n".join(data_lines)
            data_lines.clear()

            if raw == "[DONE]":
                break

            yield json.loads(raw)

    if data_lines:
        raw = "\n".join(data_lines)
        if raw != "[DONE]":
            yield json.loads(raw)


def finalize_stream_output(
    *,
    text_parts: list[str],
    tool_call_states: dict[int, dict[str, Any]],
    output_order: list[tuple[str, int]],
    message_item_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for kind, index in output_order:
        if kind == "message":
            output.append(
                {
                    "type": "message",
                    "id": message_item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "".join(text_parts),
                            "annotations": [],
                        }
                    ],
                }
            )
            continue

        state = tool_call_states[index]
        output.append(
            {
                "type": "function_call",
                "id": state["item_id"],
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": "".join(state["arguments_parts"]),
                "status": "completed",
            }
        )

    return output


@app.get("/v1/models")
async def models():
    """Minimal /v1/models endpoint for clients that probe available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-vllm",
            }
        ],
    }


@app.post("/v1/responses")
async def create_response(request: Request):
    req = await request.json()
    wants_stream = bool(req.get("stream", False))
    chat_payload = build_chat_payload(req, stream=wants_stream)
    model = chat_payload["model"]

    log_messages("input", chat_payload["messages"])

    if not wants_stream:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {VLLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=chat_payload,
            )

        if r.status_code >= 400:
            return JSONResponse(
                status_code=r.status_code,
                content={
                    "error": {
                        "message": r.text,
                        "type": "vllm_error",
                        "code": r.status_code,
                    }
                },
            )

        chat_response = r.json()
        responses_response = chat_message_to_responses_response(
            model=model,
            chat_response=chat_response,
        )
        log_response_output(responses_response["output"])
        return JSONResponse(content=responses_response)

    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(
        client.build_request(
            "POST",
            f"{VLLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {VLLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json=chat_payload,
        ),
        stream=True,
    )

    if upstream.status_code >= 400:
        error_text = (await upstream.aread()).decode("utf-8", errors="replace")
        await upstream.aclose()
        await client.aclose()
        return JSONResponse(
            status_code=upstream.status_code,
            content={
                "error": {
                    "message": error_text,
                    "type": "vllm_error",
                    "code": upstream.status_code,
                }
            },
        )

    async def sse():
        response_id = f"resp_{uuid.uuid4().hex}"
        created_at = int(time.time())
        message_item_id = f"msg_{uuid.uuid4().hex}"
        text_parts: list[str] = []
        tool_call_states: dict[int, dict[str, Any]] = {}
        output_order: list[tuple[str, int]] = []
        usage: dict[str, Any] = {}
        message_started = False
        message_output_index: int | None = None

        initial_response = build_responses_response(
            response_id=response_id,
            created_at=created_at,
            model=model,
            status="in_progress",
            output=[],
            usage={},
        )
        yield sse_data({"type": "response.created", "response": initial_response})

        try:
            async for chunk in iter_sse_json(upstream):
                if not isinstance(chunk, dict):
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                if not message_started and delta.get("content"):
                    message_started = True
                    output_order.append(("message", 0))
                    message_output_index = len(output_order) - 1
                    message_item = {
                        "type": "message",
                        "id": message_item_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    yield sse_data(
                        {
                            "type": "response.output_item.added",
                            "output_index": message_output_index,
                            "item": message_item,
                        }
                    )
                    yield sse_data(
                        {
                            "type": "response.content_part.added",
                            "item_id": message_item_id,
                            "output_index": message_output_index,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        }
                    )

                content_delta = delta.get("content")
                if content_delta:
                    text_parts.append(content_delta)
                    yield sse_data(
                        {
                            "type": "response.output_text.delta",
                            "item_id": message_item_id,
                            "output_index": message_output_index,
                            "content_index": 0,
                            "delta": content_delta,
                        }
                    )

                for tool_call in delta.get("tool_calls") or []:
                    index = tool_call.get("index", 0)
                    state = tool_call_states.get(index)
                    if state is None:
                        state = {
                            "item_id": f"fc_{uuid.uuid4().hex}",
                            "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                            "name": "",
                            "arguments_parts": [],
                        }
                        tool_call_states[index] = state
                        output_order.append(("function_call", index))
                        yield sse_data(
                            {
                                "type": "response.output_item.added",
                                "output_index": len(output_order) - 1,
                                "item": {
                                    "type": "function_call",
                                    "id": state["item_id"],
                                    "call_id": state["call_id"],
                                    "name": "",
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            }
                        )

                    if tool_call.get("id"):
                        state["call_id"] = tool_call["id"]

                    function_delta = tool_call.get("function") or {}
                    if function_delta.get("name"):
                        state["name"] += function_delta["name"]

                    arguments_delta = function_delta.get("arguments")
                    if arguments_delta:
                        state["arguments_parts"].append(arguments_delta)
                        yield sse_data(
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": state["item_id"],
                                "output_index": output_order.index(("function_call", index)),
                                "delta": arguments_delta,
                            }
                        )

            if message_started:
                full_text = "".join(text_parts)
                yield sse_data(
                    {
                        "type": "response.output_text.done",
                        "item_id": message_item_id,
                        "output_index": message_output_index,
                        "content_index": 0,
                        "text": full_text,
                    }
                )
                yield sse_data(
                    {
                        "type": "response.content_part.done",
                        "item_id": message_item_id,
                        "output_index": message_output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": [],
                        },
                    }
                )
                yield sse_data(
                    {
                        "type": "response.output_item.done",
                        "output_index": message_output_index,
                        "item": {
                            "type": "message",
                            "id": message_item_id,
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": full_text,
                                    "annotations": [],
                                }
                            ],
                        },
                    }
                )

            for kind, index in output_order:
                if kind != "function_call":
                    continue

                state = tool_call_states[index]
                arguments = "".join(state["arguments_parts"])
                output_index = output_order.index((kind, index))
                yield sse_data(
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": state["item_id"],
                        "output_index": output_index,
                        "arguments": arguments,
                    }
                )
                yield sse_data(
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": state["item_id"],
                            "call_id": state["call_id"],
                            "name": state["name"],
                            "arguments": arguments,
                            "status": "completed",
                        },
                    }
                )

            final_output = finalize_stream_output(
                text_parts=text_parts,
                tool_call_states=tool_call_states,
                output_order=output_order,
                message_item_id=message_item_id,
            )
            final_response = build_responses_response(
                response_id=response_id,
                created_at=created_at,
                model=model,
                status="completed",
                output=final_output,
                usage=usage,
            )
            log_response_output(final_output)
            yield sse_data({"type": "response.completed", "response": final_response})
            yield "data: [DONE]\n\n"
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health():
    return {"ok": True}
