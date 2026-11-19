"""
LM Studio connection layer.

Every model call in the app (the CrewAI agents and the vision/text tools) goes
through the local LM Studio server via its OpenAI-compatible API. Models are
loaded with a 65,536-token context window through LM Studio's REST API
(/api/v1/models/load) before they are used.

All settings can be overridden with environment variables or a .env file:

    LMSTUDIO_BASE_URL        OpenAI-compatible endpoint (default http://localhost:1234/v1)
    LMSTUDIO_API_KEY         Any non-empty string unless LM Studio auth is enabled
    LMSTUDIO_TEXT_MODEL      Model for the CrewAI agents and text-only tools
    LMSTUDIO_VISION_MODEL    Vision-capable model for image analysis
    LMSTUDIO_CONTEXT_LENGTH  Context window to load models with (default 65536)
    LMSTUDIO_MAX_TOKENS      Max tokens generated per response (default 8192)
    LMSTUDIO_TEMPERATURE     Sampling temperature (default 0.2)
    LMSTUDIO_TIMEOUT         Request timeout in seconds (default 600)
    LMSTUDIO_AUTO_LOAD       Load/reload models with the context length above (default true)
    LMSTUDIO_REASONING_EFFORT  Reasoning for thinking models such as Qwen3.5:
                             none/low/medium/high, or empty to use the model default
                             (default none: much faster and keeps the token budget for answers)
"""

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

import requests
from crewai import LLM
from crewai.utilities.converter import Converter, ConverterError
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
TEXT_MODEL = os.getenv("LMSTUDIO_TEXT_MODEL", "qwen/qwen3.5-9b")
VISION_MODEL = os.getenv("LMSTUDIO_VISION_MODEL", "qwen/qwen3.5-9b")
CONTEXT_LENGTH = int(os.getenv("LMSTUDIO_CONTEXT_LENGTH", "65536"))
MAX_TOKENS = int(os.getenv("LMSTUDIO_MAX_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("LMSTUDIO_TEMPERATURE", "0.2"))
TIMEOUT = float(os.getenv("LMSTUDIO_TIMEOUT", "600"))
AUTO_LOAD = os.getenv("LMSTUDIO_AUTO_LOAD", "true").strip().lower() not in ("0", "false", "no", "off")
REASONING_EFFORT = os.getenv("LMSTUDIO_REASONING_EFFORT", "none").strip()
# Sent as extra request-body fields so LiteLLM forwards them to LM Studio untouched.
EXTRA_BODY = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}

# Root of the LM Studio REST API (http://host:port), derived from the /v1 endpoint.
_SERVER_ROOT = re.sub(r"/v1$", "", LMSTUDIO_BASE_URL)
_HEADERS = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}"}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_ready_models = set()
_load_lock = threading.Lock()

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY, timeout=TIMEOUT)


def strip_reasoning(text: Optional[str]) -> str:
    """Remove <think>...</think> blocks that some reasoning models emit inline."""
    return _THINK_RE.sub("", text or "").strip()


def ensure_model_loaded(model: str, context_length: int = CONTEXT_LENGTH) -> None:
    """
    Make sure `model` is loaded in LM Studio with at least `context_length` tokens.

    If it is not loaded, it is loaded with that context length. If it is loaded
    with a smaller context, it is unloaded and reloaded. Without this, LM Studio's
    just-in-time loading would use the model's default (usually much smaller)
    context window.
    """
    if not AUTO_LOAD or model in _ready_models:
        return

    with _load_lock:
        if model in _ready_models:
            return

        try:
            resp = requests.get(f"{_SERVER_ROOT}/api/v1/models", headers=_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(
                f"Could not reach LM Studio at {_SERVER_ROOT}. Start the LM Studio server "
                f"(Developer tab > Start Server, or `lms server start`). Details: {e}"
            ) from e

        info = next((m for m in resp.json().get("models", []) if m.get("key") == model), None)
        if info is None:
            raise ValueError(
                f"Model '{model}' is not downloaded in LM Studio. "
                f"Download it there or run `lms get {model}`."
            )

        max_ctx = info.get("max_context_length")
        if max_ctx and max_ctx < context_length:
            raise ValueError(
                f"Model '{model}' supports at most {max_ctx} tokens of context, "
                f"but {context_length} were requested. Choose another model."
            )

        instances = info.get("loaded_instances") or []
        if any((i.get("config") or {}).get("context_length", 0) >= context_length for i in instances):
            _ready_models.add(model)
            return

        for inst in instances:
            logger.info("Unloading %s (context %s is below %s)", inst.get("id"),
                        (inst.get("config") or {}).get("context_length"), context_length)
            requests.post(f"{_SERVER_ROOT}/api/v1/models/unload", headers=_HEADERS,
                          json={"instance_id": inst["id"]}, timeout=120).raise_for_status()

        logger.info("Loading %s in LM Studio with a %s-token context...", model, context_length)
        load = requests.post(
            f"{_SERVER_ROOT}/api/v1/models/load",
            headers=_HEADERS,
            # parallel=1 gives the single active request the whole context window.
            json={"model": model, "context_length": context_length, "parallel": 1},
            timeout=TIMEOUT,
        )
        if not load.ok:
            raise RuntimeError(f"LM Studio failed to load '{model}': {load.status_code} {load.text}")
        _ready_models.add(model)


def chat(messages: List[Dict[str, Any]], model: str = TEXT_MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """Send a chat completion to LM Studio and return the reply text."""
    ensure_model_loaded(model)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        extra_body=EXTRA_BODY or None,
    )
    return strip_reasoning(response.choices[0].message.content)


class LMStudioLLM(LLM):
    """CrewAI LLM that targets LM Studio and knows its real context window."""

    def __init__(self, model: str = TEXT_MODEL, context_length: int = CONTEXT_LENGTH, **kwargs):
        self.lmstudio_model = model
        self.context_length = context_length
        kwargs.setdefault("temperature", TEMPERATURE)
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        kwargs.setdefault("timeout", TIMEOUT)
        if EXTRA_BODY:
            kwargs.setdefault("extra_body", dict(EXTRA_BODY))
        # The "openai/" prefix tells LiteLLM to speak the OpenAI protocol to base_url.
        super().__init__(
            model=f"openai/{model}",
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY,
            **kwargs,
        )

    def call(self, messages, callbacks=[]):
        ensure_model_loaded(self.lmstudio_model, self.context_length)
        return strip_reasoning(super().call(messages, callbacks))

    def supports_function_calling(self) -> bool:
        # CrewAI's instructor-based paths (structured output, tool calling) call
        # LiteLLM with only the model name, dropping base_url/api_key, so they
        # would go to api.openai.com. Returning False routes them through
        # self.call(), which talks to LM Studio.
        return False

    def get_context_window_size(self) -> int:
        # CrewAI uses 75% of the window as its safety margin when summarising history.
        return int(self.context_length * 0.75)


def crew_llm() -> LMStudioLLM:
    """LLM instance for the CrewAI agents."""
    return LMStudioLLM(model=TEXT_MODEL)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be wrapped in ``` fences or prose."""
    text = strip_reasoning(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text.strip()


class LMStudioConverter(Converter):
    """
    Turns an agent's final answer into the task's Pydantic schema using LM Studio.

    CrewAI 0.75's built-in converter either bypasses LM Studio (instructor path)
    or, on the plain-LLM path, omits the schema and returns a string instead of a
    dict. This converter sends the JSON schema, strips code fences and validates
    the result, retrying up to max_attempts times.
    """

    def _convert(self):
        schema = json.dumps(self.model.model_json_schema(), indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    "Convert the user's text into a single JSON object that matches this JSON schema. "
                    "Use only information from the text. Reply with the JSON object only, "
                    "no markdown fences and no commentary.\n\n" + schema
                ),
            },
            {"role": "user", "content": self.text},
        ]
        last_error = None
        for _ in range(self.max_attempts):
            reply = self.llm.call(messages)
            try:
                return self.model.model_validate_json(extract_json(reply))
            except Exception as e:  # invalid JSON or schema mismatch: ask again with the error
                last_error = e
                logger.warning("JSON conversion attempt failed: %s", e)
                messages = messages[:2] + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"That JSON was invalid: {e}\nReply with corrected JSON only."},
                ]
        return ConverterError(f"Failed to convert text into {self.model.__name__}: {last_error}")

    def to_pydantic(self, current_attempt=1):
        return self._convert()

    def to_json(self, current_attempt=1):
        result = self._convert()
        return result if isinstance(result, ConverterError) else result.model_dump()
