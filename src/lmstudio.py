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

import ast
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
TEXT_MODEL = os.getenv("LMSTUDIO_TEXT_MODEL", "qwen/qwen3.6-35b-a3b")
VISION_MODEL = os.getenv("LMSTUDIO_VISION_MODEL", "qwen/qwen3.6-35b-a3b")
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
# Inline reasoning: <think>...</think> (Qwen, DeepSeek) and <|channel>...<channel|> (Gemma 4)
_THINK_RE = re.compile(r"<think>.*?</think>\s*|<\|channel>.*?<channel\|>\s*", re.DOTALL | re.IGNORECASE)
# Gemma 4's native tool-call syntax: <|tool_call>call:NAME{key: value, ...}<tool_call|>
# Gemma writes arguments either as {key: value} or Python-style (key="value");
# the closing <tool_call|> token is sometimes dropped.
_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>\s*call:\s*([^{(]+?)\s*(\{.*\}|\(.*\))\s*(?:<tool_call\|>|$)", re.DOTALL
)

_ready_models = set()
_load_lock = threading.Lock()

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY, timeout=TIMEOUT)


def strip_reasoning(text: Optional[str]) -> str:
    """Remove <think>...</think> blocks that some reasoning models emit inline."""
    return _THINK_RE.sub("", text or "").strip()


def _gemma_args_to_json(args: str) -> str:
    """
    Convert Gemma's tool-argument notation to JSON.

    Gemma writes {image_input: "x.jpg"} or {image_input:<|"|>x.jpg<|"|>}: keys are
    unquoted and strings may use <|"|> as the quote token. Keys are quoted only
    outside string literals, so values containing ':' or ',' are left intact.
    """
    args = args.replace('<|"|>', '"')
    out, i, in_str = [], 0, False
    while i < len(args):
        ch = args[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(args):
                out.append(args[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        else:
            key = re.match(r"([A-Za-z_][\w\-]*)\s*:", args[i:])
            prev = "".join(out).rstrip()[-1:] if out else ""
            if key and prev in ("{", ","):
                out.append(f'"{key.group(1)}":')
                i += key.end()
                continue
            out.append(ch)
        i += 1
    return "".join(out)


def native_tool_call_to_react(text: str) -> str:
    """
    Rewrite a Gemma-style native tool call as the Thought/Action/Action Input
    text that CrewAI's ReAct parser expects. Other text is returned unchanged.
    """
    match = _GEMMA_TOOL_CALL_RE.search(text or "")
    if not match:
        return text
    name, raw_args = match.group(1).strip(), match.group(2).replace('<|"|>', '"')
    try:
        if raw_args.startswith("("):
            # Python-style keyword arguments: (image_input="x.jpg", n=2)
            call = ast.parse(f"f{raw_args}", mode="eval").body
            if call.args:
                raise ValueError("positional arguments")
            args = json.dumps({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
        else:
            args = json.dumps(json.loads(_gemma_args_to_json(raw_args)))
    except (ValueError, SyntaxError, json.JSONDecodeError):
        args = raw_args
    thought = text[:match.start()].strip() or f"I should use the {name} tool."
    return f"Thought: {thought}\nAction: {name}\nAction Input: {args}"


def ensure_model_loaded(model: str, context_length: int = CONTEXT_LENGTH, force_reload: bool = False) -> None:
    """
    Make sure `model` is loaded in LM Studio with exactly `context_length` tokens.

    If it is not loaded, it is loaded with that context length. If it is loaded
    with a different context, it is unloaded and reloaded (a larger window than
    configured wastes RAM, a smaller one truncates prompts). Without this, LM Studio's
    just-in-time loading would use the model's default (usually much smaller)
    context window. With force_reload=True the model is always unloaded and
    loaded again, which recovers it after an engine crash.
    """
    if force_reload:
        _ready_models.discard(model)
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
        if not force_reload and any(
            (i.get("config") or {}).get("context_length", 0) == context_length for i in instances
        ):
            _ready_models.add(model)
            return

        for inst in instances:
            logger.info("Unloading %s (context %s, want %s)", inst.get("id"),
                        (inst.get("config") or {}).get("context_length"), context_length)
            unload = requests.post(f"{_SERVER_ROOT}/api/v1/models/unload", headers=_HEADERS,
                                   json={"instance_id": inst["id"]}, timeout=120)
            if not unload.ok and not force_reload:  # a crashed instance may already be gone
                unload.raise_for_status()

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


# Errors LM Studio returns when its inference engine crashed (often out of memory)
# and the model has to be reloaded before it can answer again.
_ENGINE_CRASH_MARKERS = ("bad allocation", "fetch failed", "model unloaded", "model has crashed", "no models loaded")


def with_engine_recovery(model: str, request, context_length: int = CONTEXT_LENGTH):
    """Run `request()`; if LM Studio's engine crashed, reload the model once and retry."""
    try:
        return request()
    except Exception as e:
        if not any(marker in str(e).lower() for marker in _ENGINE_CRASH_MARKERS):
            raise
        logger.warning("LM Studio engine error (%s); reloading %s and retrying once", e, model)
        ensure_model_loaded(model, context_length, force_reload=True)
        return request()


def chat(messages: List[Dict[str, Any]], model: str = TEXT_MODEL, max_tokens: int = MAX_TOKENS) -> str:
    """Send a chat completion to LM Studio and return the reply text."""
    ensure_model_loaded(model)
    response = with_engine_recovery(model, lambda: client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        extra_body=EXTRA_BODY or None,
    ))
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
        reply = with_engine_recovery(
            self.lmstudio_model, lambda: super(LMStudioLLM, self).call(messages, callbacks), self.context_length
        )
        if not (reply or "").strip():
            # Some models occasionally return an empty turn (e.g. right after a tool
            # result); CrewAI would count it as a format error, so ask once more.
            logger.warning("Empty reply from %s; retrying once", self.lmstudio_model)
            reply = with_engine_recovery(
                self.lmstudio_model, lambda: super(LMStudioLLM, self).call(messages, callbacks), self.context_length
            )
        logger.debug("LM Studio raw reply: %r", reply)
        return native_tool_call_to_react(strip_reasoning(reply))

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
