"""LLM client implementation supporting OpenRouter and OpenAI-compatible providers.

Environment variables required:
- OPENROUTER_API_KEY
Optional overrides:
- OPENROUTER_BASE_URL (defaults to https://openrouter.ai/api/v1)
- CHAT_MODEL (defaults to nvidia/nemotron-3.5-lightning:free)
- EMBEDDING_MODEL (defaults to nvidia/nemotron-3-embed-1b:free)

Security notice:
API keys and secrets must never be hardcoded, logged, or printed.
Unmasked PII or credit card details must never be sent to external APIs.
"""

from __future__ import annotations

import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Custom Exception Hierarchy
class LLMClientError(Exception):
    """Base exception for all LLM client errors."""
    pass


class MissingCredentialsError(LLMClientError):
    """Raised when required API credentials are not set in the environment."""
    pass


class RateLimitError(LLMClientError):
    """Raised when the provider returns HTTP 429 (rate limit exceeded)."""
    pass


class LLMTimeoutError(LLMClientError):
    """Raised when an API request times out."""
    pass


class ProviderAPIError(LLMClientError):
    """Raised when the provider returns an unexpected HTTP error."""
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class MalformedResponseError(LLMClientError):
    """Raised when the API returns an unparseable or unexpected response body."""
    pass


def _load_env_if_needed() -> None:
    """Safely load .env into os.environ if present without exposing values."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("\"'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        # Ignore disk read errors quietly to preserve confidentiality
        pass


def get_api_key() -> str:
    """Retrieve the API key from environment variables at runtime.

    Reads the OpenRouter key only; other provider keys will not authenticate here.
    Never prints or logs the key value.
    """
    _load_env_if_needed()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or not key.strip():
        raise MissingCredentialsError(
            "Missing API key. Set OPENROUTER_API_KEY in the environment or local .env file."
        )
    return key.strip()


def get_base_url() -> str:
    """Return the provider base URL."""
    _load_env_if_needed()
    return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def _get_ssl_context() -> ssl.SSLContext:
    """Create an SSL context that leverages certifi if available for macOS compatibility."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def extract_json(content: str) -> dict[str, Any] | list[Any] | None:
    """Safely extract and parse JSON from text, handling markdown fences and raw text.

    Returns the parsed dict/list, or None if no valid JSON structure could be found.
    """
    if not content or not isinstance(content, str):
        return None

    cleaned = content.strip()

    # Case 1: Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Case 2: Handle markdown code fences ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned, re.IGNORECASE)
    if match:
        snippet = match.group(1).strip()
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # Case 3: Find outermost curly braces or square brackets
    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        snippet = cleaned[start_brace : end_brace + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    start_bracket = cleaned.find("[")
    end_bracket = cleaned.rfind("]")
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        snippet = cleaned[start_bracket : end_bracket + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    return None


def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Execute a chat completion call with the configured provider.

    Parameters:
    - messages: list of dicts with 'role' and 'content' keys.
    - model: provider model ID (defaults to CHAT_MODEL or 'nvidia/nemotron-3.5-lightning:free').
    - temperature: sampling temperature (default 0.0).
    - response_format: optional format dict, e.g. {"type": "json_object"}.
    - max_tokens: optional completion token limit.
    - timeout: request timeout in seconds (default 30.0).
    - max_retries: number of retry attempts on transient 429/5xx errors (default 2).

    Returns:
    dict containing:
    - content: str
    - role: str
    - model: str
    - finish_reason: str
    - usage: dict
    - parsed_json: dict | list | None
    """
    api_key = get_api_key()
    base_url = get_base_url()
    selected_model = model or os.environ.get("CHAT_MODEL", "nvidia/nemotron-3.5-lightning:free")

    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.environ.get("OPENROUTER_SITE_URL")
    app_name = os.environ.get("OPENROUTER_APP_NAME", "Retail Operations Copilot")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    # Nemotron Lightning Free does not support response_format.
    # Ask for JSON in the prompt and validate it with extract_json instead.
    if selected_model == "nvidia/nemotron-3.5-lightning:free":
        payload.pop("response_format", None)

    data_bytes = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/chat/completions"
    ssl_context = _get_ssl_context()

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as response:
                raw_bytes = response.read()
                raw_text = raw_bytes.decode("utf-8", errors="replace")

            try:
                res_json = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise MalformedResponseError("Provider returned invalid JSON response.") from e

            choices = res_json.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                raise MalformedResponseError("Provider returned a response with no choices.")

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            role = message.get("role", "assistant")
            finish_reason = choice.get("finish_reason", "stop")
            usage = res_json.get("usage", {})
            actual_model = res_json.get("model", selected_model)

            parsed_json = extract_json(content) if (response_format or content.strip().startswith(("{", "["))) else None

            return {
                "content": content,
                "role": role,
                "model": actual_model,
                "finish_reason": finish_reason,
                "usage": usage,
                "parsed_json": parsed_json,
            }

        except urllib.error.HTTPError as http_err:
            status_code = http_err.code
            if status_code == 429:
                last_error = RateLimitError("Rate limit exceeded (HTTP 429).")
                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise last_error

            if status_code in (500, 502, 503, 504):
                last_error = ProviderAPIError(
                    f"Provider service unavailable (HTTP {status_code}).",
                    status_code=status_code,
                )
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error

            raise ProviderAPIError(
                f"Provider API call failed with HTTP {status_code}.",
                status_code=status_code,
            )

        except (urllib.error.URLError, socket.timeout, TimeoutError) as err:
            err_str = str(err).lower()
            if "timed out" in err_str:
                last_error = LLMTimeoutError(f"The API request timed out after {timeout} seconds.")
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                raise last_error
            raise ProviderAPIError(f"Network error connecting to provider: {err}")

        except Exception as e:
            if isinstance(e, LLMClientError):
                raise
            raise ProviderAPIError(f"Unexpected error calling provider: {e}") from e

    if last_error:
        raise last_error
    raise ProviderAPIError("Failed to obtain response from provider.")


def get_embedding(
    text: str | list[str],
    model: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate vector embeddings for a string or list of strings.

    Parameters:
    - text: single string or list of strings to embed.
    - model: embedding model name (defaults to EMBEDDING_MODEL or 'nvidia/nemotron-3-embed-1b:free').
    - timeout: request timeout in seconds.
    - max_retries: retry attempts for transient errors.

    Returns:
    dict containing:
    - embeddings: list[list[float]]
    - model: str
    - dimensions: int
    """
    api_key = get_api_key()
    base_url = get_base_url()
    selected_model = model or os.environ.get("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")

    payload = {
        "model": selected_model,
        "input": text,
    }

    data_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.environ.get("OPENROUTER_SITE_URL")
    app_name = os.environ.get("OPENROUTER_APP_NAME", "Retail Operations Copilot")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    url = f"{base_url}/embeddings"
    ssl_context = _get_ssl_context()

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as response:
                raw_bytes = response.read()
                raw_text = raw_bytes.decode("utf-8", errors="replace")

            try:
                res_json = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise MalformedResponseError("Provider returned invalid JSON for embeddings.") from e

            data_list = res_json.get("data")
            if not data_list or not isinstance(data_list, list):
                raise MalformedResponseError("Provider embeddings response is missing its data list.")

            embeddings: list[list[float]] = [item.get("embedding", []) for item in data_list]
            dimensions = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
            actual_model = res_json.get("model", selected_model)

            return {
                "embeddings": embeddings,
                "model": actual_model,
                "dimensions": dimensions,
            }

        except urllib.error.HTTPError as http_err:
            status_code = http_err.code
            if status_code == 429:
                last_error = RateLimitError("Embedding rate limit exceeded (HTTP 429).")
                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise last_error

            if status_code in (500, 502, 503, 504):
                last_error = ProviderAPIError(
                    f"Provider service unavailable for embeddings (HTTP {status_code}).",
                    status_code=status_code,
                )
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error

            raise ProviderAPIError(
                f"Embedding API call failed with HTTP {status_code}.",
                status_code=status_code,
            )

        except Exception as e:
            if isinstance(e, LLMClientError):
                raise
            raise ProviderAPIError(f"Unexpected error getting embeddings: {e}") from e

    if last_error:
        raise last_error
    raise ProviderAPIError("Failed to obtain embeddings from provider.")