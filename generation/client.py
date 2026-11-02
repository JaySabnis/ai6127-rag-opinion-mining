"""Small DeepSeek JSON client. Standard library only; no local model weights."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def load_env(path=None):
    """Read simple KEY=VALUE dotenv files without executing/interpolating them."""
    path = Path(path or ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.replace("_", "").isalnum() and value:
            os.environ.setdefault(key, value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as file:
        name = file.name
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(name, path)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # never forward an Authorization header to another host


class DeepSeekClient:
    def __init__(self, *, env_file=None, model=None, cache_dir=None,
                 timeout=60, max_calls=32, transport=None):
        load_env(env_file)
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        if urlparse(self.base_url).scheme != "https":
            raise ValueError("DEEPSEEK_BASE_URL must use HTTPS")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        if not self.api_key and transport is None:
            raise ValueError("Set DEEPSEEK_API_KEY in the local .env or environment")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout, self.max_calls = timeout, max_calls
        self.transport = transport
        self.calls = self.cache_hits = 0
        self.usage = []

    def _request(self, body):
        for attempt in range(3):
            if self.calls >= self.max_calls:
                raise RuntimeError("API call budget reached; increase --max-api-calls explicitly")
            self.calls += 1
            try:
                if self.transport:
                    return self.transport(body)
                req = urllib.request.Request(self.base_url + "/chat/completions",
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Authorization": "Bearer " + self.api_key,
                             "Content-Type": "application/json"})
                opener = urllib.request.build_opener(_NoRedirect())
                with opener.open(req, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                # Do not print request headers, URLs, response bodies or credentials.
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"DeepSeek HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise RuntimeError("DeepSeek connection failed; no prediction was substituted") from None
            time.sleep(2 ** attempt)
        raise RuntimeError("DeepSeek retries exhausted")

    def json(self, system, payload, *, validate, max_tokens=4096):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        body = {"model": self.model, "messages": messages,
                "thinking": {"type": "disabled"}, "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": max_tokens, "stream": False}
        digest = hashlib.sha256(json.dumps([self.base_url, body], sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest()
        cache = self.cache_dir / (digest + ".json") if self.cache_dir else None
        if cache and cache.exists():
            try:
                stored = json.loads(cache.read_text())
                result = validate(stored["result"])
                self.cache_hits += 1
                return result
            except (ValueError, KeyError, TypeError):
                pass  # stale/invalid cache: recompute and validate
        for attempt in range(2):
            response = self._request(body)
            self.usage.append({"model": response.get("model", self.model),
                               **response.get("usage", {})})
            try:
                choice = response["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ValueError("incomplete response")
                value = json.loads(choice["message"]["content"])
                result = validate(value)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                if cache:
                    choices = response.get("choices") or []
                    first = choices[0] if choices and isinstance(choices[0], dict) else {}
                    message = first.get("message") or {}
                    write_json(cache.with_suffix(f".invalid-{attempt}.json"), {
                        "validation_error": str(exc),
                        "content": message.get("content") if isinstance(message, dict) else None,
                    })
                if attempt:
                    raise RuntimeError("DeepSeek returned invalid structured output after one retry: " + str(exc)) from None
                # A repair retry is bounded and remains subject to the call budget.
                body["messages"] = messages + [{"role": "user", "content":
                    "The prior JSON did not satisfy the schema: " + str(exc) + ". Return the specified JSON only. "
                    "Validate every ID, label and verbatim quote. No markdown."}]
                continue
            if cache:
                write_json(cache, {"result": result, "model": response.get("model"),
                                   "usage": response.get("usage", {})})
            return result
        raise RuntimeError("Invalid JSON output")

    def telemetry(self):
        return {"api_calls": self.calls, "cache_hits": self.cache_hits,
                "model": self.model, "usage": self.usage}
