import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from app.prompt import SYSTEM_PROMPT


load_dotenv()


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    url: str
    model: str
    temperature: float
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class OllamaGenerateRequest:
    system: str
    prompt: str


@dataclass(frozen=True, slots=True)
class OllamaGenerateResult:
    text: str
    latency_ms: int


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaInvalidResponseError(RuntimeError):
    pass


def get_ollama_config() -> OllamaConfig:
    return OllamaConfig(
        url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
        model=os.getenv("OLLAMA_MODEL", "llama3:latest"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT", "30")),
    )


async def generate_with_ollama(
    request: OllamaGenerateRequest,
    config: OllamaConfig | None = None,
) -> OllamaGenerateResult:
    active_config = config or get_ollama_config()
    payload = {
        "model": active_config.model,
        "system": request.system or SYSTEM_PROMPT,
        "prompt": request.prompt,
        "stream": False,
        "options": {"temperature": active_config.temperature},
    }

    started_at = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=active_config.timeout_seconds) as client:
            response = await client.post(active_config.url, json=payload)
            response.raise_for_status()
    except httpx.RequestError as exc:
        raise OllamaUnavailableError("Ollama no esta disponible.") from exc
    except httpx.HTTPStatusError as exc:
        raise OllamaUnavailableError(
            f"Ollama respondio con HTTP {exc.response.status_code}."
        ) from exc

    latency_ms = round((time.perf_counter() - started_at) * 1000)

    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaInvalidResponseError("Ollama devolvio JSON invalido.") from exc

    generated_text = data.get("response")
    if not isinstance(generated_text, str):
        raise OllamaInvalidResponseError("Ollama no devolvio el campo response.")

    return OllamaGenerateResult(text=generated_text.strip(), latency_ms=latency_ms)
