import time
from json import JSONDecodeError

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import ValidationError

from app.llm_client import (
    OllamaGenerateRequest,
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    generate_with_ollama,
)
from app.logger import InferenceLog, log_inference
from app.prompt import PromptInput, SYSTEM_PROMPT, build_user_prompt
from app.schemas import GenerateTestRequest, GenerateTestResponse
from app.validators import looks_like_code, strip_markdown_fences


app = FastAPI(title="Proxy Inteligente LLM")


def _is_json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    return content_type == "application/json" or content_type.endswith("+json")


@app.post("/generar-prueba", response_model=GenerateTestResponse)
async def generar_prueba(request: Request) -> GenerateTestResponse:
    started_at = time.perf_counter()
    lenguaje = "-"

    if not _is_json_request(request):
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        log_inference(InferenceLog(lenguaje=lenguaje, latencia_ms=latency_ms, status="Fallo"))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="La solicitud debe usar Content-Type application/json.",
        )

    try:
        raw_payload = await request.json()
    except JSONDecodeError as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        log_inference(InferenceLog(lenguaje=lenguaje, latencia_ms=latency_ms, status="Fallo"))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El cuerpo de la solicitud debe ser JSON valido.",
        ) from exc

    if isinstance(raw_payload, dict):
        lenguaje = str(raw_payload.get("lenguaje") or "-")

    try:
        payload = GenerateTestRequest.model_validate(raw_payload)
    except ValidationError as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        log_inference(InferenceLog(lenguaje=lenguaje, latencia_ms=latency_ms, status="Fallo"))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(),
        ) from exc

    lenguaje = payload.lenguaje
    prompt = build_user_prompt(PromptInput(lenguaje=payload.lenguaje, codigo=payload.codigo))

    try:
        llm_result = await generate_with_ollama(
            OllamaGenerateRequest(system=SYSTEM_PROMPT, prompt=prompt)
        )
        generated_code = strip_markdown_fences(llm_result.text)

        if not looks_like_code(generated_code):
            log_inference(
                InferenceLog(
                    lenguaje=lenguaje,
                    latencia_ms=llm_result.latency_ms,
                    status="Fallo",
                )
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="La respuesta del modelo esta vacia.",
            )

        log_inference(
            InferenceLog(
                lenguaje=lenguaje,
                latencia_ms=llm_result.latency_ms,
                status="Exito",
            )
        )
        return GenerateTestResponse(prueba=generated_code, latencia_ms=llm_result.latency_ms)

    except (OllamaUnavailableError, OllamaInvalidResponseError) as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        log_inference(InferenceLog(lenguaje=lenguaje, latencia_ms=latency_ms, status="Fallo"))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
