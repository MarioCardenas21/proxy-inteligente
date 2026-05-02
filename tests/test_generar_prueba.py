import asyncio

import httpx

from app import main
from app.llm_client import OllamaGenerateResult, OllamaUnavailableError


async def _post(path, **kwargs):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, **kwargs)


def test_generar_prueba_exitoso_con_mock(monkeypatch):
    async def fake_generate_with_ollama(request):
        return OllamaGenerateResult(
            text="import unittest\n\nclass TestSuma(unittest.TestCase):\n    def test_suma(self):\n        assert suma(2, 3) == 5",
            latency_ms=12,
        )

    monkeypatch.setattr(main, "generate_with_ollama", fake_generate_with_ollama)

    response = asyncio.run(
        _post(
            "/generar-prueba",
            json={"lenguaje": "Python", "codigo": "def suma(a, b): return a + b"},
        )
    )

    assert response.status_code == 200
    assert response.json()["latencia_ms"] == 12
    assert "class TestSuma" in response.json()["prueba"]


def test_rechaza_content_type_no_json():
    response = asyncio.run(
        _post(
            "/generar-prueba",
            content='{"lenguaje": "Python", "codigo": "def x(): return 1"}',
            headers={"Content-Type": "text/plain"},
        )
    )

    assert response.status_code == 415


def test_rechaza_lenguaje_vacio_sin_llamar_llm(monkeypatch):
    async def fake_generate_with_ollama(request):
        raise AssertionError("No debe llamarse al cliente LLM con payload invalido.")

    monkeypatch.setattr(main, "generate_with_ollama", fake_generate_with_ollama)

    response = asyncio.run(
        _post(
            "/generar-prueba",
            json={"lenguaje": "   ", "codigo": "def x(): return 1"},
        )
    )

    assert response.status_code == 422


def test_respuesta_no_vacia_se_acepta(monkeypatch):
    async def fake_generate_with_ollama(request):
        return OllamaGenerateResult(text="Aqui tienes una explicacion sin codigo.", latency_ms=7)

    monkeypatch.setattr(main, "generate_with_ollama", fake_generate_with_ollama)

    response = asyncio.run(
        _post(
            "/generar-prueba",
            json={"lenguaje": "Python", "codigo": "def x(): return 1"},
        )
    )

    assert response.status_code == 200
    assert response.json() == {
        "prueba": "Aqui tienes una explicacion sin codigo.",
        "latencia_ms": 7,
    }


def test_respuesta_vacia_devuelve_500(monkeypatch):
    async def fake_generate_with_ollama(request):
        return OllamaGenerateResult(text="   ", latency_ms=7)

    monkeypatch.setattr(main, "generate_with_ollama", fake_generate_with_ollama)

    response = asyncio.run(
        _post(
            "/generar-prueba",
            json={"lenguaje": "Python", "codigo": "def x(): return 1"},
        )
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "La respuesta del modelo esta vacia."


def test_ollama_no_disponible_devuelve_500(monkeypatch):
    async def fake_generate_with_ollama(request):
        raise OllamaUnavailableError("Ollama no esta disponible.")

    monkeypatch.setattr(main, "generate_with_ollama", fake_generate_with_ollama)

    response = asyncio.run(
        _post(
            "/generar-prueba",
            json={"lenguaje": "Python", "codigo": "def x(): return 1"},
        )
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Ollama no esta disponible."
