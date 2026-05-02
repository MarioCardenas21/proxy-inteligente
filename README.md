# Proxy Inteligente LLM

Sistema desarrollado con FastAPI que actúa como intermediario entre un cliente y un modelo de lenguaje (LLM) utilizando Ollama.

## Características

- Generación automática de pruebas unitarias
- Soporte para múltiples lenguajes (Python, Java, JavaScript)
- Validación de entrada y salida
- Manejo de errores (HTTP 500)
- Logs de inferencia (latencia, lenguaje, estado)

## Tecnologías

- FastAPI
- Ollama
- Python
- httpx

## Ejecución

```bash
uvicorn app.main:app --reload
