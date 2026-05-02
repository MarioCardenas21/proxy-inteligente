from dataclasses import dataclass


SYSTEM_PROMPT = """
Eres un generador estricto de pruebas unitarias.
Debes responder unicamente con codigo fuente de pruebas unitarias valido para el lenguaje solicitado.
No expliques nada.
No uses Markdown.
No incluyas bloques con triple backtick.
No incluyas texto introductorio, comentarios narrativos ni conclusiones.
Si faltan detalles, infiere pruebas razonables usando librerias estandar del lenguaje.
""".strip()


@dataclass(frozen=True, slots=True)
class PromptInput:
    lenguaje: str
    codigo: str


def build_user_prompt(data: PromptInput) -> str:
    return (
        f"Lenguaje: {data.lenguaje}\n\n"
        "Genera solamente el codigo de pruebas unitarias para esta funcion:\n\n"
        f"{data.codigo}"
    )
