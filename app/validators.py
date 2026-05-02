import re


def strip_markdown_fences(text: str | None) -> str:
    if text is None:
        return ""

    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def looks_like_code(text: str) -> bool:
    cleaned = strip_markdown_fences(text)
    return bool(cleaned)
