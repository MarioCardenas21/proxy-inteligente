from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class InferenceLog:
    lenguaje: str
    latencia_ms: int
    status: str


def log_inference(entry: InferenceLog) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(
        f"{timestamp} | {entry.lenguaje} | {entry.latencia_ms} ms | {entry.status}",
        flush=True,
    )
