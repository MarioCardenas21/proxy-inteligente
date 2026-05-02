from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    lenguaje: str = Field(..., min_length=1)
    codigo: str = Field(..., min_length=1)

    @field_validator("lenguaje", "codigo")
    @classmethod
    def validate_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("El campo no puede estar vacio.")
        return value.strip()


class GenerateTestResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    prueba: str
    latencia_ms: int
