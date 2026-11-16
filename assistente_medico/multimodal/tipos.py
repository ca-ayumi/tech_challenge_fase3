"""Tipos comuns do monitoramento multimodal.

Todo detector, seja de sinais vitais, prescricao, movimentacao, audio ou video,
devolve ``Anomalia``. E esse contrato unico que permite fundir as modalidades e
encaminhar qualquer achado pelo mesmo caminho de alerta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

MODALIDADES = ("sinais_vitais", "prescricao", "movimentacao", "audio", "video", "texto",
               "multimodal")

SEVERIDADES = ("critica", "alta", "moderada", "baixa")

PRIORIDADE_POR_SEVERIDADE = {
    "critica": "maxima",
    "alta": "alta",
    "moderada": "media",
    "baixa": "informativa",
}


def nivel_severidade(severidade: str) -> int:
    """Quanto menor, mais grave: critica=0 ... baixa=3."""
    try:
        return SEVERIDADES.index(severidade)
    except ValueError as erro:
        raise ValueError(f"Severidade desconhecida: {severidade!r}") from erro


def mais_grave(a: str, b: str) -> bool:
    """Verdadeiro se ``a`` for estritamente mais grave que ``b``."""
    return nivel_severidade(a) < nivel_severidade(b)


@dataclass(frozen=True)
class Anomalia:
    """Achado fora do padrao esperado, em qualquer modalidade."""

    modalidade: str
    tipo: str
    severidade: str
    descricao: str
    prontuario: str | None = None
    inicio: str | None = None
    fim: str | None = None
    escore: float = 1.0
    detector: str = ""
    evidencias: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modalidade not in MODALIDADES:
            raise ValueError(f"Modalidade desconhecida: {self.modalidade!r}")
        nivel_severidade(self.severidade)

    @property
    def prioridade(self) -> str:
        return PRIORIDADE_POR_SEVERIDADE[self.severidade]

    @property
    def instante(self) -> datetime | None:
        """Momento de referencia do achado: o fim do trecho, ou o inicio."""
        referencia = self.fim or self.inicio
        return datetime.fromisoformat(referencia) if referencia else None

    def como_dicionario(self) -> dict[str, Any]:
        return asdict(self)
