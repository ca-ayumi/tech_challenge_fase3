"""Verificacao das credenciais e da disponibilidade dos servicos Azure."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from ...config import AZURE, ConfigAzure
from .cliente import ErroAzure, requisitar

VERSAO_API_LINGUAGEM = "2023-04-01"


@dataclass(frozen=True)
class StatusServico:
    servico: str
    configurado: bool
    disponivel: bool
    detalhe: str
    latencia_ms: int | None = None

    def como_dicionario(self) -> dict:
        return asdict(self)


def _verificar_fala(config: ConfigAzure) -> StatusServico:
    if not config.fala_configurada:
        return StatusServico("speech", False, False,
                             "AZURE_SPEECH_KEY e AZURE_SPEECH_REGION nao definidos")
    inicio = time.perf_counter()
    try:
        requisitar(
            "POST",
            f"https://{config.regiao_fala}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
            config.chave_fala, timeout_s=config.timeout_s, tentativas=1,
        )
    except ErroAzure as erro:
        return StatusServico("speech", True, False, str(erro))
    return StatusServico("speech", True, True, f"regiao {config.regiao_fala}",
                         int((time.perf_counter() - inicio) * 1000))


def _verificar_linguagem(config: ConfigAzure) -> StatusServico:
    if not config.linguagem_configurada:
        return StatusServico("language", False, False,
                             "AZURE_LANGUAGE_KEY e AZURE_LANGUAGE_ENDPOINT nao definidos")
    inicio = time.perf_counter()
    try:
        resposta = requisitar(
            "POST",
            f"{config.endpoint_linguagem}/language/:analyze-text?api-version={VERSAO_API_LINGUAGEM}",
            config.chave_linguagem,
            corpo={"kind": "LanguageDetection",
                   "analysisInput": {"documents": [{"id": "1", "text": "Paciente estavel."}]}},
            timeout_s=config.timeout_s, tentativas=1,
        )
        idioma = resposta.json()["results"]["documents"][0]["detectedLanguage"]["iso6391Name"]
    except ErroAzure as erro:
        return StatusServico("language", True, False, str(erro))
    except (KeyError, IndexError, TypeError, ValueError):
        return StatusServico("language", True, False, "resposta inesperada do servico")
    return StatusServico("language", True, True, f"idioma detectado no teste: {idioma}",
                         int((time.perf_counter() - inicio) * 1000))


def verificar_azure(config: ConfigAzure | None = None) -> list[StatusServico]:
    config = config or AZURE
    return [_verificar_fala(config), _verificar_linguagem(config)]
