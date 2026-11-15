"""Azure Language: analise de sentimento e Text Analytics for Health.

O texto que chega aqui ja passou pelo anonimizador: nomes, documentos e
contatos citados na consulta viram pseudonimos antes de sair do hospital.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ...config import AZURE, ConfigAzure
from .cliente import ErroAzure, requisitar

VERSAO_API = "2023-04-01"


@dataclass(frozen=True)
class EntidadeSaude:
    texto: str
    categoria: str
    confianca: float
    certeza: str | None = None
    condicionalidade: str | None = None
    associacao: str | None = None
    codigos: tuple[str, ...] = field(default_factory=tuple)
    deslocamento: int = 0

    @property
    def negada(self) -> bool:
        return self.certeza in {"negative", "negativePossible"}

    def como_dicionario(self) -> dict[str, Any]:
        dados = asdict(self)
        dados["codigos"] = list(self.codigos)
        dados["negada"] = self.negada
        return dados


def _exigir(config: ConfigAzure) -> None:
    if not config.linguagem_configurada:
        raise ErroAzure("Azure Language nao configurado (AZURE_LANGUAGE_KEY / AZURE_LANGUAGE_ENDPOINT).")


def analisar_sentimento(texto: str, config: ConfigAzure | None = None,
                        idioma: str = "pt") -> dict[str, Any]:
    config = config or AZURE
    _exigir(config)
    resposta = requisitar(
        "POST", f"{config.endpoint_linguagem}/language/:analyze-text?api-version={VERSAO_API}",
        config.chave_linguagem, timeout_s=config.timeout_s,
        corpo={"kind": "SentimentAnalysis", "parameters": {"opinionMining": True},
               "analysisInput": {"documents": [{"id": "1", "language": idioma, "text": texto}]}},
    )
    documento = resposta.json()["results"]["documents"][0]
    opinioes = []
    for frase in documento.get("sentences", []):
        for alvo in frase.get("targets", []) or []:
            opinioes.append({"alvo": alvo["text"], "sentimento": alvo["sentiment"]})
    return {
        "sentimento": documento["sentiment"],
        "confianca": documento["confidenceScores"],
        "frases": [{"texto": f["text"], "sentimento": f["sentiment"],
                    "confianca": f["confidenceScores"]} for f in documento.get("sentences", [])],
        "opinioes": opinioes,
    }


def analisar_saude(texto: str, config: ConfigAzure | None = None, idioma: str = "pt",
                   dormir: Callable[[float], None] = time.sleep,
                   espera_maxima_s: float = 60.0) -> list[EntidadeSaude]:
    """Text Analytics for Health (operacao assincrona: envia o job e consulta ate concluir)."""
    config = config or AZURE
    _exigir(config)
    envio = requisitar(
        "POST", f"{config.endpoint_linguagem}/language/analyze-text/jobs?api-version={VERSAO_API}",
        config.chave_linguagem, timeout_s=config.timeout_s,
        corpo={"analysisInput": {"documents": [{"id": "1", "language": idioma, "text": texto}]},
               "tasks": [{"kind": "Healthcare", "taskName": "saude"}]},
    )
    local = envio.cabecalhos.get("Operation-Location") or envio.cabecalhos.get("operation-location")
    if not local:
        raise ErroAzure("Resposta sem Operation-Location ao criar o job de saude.")

    decorrido = 0.0
    while True:
        dormir(1.5)
        decorrido += 1.5
        estado = requisitar("GET", local, config.chave_linguagem, timeout_s=config.timeout_s).json()
        if estado["status"] in {"succeeded", "partiallyCompleted"}:
            break
        if estado["status"] in {"failed", "cancelled"} or decorrido > espera_maxima_s:
            raise ErroAzure(f"Job de saude terminou com status {estado['status']}.")

    resultados = estado["tasks"]["items"][0].get("results", {})
    if resultados.get("errors"):
        raise ErroAzure(f"Erro no Text Analytics for Health: {resultados['errors'][0]}")
    entidades = []
    for documento in resultados.get("documents", []):
        for entidade in documento.get("entities", []):
            assercao = entidade.get("assertion") or {}
            entidades.append(EntidadeSaude(
                texto=entidade["text"], categoria=entidade["category"],
                confianca=float(entidade.get("confidenceScore", 0.0)),
                certeza=assercao.get("certainty"),
                condicionalidade=assercao.get("conditionality"),
                associacao=assercao.get("association"),
                codigos=tuple(f"{link['dataSource']}:{link['id']}" for link in entidade.get("links", []) or []),
                deslocamento=int(entidade.get("offset", 0)),
            ))
    return entidades
