"""Cliente HTTP minimo para as APIs REST dos servicos cognitivos da Azure.

Usa apenas a biblioteca padrao: o diagnostico e o CI nao dependem dos SDKs.
Repete a chamada em 429 (limite de taxa da camada gratuita) e 5xx, respeitando
o cabecalho Retry-After quando presente.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

REPETIVEIS = {429, 500, 502, 503, 504}


class ErroAzure(RuntimeError):
    def __init__(self, mensagem: str, status: int | None = None) -> None:
        super().__init__(mensagem)
        self.status = status


@dataclass(frozen=True)
class RespostaAzure:
    status: int
    corpo: bytes
    cabecalhos: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.corpo.decode("utf-8")) if self.corpo else None


def requisitar(metodo: str, url: str, chave: str, corpo: Any = None,
               cabecalhos: dict[str, str] | None = None, timeout_s: float = 30.0,
               tentativas: int = 3,
               dormir: Callable[[float], None] = time.sleep) -> RespostaAzure:
    """Chama a API e devolve a resposta; levanta ErroAzure em falha definitiva."""
    todos = {"Ocp-Apim-Subscription-Key": chave, **(cabecalhos or {})}
    dados: bytes | None
    if corpo is None:
        dados = b"" if metodo == "POST" else None
    elif isinstance(corpo, bytes):
        dados = corpo
    else:
        dados = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        todos.setdefault("Content-Type", "application/json")

    for tentativa in range(1, tentativas + 1):
        requisicao = urllib.request.Request(url, data=dados, headers=todos, method=metodo)
        try:
            with urllib.request.urlopen(requisicao, timeout=timeout_s) as resposta:
                return RespostaAzure(resposta.status, resposta.read(), dict(resposta.headers))
        except urllib.error.HTTPError as erro:
            detalhe = erro.read().decode("utf-8", "replace")[:300]
            if erro.code in REPETIVEIS and tentativa < tentativas:
                espera = erro.headers.get("Retry-After") if erro.headers else None
                dormir(float(espera) if espera and espera.isdigit() else 2.0 ** tentativa)
                continue
            raise ErroAzure(f"HTTP {erro.code}: {detalhe}", status=erro.code) from erro
        except (urllib.error.URLError, TimeoutError) as erro:
            if tentativa < tentativas:
                dormir(2.0 ** tentativa)
                continue
            raise ErroAzure(f"Falha de conexao: {erro}") from erro
    raise ErroAzure("Tentativas esgotadas.")  # pragma: no cover
