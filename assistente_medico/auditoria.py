"""Registro de auditoria e configuracao de logging.

Cada pergunta feita ao assistente abre uma *interacao* com identificador unico.
Todo passo do fluxo grava um evento nessa interacao: entrada recebida,
guardrail aplicado, paciente resolvido, documentos recuperados, chamada ao
modelo, resposta entregue, decisao humana. A trilha responde as perguntas que
uma auditoria clinica faz: quem perguntou, o que o sistema consultou, em que se
baseou, o que respondeu, e quem validou.

Os eventos vao para dois destinos:

- SQLite (tabela ``auditoria``), para consulta relacional e para ligar a
  interacao a validacao humana correspondente;
- arquivo JSONL em ``logs/auditoria.jsonl``, append-only, para retencao longa e
  ingestao por ferramenta externa.

Textos livres passam pelo anonimizador antes de serem gravados
(PROT-GOV-010 §6 e §8): a trilha guarda pseudonimos, nao identificadores.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CAMINHOS, NOME_ASSISTENTE, SEGURANCA, VERSAO_ASSISTENTE
from .dados.anonimizacao import Anonimizador
from .dados.banco import BancoHospital

FORMATO_LOG = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_configurado = False


def configurar_logging(nivel: int = logging.INFO, arquivo: Path | None = None) -> None:
    """Configura logging para console e arquivo rotativo. Idempotente."""
    global _configurado
    if _configurado:
        return

    CAMINHOS.garantir()
    arquivo = arquivo or CAMINHOS.logs / "assistente.log"
    raiz = logging.getLogger()
    raiz.setLevel(nivel)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FORMATO_LOG))
    console.setLevel(logging.WARNING)

    rotativo = logging.handlers.RotatingFileHandler(
        arquivo, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    rotativo.setFormatter(logging.Formatter(FORMATO_LOG))
    rotativo.setLevel(nivel)

    raiz.addHandler(console)
    raiz.addHandler(rotativo)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _configurado = True


@dataclass
class EventoAuditoria:
    """Um passo registrado dentro de uma interacao."""

    interacao: str
    evento: str
    registrado_em: str
    conteudo: dict[str, Any] = field(default_factory=dict)
    usuario: str | None = None
    perfil: str | None = None
    prontuario: str | None = None

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "interacao": self.interacao,
            "evento": self.evento,
            "registrado_em": self.registrado_em,
            "usuario": self.usuario,
            "perfil": self.perfil,
            "prontuario": self.prontuario,
            "conteudo": self.conteudo,
            "assistente": NOME_ASSISTENTE,
            "versao": VERSAO_ASSISTENTE,
        }


class Auditoria:
    """Grava a trilha de auditoria em SQLite e em arquivo JSONL."""

    CAMPOS_TEXTO = {"pergunta", "resposta", "corpo", "mensagem", "sugestao", "justificativa",
                    "texto", "contexto_paciente"}

    def __init__(self, banco: BancoHospital | None = None,
                 anonimizador: Anonimizador | None = None,
                 caminho_jsonl: Path | None = None,
                 mascarar: bool | None = None) -> None:
        self.banco = banco or BancoHospital()
        self.anonimizador = anonimizador or Anonimizador()
        self.caminho_jsonl = Path(caminho_jsonl or CAMINHOS.logs / "auditoria.jsonl")
        self.caminho_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.mascarar = SEGURANCA.mascarar_pii_nos_logs if mascarar is None else mascarar
        self.logger = logging.getLogger("assistente_medico.auditoria")

    @staticmethod
    def nova_interacao() -> str:
        """Identificador unico da interacao, citavel em qualquer relatorio."""
        return f"INT-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"

    def _mascarar(self, valor: Any, chave: str = "") -> Any:
        if not self.mascarar:
            return valor
        if isinstance(valor, str) and (chave in self.CAMPOS_TEXTO or len(valor) > 40):
            return self.anonimizador.anonimizar_texto(valor)
        if isinstance(valor, dict):
            return {c: self._mascarar(v, c) for c, v in valor.items()}
        if isinstance(valor, list):
            return [self._mascarar(item, chave) for item in valor]
        return valor

    def registrar(self, interacao: str, evento: str, conteudo: dict[str, Any] | None = None,
                  usuario: str | None = None, perfil: str | None = None,
                  prontuario: str | None = None) -> EventoAuditoria:
        """Grava um evento da interacao nos dois destinos."""
        conteudo_tratado = self._mascarar(conteudo or {})
        registro = EventoAuditoria(
            interacao=interacao, evento=evento,
            registrado_em=datetime.now().isoformat(timespec="seconds"),
            conteudo=conteudo_tratado, usuario=usuario, perfil=perfil, prontuario=prontuario,
        )

        try:
            self.banco.registrar_auditoria(
                interacao=interacao, evento=evento, conteudo=conteudo_tratado,
                usuario=usuario, perfil=perfil, prontuario=prontuario,
            )
        except Exception:  # pragma: no cover - auditoria nunca derruba o atendimento
            self.logger.exception("Falha ao gravar auditoria em banco (interacao %s)", interacao)

        try:
            with self.caminho_jsonl.open("a", encoding="utf-8") as arquivo:
                arquivo.write(json.dumps(registro.como_dicionario(), ensure_ascii=False,
                                         default=str) + "\n")
        except Exception:  # pragma: no cover
            self.logger.exception("Falha ao gravar auditoria em arquivo (interacao %s)", interacao)

        self.logger.info("[%s] %s", interacao, evento)
        return registro

    @contextmanager
    def medir(self, interacao: str, evento: str, **campos: Any) -> Iterator[dict[str, Any]]:
        """Mede a duracao de uma etapa e registra o resultado ao final."""
        dados: dict[str, Any] = {}
        inicio = time.perf_counter()
        erro: Exception | None = None
        try:
            yield dados
        except Exception as excecao:
            erro = excecao
            raise
        finally:
            duracao = int((time.perf_counter() - inicio) * 1000)
            conteudo = {**campos, **dados, "duracao_ms": duracao}
            if erro is not None:
                conteudo["erro"] = f"{type(erro).__name__}: {erro}"
            self.registrar(interacao, evento, conteudo)

    def trilha(self, interacao: str) -> list[dict[str, Any]]:
        """Devolve todos os eventos de uma interacao, em ordem."""
        eventos = self.banco.trilha_auditoria(interacao)
        for evento in eventos:
            if isinstance(evento.get("conteudo"), str):
                try:
                    evento["conteudo"] = json.loads(evento["conteudo"])
                except json.JSONDecodeError:
                    pass
        return eventos

    def ultimas_interacoes(self, limite: int = 20) -> list[dict[str, Any]]:
        """Lista as interacoes mais recentes, uma linha por interacao."""
        eventos = self.banco.trilha_auditoria(limite=limite * 12)
        vistas: dict[str, dict[str, Any]] = {}
        for evento in eventos:
            interacao = evento["interacao"]
            if interacao not in vistas:
                vistas[interacao] = {
                    "interacao": interacao,
                    "registrado_em": evento["registrado_em"],
                    "perfil": evento.get("perfil"),
                    "prontuario": evento.get("prontuario"),
                    "eventos": 0,
                }
            vistas[interacao]["eventos"] += 1
        return list(vistas.values())[:limite]


def relatorio_retencao() -> dict[str, Any]:
    """Informa a politica de retencao vigente, exigida pela auditoria clinica."""
    return {
        "retencao_anos": SEGURANCA.retencao_auditoria_anos,
        "mascaramento_pii": SEGURANCA.mascarar_pii_nos_logs,
        "fonte": "PROT-GOV-010 §6",
        "destinos": ["sqlite:auditoria", str(CAMINHOS.logs / "auditoria.jsonl")],
    }
