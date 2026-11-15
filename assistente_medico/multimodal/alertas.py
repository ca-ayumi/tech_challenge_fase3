"""Central de alertas do monitoramento multimodal.

Todo achado de qualquer modalidade passa por aqui. A central:

1. grava a anomalia (sempre, para rastreabilidade);
2. decide se ela vira alerta: severidade minima e supressao de repeticao;
3. registra o alerta no painel institucional (tabela ``alertas``);
4. notifica a equipe pelos canais configurados (arquivo, webhook);
5. audita a decisao, inclusive quando o alerta foi suprimido.

A supressao combate a fadiga de alarme: a mesma anomalia, no mesmo paciente,
dentro da janela configurada, nao gera um segundo alerta. A excecao e o
agravamento: se a nova ocorrencia e mais grave que a ja alertada, o alerta sai.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from ..auditoria import Auditoria
from ..config import CAMINHOS, MONITORAMENTO
from ..dados.banco import BancoHospital
from .tipos import Anomalia, mais_grave, nivel_severidade

logger = logging.getLogger("assistente_medico.multimodal.alertas")

EMITIDO = "emitido"
SUPRIMIDO = "suprimido_repeticao"
ABAIXO_DO_LIMIAR = "abaixo_do_limiar"


class Notificador(Protocol):
    nome: str

    def notificar(self, alerta: dict[str, Any]) -> None: ...


class NotificadorArquivo:
    """Grava cada alerta em JSONL, para consumo por sistema externo (fila, SIEM, BI)."""

    nome = "arquivo"

    def __init__(self, caminho: Path | None = None) -> None:
        self.caminho = Path(caminho or CAMINHOS.logs / "alertas.jsonl")
        self.caminho.parent.mkdir(parents=True, exist_ok=True)

    def notificar(self, alerta: dict[str, Any]) -> None:
        with self.caminho.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(alerta, ensure_ascii=False, default=str) + "\n")


class NotificadorWebhook:
    """Envia o alerta por HTTP POST.

    O corpo traz ``text``, aceito pelos webhooks de entrada do Slack e do Teams,
    e o alerta completo em ``alerta``, para integracoes proprias.
    """

    nome = "webhook"

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url = url
        self.timeout_s = timeout_s

    def notificar(self, alerta: dict[str, Any]) -> None:
        texto = (f"[{alerta['prioridade'].upper()}] prontuario {alerta.get('prontuario') or '-'}: "
                 f"{alerta['mensagem']}")
        corpo = json.dumps({"text": texto, "alerta": alerta}, ensure_ascii=False,
                           default=str).encode("utf-8")
        requisicao = urllib.request.Request(
            self.url, data=corpo, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(requisicao, timeout=self.timeout_s):
            pass


def notificadores_padrao() -> list[Notificador]:
    canais: list[Notificador] = [NotificadorArquivo()]
    if MONITORAMENTO.webhook_alertas:
        canais.append(NotificadorWebhook(MONITORAMENTO.webhook_alertas))
    return canais


@dataclass(frozen=True)
class ResultadoAlerta:
    """O que a central fez com uma anomalia."""

    anomalia: Anomalia
    anomalia_id: int
    decisao: str
    alerta_id: int | None = None
    canais: tuple[str, ...] = ()

    @property
    def emitido(self) -> bool:
        return self.decisao == EMITIDO


class CentralAlertas:
    def __init__(self, banco: BancoHospital,
                 auditoria: Auditoria | None = None,
                 notificadores: Sequence[Notificador] | None = None,
                 severidade_minima: str | None = None,
                 janela_supressao: timedelta | None = None) -> None:
        self.banco = banco
        self.auditoria = auditoria or Auditoria(banco=banco)
        self.notificadores = list(notificadores_padrao() if notificadores is None
                                  else notificadores)
        self.severidade_minima = severidade_minima or MONITORAMENTO.severidade_minima_alerta
        nivel_severidade(self.severidade_minima)
        self.janela_supressao = janela_supressao or timedelta(
            minutes=MONITORAMENTO.janela_supressao_min)

    def _decidir(self, anomalia: Anomalia) -> str:
        if nivel_severidade(anomalia.severidade) > nivel_severidade(self.severidade_minima):
            return ABAIXO_DO_LIMIAR
        anterior = self.banco.ultima_anomalia_alertada(anomalia.prontuario, anomalia.tipo)
        if anterior is None:
            return EMITIDO
        if mais_grave(anomalia.severidade, anterior["severidade"]):
            return EMITIDO
        referencia_anterior = anterior["fim"] or anterior["inicio"] or anterior["detectada_em"]
        atual = anomalia.instante or datetime.now()
        try:
            intervalo = atual - datetime.fromisoformat(referencia_anterior)
        except (TypeError, ValueError):
            return EMITIDO
        return SUPRIMIDO if intervalo < self.janela_supressao else EMITIDO

    def _notificar(self, alerta: dict[str, Any]) -> tuple[str, ...]:
        entregues = []
        for canal in self.notificadores:
            try:
                canal.notificar(alerta)
                entregues.append(canal.nome)
            except Exception:  # notificacao nunca derruba o monitoramento
                logger.exception("Falha ao notificar pelo canal %s", canal.nome)
        return tuple(entregues)

    def processar(self, anomalias: Iterable[Anomalia], analise_id: int | None = None,
                  interacao: str | None = None) -> list[ResultadoAlerta]:
        interacao = interacao or f"MON-{datetime.now():%Y%m%d}"
        resultados = []
        for anomalia in anomalias:
            decisao = self._decidir(anomalia)
            alerta_id = None
            canais: tuple[str, ...] = ()
            if decisao == EMITIDO:
                alerta_id = self.banco.registrar_alerta(
                    prioridade=anomalia.prioridade,
                    categoria=f"anomalia_{anomalia.modalidade}",
                    mensagem=anomalia.descricao,
                    prontuario=anomalia.prontuario,
                    fonte=f"monitoramento/{anomalia.modalidade}/{anomalia.detector or anomalia.tipo}",
                    interacao=interacao,
                )
            anomalia_id = self.banco.registrar_anomalia(
                anomalia.como_dicionario(), analise_id=analise_id, alerta_id=alerta_id)
            if alerta_id is not None:
                canais = self._notificar({
                    "alerta_id": alerta_id,
                    "anomalia_id": anomalia_id,
                    "prioridade": anomalia.prioridade,
                    "prontuario": anomalia.prontuario,
                    "modalidade": anomalia.modalidade,
                    "tipo": anomalia.tipo,
                    "mensagem": anomalia.descricao,
                    "inicio": anomalia.inicio,
                    "fim": anomalia.fim,
                    "emitido_em": datetime.now().isoformat(timespec="seconds"),
                })
            self.auditoria.registrar(
                interacao, f"anomalia_{decisao}",
                {"anomalia_id": anomalia_id, "alerta_id": alerta_id,
                 "modalidade": anomalia.modalidade, "tipo": anomalia.tipo,
                 "severidade": anomalia.severidade, "detector": anomalia.detector,
                 "canais": list(canais)},
                prontuario=anomalia.prontuario,
            )
            resultados.append(ResultadoAlerta(anomalia, anomalia_id, decisao, alerta_id, canais))
        return resultados

    def reconhecer(self, alerta_id: int, profissional: str, perfil: str | None = None) -> bool:
        """Registra que a equipe viu o alerta: fecha o ciclo deteccao -> resposta."""
        reconhecido = self.banco.reconhecer_alerta(alerta_id)
        if reconhecido:
            self.auditoria.registrar(
                f"ALERTA-{alerta_id}", "alerta_reconhecido",
                {"alerta_id": alerta_id}, usuario=profissional, perfil=perfil,
            )
        return reconhecido
