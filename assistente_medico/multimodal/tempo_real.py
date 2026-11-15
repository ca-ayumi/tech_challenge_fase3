"""Monitoramento em fluxo: leituras chegam uma a uma e sao avaliadas na chegada.

Os detectores sao *online*: guardam o estado necessario (janela recente,
linha de base do paciente) e decidem a cada leitura, sem reprocessar a serie
inteira. E o mesmo contrato que um consumidor de fila (Event Hubs, Kafka, MQTT)
usaria em producao; aqui a fonte e um reprodutor que respeita o intervalo real
entre as leituras, opcionalmente acelerado para demonstracao.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from .alertas import SUPRIMIDO, CentralAlertas, ResultadoAlerta
from .tipos import Anomalia


@dataclass(frozen=True)
class Leitura:
    prontuario: str
    modalidade: str
    instante: datetime
    valores: dict[str, Any]


class DetectorOnline(Protocol):
    nome: str
    modalidades: tuple[str, ...]

    def atualizar(self, leitura: Leitura) -> list[Anomalia]: ...


@dataclass
class ResumoMonitoramento:
    leituras: int = 0
    anomalias: int = 0
    alertas: list[ResultadoAlerta] = field(default_factory=list)
    suprimidos: int = 0
    abaixo_do_limiar: int = 0

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "leituras": self.leituras,
            "anomalias": self.anomalias,
            "alertas_emitidos": len(self.alertas),
            "suprimidos": self.suprimidos,
            "abaixo_do_limiar": self.abaixo_do_limiar,
        }


class Monitor:
    def __init__(self, central: CentralAlertas, detectores: Sequence[DetectorOnline],
                 analise_id: int | None = None) -> None:
        self.central = central
        self.detectores = list(detectores)
        self.analise_id = analise_id

    def processar(self, leitura: Leitura) -> list[ResultadoAlerta]:
        anomalias: list[Anomalia] = []
        for detector in self.detectores:
            if leitura.modalidade in detector.modalidades:
                anomalias.extend(detector.atualizar(leitura))
        if not anomalias:
            return []
        return self.central.processar(anomalias, analise_id=self.analise_id)

    def reproduzir(self, leituras: Iterable[Leitura], aceleracao: float | None = None,
                   ao_processar: Callable[[Leitura, list[ResultadoAlerta]], None] | None = None,
                   dormir: Callable[[float], None] = time.sleep) -> ResumoMonitoramento:
        """Processa o fluxo em ordem.

        ``aceleracao`` None processa o mais rapido possivel; 60 reproduz uma hora
        de dados em um minuto, respeitando o espacamento real entre leituras.
        """
        resumo = ResumoMonitoramento()
        anterior: datetime | None = None
        for leitura in leituras:
            if aceleracao and anterior is not None:
                espera = (leitura.instante - anterior).total_seconds() / aceleracao
                if espera > 0:
                    dormir(espera)
            anterior = leitura.instante

            resultados = self.processar(leitura)
            resumo.leituras += 1
            resumo.anomalias += len(resultados)
            for resultado in resultados:
                if resultado.emitido:
                    resumo.alertas.append(resultado)
                elif resultado.decisao == SUPRIMIDO:
                    resumo.suprimidos += 1
                else:
                    resumo.abaixo_do_limiar += 1
            if ao_processar:
                ao_processar(leitura, resultados)
        return resumo
