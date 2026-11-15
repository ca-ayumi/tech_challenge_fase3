"""Padroes de movimentacao do paciente durante a internacao.

Duas fontes, mesma modalidade:

- **sensor de leito** (uma leitura a cada 15 min): fracao do tempo em
  movimento e posicao. Detecta imobilidade prolongada (risco de lesao por
  pressao: o reposicionamento esperado e a cada 2 h), agitacao noturna
  (rastreio de delirium) e saida do leito a noite (risco de queda).
- **acelerometro** (dezenas de leituras por segundo): detecta queda por pico
  de aceleracao. O limiar de 3,0 g foi calibrado nas sequencias impares do UR
  Fall e avaliado nas pares. A postura apos o impacto vai como evidencia, nao
  como condicao: na calibracao ela nao melhorou a separacao.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from ..tempo_real import Leitura
from ..tipos import Anomalia
from .episodios import GerenciadorEpisodios

POSICOES_NO_LEITO = {"decubito_dorsal", "decubito_lateral_esquerdo", "decubito_lateral_direito"}


@dataclass(frozen=True)
class ParametrosMovimentacao:
    imobilidade_moderada: timedelta = timedelta(hours=3)
    imobilidade_alta: timedelta = timedelta(hours=5)
    hora_inicio_noite: int = 22
    hora_fim_noite: int = 6
    janela_agitacao: timedelta = timedelta(minutes=45)
    fator_agitacao: float = 3.0
    atividade_minima_agitacao: float = 0.35
    min_amostras_base_noturna: int = 8
    limiar_impacto_g: float = 3.0
    espera_pos_impacto: timedelta = timedelta(milliseconds=500)
    janela_postura: tuple[float, float] = (0.2, 0.5)
    limiar_vertical_g: float = 0.6
    refratario_queda: timedelta = timedelta(seconds=10)


@dataclass
class _Estado:
    episodios: GerenciadorEpisodios = field(default_factory=lambda: GerenciadorEpisodios(1))
    posicao: str | None = None
    posicao_desde: datetime | None = None
    noite_base: deque = field(default_factory=lambda: deque(maxlen=200))
    noite_recente: deque = field(default_factory=deque)
    pico: tuple[datetime, float] | None = None
    pos_impacto: list[tuple[datetime, float | None]] = field(default_factory=list)
    ultima_queda: datetime | None = None


def _iso(instante: datetime) -> str:
    return instante.isoformat(timespec="seconds")


def _horas(intervalo: timedelta) -> str:
    minutos = int(intervalo.total_seconds() // 60)
    return f"{minutos // 60} h {minutos % 60:02d} min"


class DetectorMovimentacao:
    nome = "movimentacao"
    modalidades = ("movimentacao",)

    def __init__(self, parametros: ParametrosMovimentacao | None = None,
                 risco_queda: Collection[str] = ()) -> None:
        self.parametros = parametros or ParametrosMovimentacao()
        self.risco_queda = set(risco_queda)
        self._estados: dict[str, _Estado] = {}

    def _noturno(self, instante: datetime) -> bool:
        p = self.parametros
        return instante.hour >= p.hora_inicio_noite or instante.hour < p.hora_fim_noite

    def atualizar(self, leitura: Leitura) -> list[Anomalia]:
        estado = self._estados.setdefault(leitura.prontuario, _Estado())
        if leitura.valores.get("aceleracao_g") is not None:
            return self._queda(leitura, estado)
        achados: list[Anomalia] = []
        if leitura.valores.get("posicao"):
            achados.extend(self._posicao(leitura, estado))
        if leitura.valores.get("atividade") is not None:
            achados.extend(self._agitacao(leitura, estado))
        return achados

    def _anomalia(self, leitura: Leitura, inicio: datetime, tipo: str, severidade: str,
                  descricao: str, detector: str, evidencias: dict[str, Any],
                  escore: float = 1.0) -> Anomalia:
        return Anomalia(
            modalidade="movimentacao", tipo=tipo, severidade=severidade, descricao=descricao,
            prontuario=leitura.prontuario, inicio=_iso(inicio), fim=_iso(leitura.instante),
            escore=round(escore, 3), detector=detector, evidencias=evidencias,
        )

    def _posicao(self, leitura: Leitura, estado: _Estado) -> list[Anomalia]:
        p = self.parametros
        posicao = leitura.valores["posicao"]
        achados = []
        if posicao != estado.posicao:
            estado.posicao, estado.posicao_desde = posicao, leitura.instante
            estado.episodios.negativo("imobilidade")

        if posicao in POSICOES_NO_LEITO and estado.posicao_desde is not None:
            parado = leitura.instante - estado.posicao_desde
            if parado >= p.imobilidade_moderada:
                severidade = "alta" if parado >= p.imobilidade_alta else "moderada"
                inicio = estado.episodios.positivo("imobilidade", leitura.instante, severidade)
                if inicio is not None:
                    achados.append(self._anomalia(
                        leitura, estado.posicao_desde, "imobilidade_prolongada", severidade,
                        f"Paciente na mesma posicao ({posicao.replace('_', ' ')}) ha "
                        f"{_horas(parado)}; o reposicionamento esperado e a cada 2 h "
                        "(risco de lesao por pressao)",
                        "regra_reposicionamento",
                        {"posicao": posicao, "minutos_sem_mudanca": int(parado.total_seconds() // 60)},
                    ))

        if posicao == "fora_do_leito" and self._noturno(leitura.instante):
            risco = leitura.prontuario in self.risco_queda
            severidade = "alta" if risco else "moderada"
            inicio = estado.episodios.positivo("saida_noturna", leitura.instante, severidade)
            if inicio is not None:
                complemento = " em paciente com risco de queda" if risco else ""
                achados.append(self._anomalia(
                    leitura, inicio, "saida_leito_noturna", severidade,
                    f"Saida do leito as {leitura.instante:%H:%M}{complemento}",
                    "sensor_leito", {"risco_queda": risco},
                ))
        else:
            estado.episodios.negativo("saida_noturna")
        return achados

    def _agitacao(self, leitura: Leitura, estado: _Estado) -> list[Anomalia]:
        p = self.parametros
        atividade = float(leitura.valores["atividade"])
        if not self._noturno(leitura.instante):
            estado.noite_recente.clear()
            estado.episodios.negativo("agitacao")
            return []

        estado.noite_recente.append((leitura.instante, atividade))
        while estado.noite_recente and leitura.instante - estado.noite_recente[0][0] >= p.janela_agitacao:
            estado.noite_recente.popleft()
        base = median(estado.noite_base) if len(estado.noite_base) >= p.min_amostras_base_noturna else None
        if not estado.episodios.ativo("agitacao"):
            estado.noite_base.append(atividade)
        if base is None or len(estado.noite_recente) < 3:
            return []

        media_recente = sum(v for _, v in estado.noite_recente) / len(estado.noite_recente)
        limiar = max(p.fator_agitacao * max(base, 0.05), p.atividade_minima_agitacao)
        if media_recente < limiar:
            estado.episodios.negativo("agitacao")
            return []
        inicio = estado.episodios.positivo("agitacao", leitura.instante, "moderada")
        if inicio is None:
            return []
        return [self._anomalia(
            leitura, estado.noite_recente[0][0], "agitacao_noturna", "moderada",
            f"Atividade noturna {media_recente:.0%} do tempo nos ultimos "
            f"{int(p.janela_agitacao.total_seconds() // 60)} min, {media_recente / max(base, 0.05):.1f}x "
            "a linha de base noturna do paciente (rastrear delirium)",
            "linha_base_noturna",
            {"atividade_media": round(media_recente, 3), "base_noturna": round(base, 3)},
            escore=min(media_recente, 1.0),
        )]

    def _queda(self, leitura: Leitura, estado: _Estado) -> list[Anomalia]:
        p = self.parametros
        aceleracao = float(leitura.valores["aceleracao_g"])
        instante = leitura.instante
        if estado.ultima_queda and instante - estado.ultima_queda < p.refratario_queda:
            return []

        if aceleracao >= p.limiar_impacto_g and (estado.pico is None or aceleracao > estado.pico[1]):
            estado.pico = (instante, aceleracao)
            estado.pos_impacto = []
            return []
        if estado.pico is None:
            return []

        decorrido = (instante - estado.pico[0]).total_seconds()
        if p.janela_postura[0] <= decorrido <= p.janela_postura[1]:
            estado.pos_impacto.append((instante, leitura.valores.get("ay")))
        if instante - estado.pico[0] < p.espera_pos_impacto:
            return []

        momento, pico = estado.pico
        verticais = [abs(v) for _, v in estado.pos_impacto if v is not None]
        if verticais:
            media_vertical = sum(verticais) / len(verticais)
            postura = "horizontal" if media_vertical < p.limiar_vertical_g else "vertical"
        else:
            media_vertical, postura = None, "indeterminada"
        estado.pico, estado.pos_impacto, estado.ultima_queda = None, [], instante
        complemento = {"horizontal": "; corpo na horizontal apos o impacto",
                       "vertical": "; corpo ainda na vertical apos o impacto",
                       "indeterminada": ""}[postura]
        return [Anomalia(
            modalidade="movimentacao", tipo="queda", severidade="critica",
            descricao=f"Provavel queda: impacto de {pico:.1f} g as {momento:%H:%M:%S}{complemento}",
            prontuario=leitura.prontuario, inicio=_iso(momento), fim=_iso(instante),
            escore=round(min(pico / (2 * p.limiar_impacto_g), 1.0), 3),
            detector="pico_aceleracao",
            evidencias={"pico_g": round(pico, 2), "postura_pos_impacto": postura,
                        "vertical_media_g": round(media_vertical, 2) if media_vertical is not None else None},
        )]
