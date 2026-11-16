"""Fusao multimodal tardia (no nivel de decisao) por paciente.

Cada modalidade ja decidiu, com seu proprio detector, o que e anomalo. A fusao
junta essas decisoes de duas formas:

1. **Escore de risco (0-100)**: cada anomalia pesa pela severidade e perde peso
   com o tempo (meia-vida configuravel). Dentro de uma modalidade o retorno e
   decrescente (a mesma piora vista por varios detectores nao conta varias vezes
   cheia); entre modalidades diferentes, a concordancia aumenta o escore. Um
   achado critico das ultimas 2 h garante no minimo o nivel alto, mesmo isolado.
2. **Correlacoes clinicas**: combinacoes de modalidades que, juntas, dizem mais
   que separadas (queda vista pelo acelerometro *e* pela camera; fala arrastada
   *e* relato de fraqueza num lado). Cada correlacao vira um unico alerta
   consolidado, com as evidencias de cada fonte e o protocolo de referencia.

O resultado combate a fadiga de alarme: a equipe ve um paciente com um nivel
de risco e uma explicacao, em vez de uma dezena de alertas soltos.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..config import MONITORAMENTO
from .tipos import Anomalia

PESOS = {"critica": 40.0, "alta": 25.0, "moderada": 12.0, "baixa": 4.0}
NIVEIS = ((70.0, "critico"), (45.0, "alto"), (20.0, "moderado"), (0.0, "baixo"))
JANELA_ANALISE = timedelta(hours=24)
RISCO_ELEVADO = 70.0
PISO_CRITICO = 45.0
PISO_CRITICO_H = 2.0

NOMES_MODALIDADE = {
    "sinais_vitais": "sinais vitais", "prescricao": "prescricao", "movimentacao": "movimentacao",
    "audio": "voz", "texto": "fala (conteudo)", "video": "video", "multimodal": "fusao",
}


@dataclass(frozen=True)
class Regra:
    chave: str
    titulo: str
    severidade: str
    condicoes: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    janela: timedelta
    fonte: str | None = None
    ordenada: bool = False


SINAIS_RESPIRATORIOS = ("parametro_critico_saturacao", "parametro_critico_fr",
                        "desvio_basal_saturacao", "desvio_basal_fr", "tendencia_saturacao",
                        "tendencia_fr")
REGRAS = (
    Regra("queda_confirmada", "Queda confirmada por acelerometro e camera", "critica",
          ((("movimentacao",), ("queda",)), (("video",), ("video_queda",))),
          timedelta(minutes=2)),
    Regra("suspeita_avc", "Suspeita de AVC: fala arrastada e deficit neurologico relatado",
          "critica",
          ((("audio",), ("voz_disartria",)), (("texto",), ("termo_critico_deficit_neurologico",))),
          timedelta(hours=6), fonte="PROT-NEU-003"),
    Regra("piora_respiratoria", "Piora respiratoria nos sinais vitais e na fala", "critica",
          ((("sinais_vitais",), SINAIS_RESPIRATORIOS),
           (("audio", "texto"), ("voz_dispneia", "termo_critico_dispneia"))),
          timedelta(hours=12), fonte="PROT-PNM-006"),
    Regra("febre_apos_suspensao", "Febre apos suspensao de antimicrobiano", "alta",
          ((("prescricao",), ("suspensao_sem_substituto",)),
           (("sinais_vitais",), ("desvio_basal_temperatura", "tendencia_temperatura",
                                 "parametro_critico_temperatura"))),
          timedelta(hours=24), fonte="PROT-ATB-003", ordenada=True),
    Regra("declinio_funcional", "Fadiga na fala e imobilidade no leito", "alta",
          ((("audio", "texto"), ("voz_fadiga", "termo_critico_fadiga")),
           (("movimentacao",), ("imobilidade_prolongada",))),
          timedelta(hours=24)),
    Regra("risco_queda_noturna", "Agitacao noturna seguida de saida do leito", "alta",
          ((("movimentacao",), ("agitacao_noturna",)), (("movimentacao",), ("saida_leito_noturna",))),
          timedelta(hours=3), ordenada=True),
)


def instante(anomalia: dict[str, Any]) -> datetime | None:
    referencia = anomalia.get("fim") or anomalia.get("inicio") or anomalia.get("detectada_em")
    try:
        return datetime.fromisoformat(referencia) if referencia else None
    except ValueError:
        return None


@dataclass
class Correlacao:
    regra: Regra
    anomalias: list[dict[str, Any]]
    inicio: datetime
    fim: datetime

    def descricao(self) -> str:
        partes = [f"{NOMES_MODALIDADE.get(a['modalidade'], a['modalidade'])}: {a['descricao']}"
                  for a in self.anomalias]
        fonte = f" (ver {self.regra.fonte})" if self.regra.fonte else ""
        return f"{self.regra.titulo}{fonte}. Evidencias: " + " | ".join(partes)


@dataclass
class AvaliacaoPaciente:
    prontuario: str
    escore: float
    nivel: str
    contribuicoes: dict[str, float]
    correlacoes: list[Correlacao]
    anomalias: list[dict[str, Any]] = field(default_factory=list)
    referencia: datetime | None = None

    @property
    def modalidades(self) -> list[str]:
        return [m for m, v in sorted(self.contribuicoes.items(), key=lambda x: -x[1]) if v >= 1]

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "prontuario": self.prontuario, "escore": round(self.escore, 1), "nivel": self.nivel,
            "contribuicoes": {k: round(v, 1) for k, v in self.contribuicoes.items()},
            "correlacoes": [{"regra": c.regra.chave, "titulo": c.regra.titulo,
                             "severidade": c.regra.severidade, "fonte": c.regra.fonte,
                             "anomalias": [a.get("id") for a in c.anomalias]}
                            for c in self.correlacoes],
            "anomalias": len(self.anomalias),
        }


def nivel(escore: float) -> str:
    return next(rotulo for limite, rotulo in NIVEIS if escore >= limite)


def _casa(anomalia: dict[str, Any], condicao: tuple[tuple[str, ...], tuple[str, ...]]) -> bool:
    modalidades, prefixos = condicao
    return anomalia["modalidade"] in modalidades and any(
        anomalia["tipo"].startswith(p) for p in prefixos)


def correlacionar(anomalias: list[dict[str, Any]], regras=REGRAS) -> list[Correlacao]:
    """Para cada regra, a combinacao mais recente de anomalias que a satisfaz na janela."""
    correlacoes = []
    for regra in regras:
        grupos = [[a for a in anomalias if _casa(a, c) and instante(a)] for c in regra.condicoes]
        if not all(grupos):
            continue
        melhor = None
        for ancora in sorted(grupos[0], key=instante, reverse=True):
            t0 = instante(ancora)
            escolhidas = [ancora]
            for grupo in grupos[1:]:
                candidatas = [a for a in grupo if a is not ancora
                              and abs(instante(a) - t0) <= regra.janela
                              and (not regra.ordenada or instante(a) >= t0)]
                if not candidatas:
                    break
                escolhidas.append(min(candidatas, key=lambda a: abs(instante(a) - t0)))
            else:
                tempos = [instante(a) for a in escolhidas]
                melhor = Correlacao(regra, escolhidas, min(tempos), max(tempos))
                break
        if melhor:
            correlacoes.append(melhor)
    return correlacoes


def escore(anomalias: list[dict[str, Any]], referencia: datetime,
           meia_vida_h: float | None = None) -> tuple[float, dict[str, float]]:
    meia_vida = meia_vida_h or MONITORAMENTO.meia_vida_risco_h
    por_modalidade: dict[str, list[float]] = defaultdict(list)
    for anomalia in anomalias:
        if anomalia["modalidade"] == "multimodal":
            continue
        momento = instante(anomalia) or referencia
        idade_h = max((referencia - momento).total_seconds() / 3600, 0.0)
        por_modalidade[anomalia["modalidade"]].append(
            PESOS.get(anomalia["severidade"], 0.0) * 0.5 ** (idade_h / meia_vida))
    contribuicoes = {}
    for modalidade, pesos in por_modalidade.items():
        ordenados = sorted(pesos, reverse=True)
        contribuicoes[modalidade] = ordenados[0] + 0.3 * sum(ordenados[1:3]) + 0.1 * sum(ordenados[3:])
    relevantes = sum(1 for v in contribuicoes.values() if v >= PESOS["moderada"] / 2)
    total = sum(contribuicoes.values()) * (1 + 0.15 * max(relevantes - 1, 0))
    criticas_recentes = [a for a in anomalias if a["severidade"] == "critica"
                         and a["modalidade"] != "multimodal"
                         and referencia - (instante(a) or referencia) <= timedelta(hours=PISO_CRITICO_H)]
    if criticas_recentes:
        total = max(total, PISO_CRITICO)
    return min(total, 100.0), contribuicoes


def avaliar_paciente(prontuario: str, anomalias: list[dict[str, Any]], referencia: datetime,
                     janela: timedelta = JANELA_ANALISE) -> AvaliacaoPaciente:
    recentes = [a for a in anomalias
                if (instante(a) or referencia) >= referencia - janela
                and (instante(a) or referencia) <= referencia + timedelta(minutes=5)]
    pontos, contribuicoes = escore(recentes, referencia)
    return AvaliacaoPaciente(prontuario, pontos, nivel(pontos), contribuicoes,
                             correlacionar([a for a in recentes if a["modalidade"] != "multimodal"]),
                             recentes, referencia)


def avaliar_todos(banco, referencia: datetime | None = None) -> list[AvaliacaoPaciente]:
    from .relogio import agora

    referencia = referencia or agora(banco)
    por_paciente: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anomalia in banco.listar_anomalias(limite=5000):
        if anomalia.get("prontuario"):
            por_paciente[anomalia["prontuario"]].append(anomalia)
    avaliacoes = [avaliar_paciente(p, lista, referencia) for p, lista in por_paciente.items()]
    return sorted(avaliacoes, key=lambda a: -a.escore)


def anomalias_consolidadas(avaliacao: AvaliacaoPaciente) -> list[Anomalia]:
    """Um alerta por correlacao; e um de risco elevado se nao houver correlacao que o explique."""
    saida = []
    for correlacao in avaliacao.correlacoes:
        saida.append(Anomalia(
            modalidade="multimodal", tipo=f"fusao_{correlacao.regra.chave}",
            severidade=correlacao.regra.severidade, descricao=correlacao.descricao(),
            prontuario=avaliacao.prontuario, inicio=correlacao.inicio.isoformat(timespec="seconds"),
            fim=correlacao.fim.isoformat(timespec="seconds"), escore=avaliacao.escore / 100,
            detector="fusao_tardia",
            evidencias={"regra": correlacao.regra.chave, "fonte": correlacao.regra.fonte,
                        "anomalias": [a.get("id") for a in correlacao.anomalias],
                        "modalidades": sorted({a["modalidade"] for a in correlacao.anomalias}),
                        "escore_paciente": round(avaliacao.escore, 1)},
        ))
    if not saida and avaliacao.escore >= RISCO_ELEVADO and avaliacao.referencia:
        principais = sorted(avaliacao.anomalias, key=lambda a: -PESOS.get(a["severidade"], 0))[:3]
        modalidades = ", ".join(NOMES_MODALIDADE.get(m, m) for m in avaliacao.modalidades)
        saida.append(Anomalia(
            modalidade="multimodal", tipo="fusao_risco_elevado", severidade="alta",
            descricao=(f"Risco multimodal {avaliacao.escore:.0f}/100 ({modalidades}). "
                       "Principais achados: " + " | ".join(a["descricao"] for a in principais)),
            prontuario=avaliacao.prontuario,
            inicio=avaliacao.referencia.isoformat(timespec="seconds"),
            fim=avaliacao.referencia.isoformat(timespec="seconds"), escore=avaliacao.escore / 100,
            detector="fusao_tardia",
            evidencias={"contribuicoes": {k: round(v, 1) for k, v in avaliacao.contribuicoes.items()},
                        "anomalias": [a.get("id") for a in principais]},
        ))
    return saida


def consolidar(banco, central, referencia: datetime | None = None) -> list[tuple[AvaliacaoPaciente, list]]:
    """Avalia todos os pacientes e envia os alertas consolidados a central."""
    resultados = []
    for avaliacao in avaliar_todos(banco, referencia):
        consolidadas = anomalias_consolidadas(avaliacao)
        resultados.append((avaliacao, central.processar(consolidadas) if consolidadas else []))
    return resultados
