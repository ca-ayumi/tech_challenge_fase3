"""Avaliacao dos detectores com anomalias injetadas em dados reais.

Os registros BIDMC nao tem rotulo de anomalia. Para medir sensibilidade e
latencia, cada registro e reproduzido limpo e depois com um evento injetado num
instante conhecido; o evento e "detectado" se alguma anomalia clinica for
emitida depois desse instante. O controle negativo (artefato de sensor) mede o
oposto: o que *nao* deveria virar alarme clinico.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..tempo_real import Leitura
from .fontes import ler_bidmc, registros_bidmc
from .sinais_vitais import DetectorSinaisVitais, ParametrosSinaisVitais

EVENTOS = ("dessaturacao", "taquicardia", "taquipneia", "piora_combinada", "artefato")
EVENTOS_CLINICOS = EVENTOS[:-1]

CONFIGURACOES = {
    "limiar (NEWS2)": frozenset({"limiar"}),
    "+ linha de base": frozenset({"limiar", "basal"}),
    "+ tendencia": frozenset({"limiar", "basal", "tendencia"}),
    "+ Isolation Forest": frozenset({"limiar", "basal", "tendencia", "multivariado"}),
}


@dataclass(frozen=True)
class Injecao:
    evento: str
    inicio: datetime
    magnitude: float


def _rampa(decorrido: float, duracao: float) -> float:
    return min(max(decorrido / duracao, 0.0), 1.0)


def injetar(leituras: Sequence[Leitura], evento: str, gerador: random.Random) -> tuple[
        list[Leitura], Injecao]:
    """Devolve uma copia das leituras com o evento aplicado a partir de um instante sorteado."""
    t0 = leituras[0].instante + timedelta(seconds=gerador.uniform(240, 330))
    magnitude = {
        "dessaturacao": gerador.uniform(6, 10),
        "taquicardia": gerador.uniform(30, 40),
        "taquipneia": gerador.uniform(8, 12),
        "piora_combinada": 1.0,
        "artefato": 1.0,
    }[evento]
    saida = []
    for leitura in leituras:
        decorrido = (leitura.instante - t0).total_seconds()
        if decorrido < 0:
            saida.append(leitura)
            continue
        valores = dict(leitura.valores)

        def somar(variavel: str, delta: float, valores: dict = valores) -> None:
            if valores.get(variavel) is not None:
                valores[variavel] = valores[variavel] + delta

        if evento == "dessaturacao":
            somar("saturacao", -magnitude * _rampa(decorrido, 60))
        elif evento == "taquicardia":
            somar("fc", magnitude * _rampa(decorrido, 30))
        elif evento == "taquipneia":
            somar("fr", magnitude * _rampa(decorrido, 60))
        elif evento == "piora_combinada":
            fator = _rampa(decorrido, 60)
            somar("fc", 12 * fator)
            somar("fr", 4 * fator)
            somar("saturacao", -3 * fator)
        elif evento == "artefato" and decorrido < 8:
            valores["saturacao"] = 0.0
            valores["fc"] = 0.0
        if valores.get("saturacao") is not None:
            valores["saturacao"] = min(valores["saturacao"], 100.0)
        saida.append(replace(leitura, valores=valores))
    return saida, Injecao(evento, t0, magnitude)


def _clinicas(detector: DetectorSinaisVitais, leituras: Sequence[Leitura]) -> list:
    achados = []
    for leitura in leituras:
        achados.extend(a for a in detector.atualizar(leitura)
                       if not a.tipo.startswith("falha_sensor"))
    return achados


def avaliar_sinais_vitais(caminhos: Sequence[Path] | None = None, semente: int = 42,
                          configuracoes: dict[str, frozenset[str]] | None = None,
                          parametros: Callable[..., ParametrosSinaisVitais] =
                          ParametrosSinaisVitais.alta_frequencia,
                          progresso: Callable[[str], None] | None = None) -> dict[str, Any]:
    caminhos = list(caminhos or registros_bidmc())
    configuracoes = configuracoes or CONFIGURACOES
    registros = {c.stem[:8]: ler_bidmc(c) for c in caminhos}
    horas_limpas = sum(
        (r[-1].instante - r[0].instante).total_seconds() for r in registros.values()) / 3600

    resultado: dict[str, Any] = {
        "dataset": "PhysioNet BIDMC (1 Hz)", "registros": len(registros),
        "horas_monitoradas": round(horas_limpas, 2), "semente": semente, "configuracoes": {},
    }
    for nome, camadas in configuracoes.items():
        if progresso:
            progresso(nome)
        gerador = random.Random(semente)
        alertas_limpos: Counter = Counter()
        registros_com_alerta = 0
        deteccoes: dict[str, list[float | None]] = {e: [] for e in EVENTOS_CLINICOS}
        artefatos_com_alarme = 0
        for leituras in registros.values():
            limpos = _clinicas(DetectorSinaisVitais(parametros(camadas=camadas)), leituras)
            alertas_limpos.update(a.detector for a in limpos)
            registros_com_alerta += bool(limpos)
            for evento in EVENTOS:
                injetadas, injecao = injetar(leituras, evento, gerador)
                achados = _clinicas(DetectorSinaisVitais(parametros(camadas=camadas)), injetadas)
                depois = [a for a in achados if a.instante and a.instante >= injecao.inicio]
                if evento == "artefato":
                    janela = injecao.inicio + timedelta(seconds=120)
                    novos = [a for a in depois if a.instante <= janela]
                    referencia = [a for a in limpos
                                  if a.instante and injecao.inicio <= a.instante <= janela]
                    artefatos_com_alarme += len(novos) > len(referencia)
                    continue
                latencia = ((depois[0].instante - injecao.inicio).total_seconds()
                            if depois else None)
                deteccoes[evento].append(latencia)

        por_evento = {}
        for evento, latencias in deteccoes.items():
            achadas = [x for x in latencias if x is not None]
            por_evento[evento] = {
                "sensibilidade": round(len(achadas) / len(latencias), 3),
                "latencia_mediana_s": round(statistics.median(achadas), 1) if achadas else None,
            }
        todas = [x for lat in deteccoes.values() for x in lat]
        resultado["configuracoes"][nome] = {
            "camadas": sorted(camadas),
            "sensibilidade_geral": round(sum(x is not None for x in todas) / len(todas), 3),
            "por_evento": por_evento,
            "alertas_sem_injecao_por_hora": round(sum(alertas_limpos.values()) / horas_limpas, 2),
            "registros_sem_injecao_com_alerta": registros_com_alerta,
            "alertas_sem_injecao_por_camada": dict(alertas_limpos),
            "artefatos_que_viraram_alarme": f"{artefatos_com_alarme}/{len(registros)}",
        }
    return resultado


def tabela_markdown(resultado: dict[str, Any]) -> str:
    eventos = EVENTOS_CLINICOS
    linhas = [
        "| Configuracao | Sensib. geral | " + " | ".join(eventos)
        + " | Alertas/h sem injecao | Artefato -> alarme |",
        "|---|---|" + "---|" * len(eventos) + "---|---|",
    ]
    for nome, dados in resultado["configuracoes"].items():
        celulas = []
        for evento in eventos:
            e = dados["por_evento"][evento]
            latencia = f" ({e['latencia_mediana_s']:.0f} s)" if e["latencia_mediana_s"] is not None else ""
            celulas.append(f"{e['sensibilidade']:.0%}{latencia}")
        linhas.append(
            f"| {nome} | {dados['sensibilidade_geral']:.0%} | " + " | ".join(celulas)
            + f" | {dados['alertas_sem_injecao_por_hora']} | {dados['artefatos_que_viraram_alarme']} |"
        )
    return "\n".join(linhas)


def leituras_internacao(dados: dict[str, Any]) -> list[Leitura]:
    """Converte as series sinteticas de internacao num unico fluxo ordenado no tempo."""
    leituras = []
    for item in dados["sinais_vitais_horarios"]:
        valores = {k: v for k, v in item.items() if k not in {"prontuario", "aferido_em"}}
        leituras.append(Leitura(item["prontuario"], "sinais_vitais",
                                datetime.fromisoformat(item["aferido_em"]), valores))
    for item in dados["prescricoes"]:
        leituras.append(Leitura(item["prontuario"], "prescricao",
                                datetime.fromisoformat(item["prescrito_em"]), dict(item)))
    for item in dados["movimentacao"]:
        valores = {k: v for k, v in item.items() if k not in {"prontuario", "registrado_em"}}
        leituras.append(Leitura(item["prontuario"], "movimentacao",
                                datetime.fromisoformat(item["registrado_em"]), valores))
    ordem = {"prescricao": 0, "sinais_vitais": 1, "movimentacao": 2}
    return sorted(leituras, key=lambda item: (item.instante, ordem[item.modalidade], item.prontuario))


def detectores_internacao(dados: dict[str, Any], alergias: dict[str, list[str]]) -> list:
    from .movimentacao import DetectorMovimentacao
    from .prescricoes import DetectorPrescricoes

    return [
        DetectorSinaisVitais(ParametrosSinaisVitais.horario()),
        DetectorPrescricoes(alergias=alergias),
        DetectorMovimentacao(risco_queda=dados["risco_queda"]),
    ]


def avaliar_internacao(dados: dict[str, Any], alergias: dict[str, list[str]]) -> dict[str, Any]:
    """Compara as anomalias detectadas com os eventos rotulados do gerador sintetico."""
    detectores = detectores_internacao(dados, alergias)
    achados = []
    for leitura in leituras_internacao(dados):
        for detector in detectores:
            if leitura.modalidade in detector.modalidades:
                achados.extend(detector.atualizar(leitura))

    tolerancia = timedelta(hours=1)
    usados: set[int] = set()
    eventos = []
    for rotulo in dados["rotulos"]:
        inicio = datetime.fromisoformat(rotulo["inicio"])
        fim = datetime.fromisoformat(rotulo["fim"]) + tolerancia
        casados = [
            (indice, a) for indice, a in enumerate(achados)
            if a.prontuario == rotulo["prontuario"] and a.modalidade == rotulo["modalidade"]
            and any(a.tipo.startswith(t) for t in rotulo["tipos_esperados"])
            and a.instante is not None and inicio <= a.instante <= fim
        ]
        usados.update(i for i, _ in casados)
        primeiro = min((a.instante for _, a in casados), default=None)
        eventos.append({
            **{k: rotulo[k] for k in ("prontuario", "modalidade", "evento")},
            "detectado": bool(casados),
            "antecedencia_ate_fim_h": (round((datetime.fromisoformat(rotulo["fim"]) - primeiro)
                                             .total_seconds() / 3600, 1) if primeiro else None),
            "tipos_detectados": sorted({a.tipo for _, a in casados}),
        })
    fora = [a for i, a in enumerate(achados) if i not in usados]
    por_modalidade = {}
    for modalidade in ("sinais_vitais", "prescricao", "movimentacao"):
        rotulados = [e for e in eventos if e["modalidade"] == modalidade]
        por_modalidade[modalidade] = {
            "eventos_rotulados": len(rotulados),
            "detectados": sum(e["detectado"] for e in rotulados),
            "anomalias_fora_de_rotulo": sum(a.modalidade == modalidade for a in fora),
        }
    return {
        "dataset": "internacao sintetica (12 pacientes, ate 72 h)",
        "por_modalidade": por_modalidade,
        "eventos": eventos,
        "anomalias_fora_de_rotulo": [
            {"prontuario": a.prontuario, "tipo": a.tipo, "severidade": a.severidade,
             "instante": a.fim, "descricao": a.descricao} for a in fora
        ],
    }


def avaliar_quedas(caminhos: Sequence[Path] | None = None) -> dict[str, Any]:
    """Sensibilidade e especificidade do detector de queda no UR Fall.

    Sequencias impares serviram para calibrar o limiar; as pares sao o teste.
    """
    from .fontes import ler_urfall, registros_urfall
    from .movimentacao import DetectorMovimentacao

    resultado: dict[str, Any] = {"dataset": "UR Fall Detection (acelerometro)", "particoes": {}}
    for particao, resto in (("calibracao (impares)", 1), ("teste (pares)", 0)):
        vp = fn = fp = vn = 0
        erros = []
        for caminho in caminhos or registros_urfall():
            nome = caminho.stem.replace("-acc", "")
            if int(nome.split("-")[1]) % 2 != resto:
                continue
            detector = DetectorMovimentacao()
            detectou = any(a.tipo == "queda" for leitura in ler_urfall(caminho)
                           for a in detector.atualizar(leitura))
            queda = nome.startswith("fall")
            vp += queda and detectou
            fn += queda and not detectou
            fp += (not queda) and detectou
            vn += (not queda) and not detectou
            if queda != detectou:
                erros.append(nome)
        resultado["particoes"][particao] = {
            "quedas": vp + fn, "atividades": fp + vn,
            "sensibilidade": round(vp / max(vp + fn, 1), 3),
            "especificidade": round(vn / max(vn + fp, 1), 3),
            "vp": vp, "fn": fn, "fp": fp, "vn": vn, "erros": erros,
        }
    return resultado
