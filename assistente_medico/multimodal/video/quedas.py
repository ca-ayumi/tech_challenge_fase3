"""Queda e permanencia no chao a partir da pose.

Estado por quadro:

- ereto (em pe ou sentado): tronco (ombros -> quadris) a menos de 35 graus da
  vertical e caixa da pessoa mais alta que larga;
- no chao: tronco a mais de 60 graus, ou caixa mais larga que alta, ou caixa
  com menos de 55% da altura que a pessoa tinha ereta nos ultimos 2 s. Os dois
  ultimos criterios cobrem a queda em direcao a camera, em que o tronco
  continua "vertical" na imagem 2D.

A altura de referencia e local (ultimos 2 s em pe) porque a pessoa muda de
tamanho na imagem ao se aproximar da camera.

Deitar nao e cair: no UR Fall, 16 das 40 atividades comuns terminam no chao ou
na cama. A queda exige chegar ao chao a partir da postura ereta *rapidamente*:
transicao curta e velocidade alta de descida do quadril ou de encolhimento da
caixa. Os limiares foram calibrados nas sequencias impares do UR Fall e
avaliados nas pares. Com cama ou sofa sob a pessoa (YOLOv8), o evento vira
"deitou em movel".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cinematica import Cinematica
from .extracao import SequenciaPose

MOVEIS_DE_DEITAR = {"bed", "couch", "bench"}


@dataclass(frozen=True)
class ParametrosQueda:
    tronco_vertical: float = 35.0
    razao_ereto: float = 0.8
    tronco_horizontal: float = 60.0
    razao_chao: float = 1.0
    altura_chao: float = 0.55
    janela_referencia_s: float = 2.0
    persistencia_chao_s: float = 0.5
    transicao_maxima_s: float = 1.0
    velocidade_minima: float = 1.0
    janela_velocidade_pos_s: float = 1.0
    descida_minima: float = 0.3
    tempo_no_chao_critico_s: float = 10.0
    sobreposicao_movel: float = 0.4


@dataclass
class EventoQueda:
    tipo: str
    inicio_s: float
    fim_s: float
    evidencias: dict[str, Any] = field(default_factory=dict)


def altura_referencia(cinematica: Cinematica, eretos: np.ndarray, janela_s: float) -> np.ndarray:
    """Mediana da altura da caixa nos quadros eretos dos ultimos ``janela_s`` segundos."""
    t, altura = cinematica.tempos, cinematica.altura_caixa
    referencia = np.full(len(t), np.nan)
    ultima = np.nan
    for i in range(len(t)):
        janela = (t >= t[i] - janela_s) & (t < t[i]) & eretos & ~np.isnan(altura)
        if janela.any():
            ultima = float(np.median(altura[janela]))
        referencia[i] = ultima
    return referencia


def estados(cinematica: Cinematica, p: ParametrosQueda) -> np.ndarray:
    """'em_pe', 'chao', 'intermediario' ou 'ausente' por quadro."""
    tronco, razao = cinematica.angulo_tronco, cinematica.razao_caixa
    eretos = (tronco < p.tronco_vertical) & (razao < p.razao_ereto)
    referencia = altura_referencia(cinematica, eretos, p.janela_referencia_s)
    relativa = cinematica.altura_caixa / referencia
    chao = (tronco > p.tronco_horizontal) | (razao > p.razao_chao) | (relativa < p.altura_chao)
    saida = np.full(len(tronco), "intermediario", dtype=object)
    saida[np.isnan(razao)] = "ausente"
    saida[eretos] = "em_pe"
    saida[chao & ~eretos] = "chao"
    return saida


def _sobreposicao(caixa: np.ndarray, outra: list[float]) -> float:
    x0, y0 = max(caixa[0], outra[0]), max(caixa[1], outra[1])
    x1, y1 = min(caixa[2], outra[2]), min(caixa[3], outra[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = (caixa[2] - caixa[0]) * (caixa[3] - caixa[1])
    return float((x1 - x0) * (y1 - y0) / area) if area > 0 else 0.0


def _movel_sob(sequencia: SequenciaPose, indice: int, p: ParametrosQueda) -> str | None:
    if not sequencia.objetos or np.isnan(sequencia.caixas[indice, 0]):
        return None
    amostra = min(sequencia.objetos, key=lambda o: abs(o["quadro"] - indice))
    for objeto in amostra["objetos"]:
        if (objeto["classe"] in MOVEIS_DE_DEITAR
                and _sobreposicao(sequencia.caixas[indice], objeto["caixa"]) >= p.sobreposicao_movel):
            return objeto["classe"]
    return None


def detectar(sequencia: SequenciaPose, cinematica: Cinematica,
             parametros: ParametrosQueda | None = None) -> list[EventoQueda]:
    p = parametros or ParametrosQueda()
    t = cinematica.tempos
    if len(t) < 3:
        return []
    estado = estados(cinematica, p)
    eretos = estado == "em_pe"
    referencia = altura_referencia(cinematica, eretos, p.janela_referencia_s)
    velocidade_quadril = np.gradient(cinematica.quadril_y, t)
    velocidade_caixa = -np.gradient(cinematica.altura_caixa, t)

    eventos: list[EventoQueda] = []
    i = 0
    while i < len(t):
        if estado[i] != "chao":
            i += 1
            continue
        fim = i
        while fim + 1 < len(t) and estado[fim + 1] in {"chao", "ausente"}:
            fim += 1
        duracao = float(t[fim] - t[i])
        if duracao < p.persistencia_chao_s and fim < len(t) - 1:
            i = fim + 1
            continue
        anteriores = np.flatnonzero(eretos[:i])
        if not len(anteriores) or not np.isfinite(referencia[i]):
            i = fim + 1
            continue
        j = int(anteriores[-1])
        altura = referencia[i]
        trecho = slice(j, min(i + int(p.persistencia_chao_s * cinematica.fps), fim) + 1)
        descida_quadril = float((np.nanmax(cinematica.quadril_y[trecho]) - cinematica.quadril_y[j]) / altura)
        encolhimento = float(1 - np.nanmin(cinematica.altura_caixa[trecho]) / altura)
        apos = min(i + int(p.janela_velocidade_pos_s * cinematica.fps), fim) + 1
        velocidade = float(max(np.nanmax(velocidade_quadril[j:apos]),
                               np.nanmax(velocidade_caixa[j:apos])) / altura)
        transicao = float(t[i] - t[j])
        movel = _movel_sob(sequencia, min(i + int(0.5 * cinematica.fps), fim), p)
        evidencias = {"descida_quadril": round(descida_quadril, 2),
                      "encolhimento_caixa": round(encolhimento, 2),
                      "velocidade_pico_alturas_s": round(velocidade, 2),
                      "transicao_s": round(transicao, 2), "tempo_no_chao_s": round(duracao, 1),
                      "movel": movel}
        rapida = transicao <= p.transicao_maxima_s and velocidade >= p.velocidade_minima
        desceu = max(descida_quadril, encolhimento) >= p.descida_minima
        if movel:
            eventos.append(EventoQueda("deitou_em_movel", float(t[j]), float(t[fim]), evidencias))
        elif rapida and desceu:
            eventos.append(EventoQueda("queda", float(t[j]), float(t[i]), evidencias))
            if duracao >= p.tempo_no_chao_critico_s:
                eventos.append(EventoQueda("permanencia_no_chao", float(t[i]), float(t[fim]),
                                           evidencias))
        else:
            eventos.append(EventoQueda("deitou_controlado", float(t[j]), float(t[fim]), evidencias))
        i = fim + 1
    return eventos
