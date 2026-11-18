"""Grandezas do movimento a partir dos 17 pontos COCO.

Tudo e calculado em coordenadas da imagem (y cresce para baixo) e com pontos de
confianca minima; ponto incerto vira NaN e lacunas curtas sao interpoladas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extracao import SequenciaPose

CONFIANCA_MINIMA = 0.3

ARTICULACOES = {
    "joelho_esq": (11, 13, 15), "joelho_dir": (12, 14, 16),
    "quadril_esq": (5, 11, 13), "quadril_dir": (6, 12, 14),
    "cotovelo_esq": (5, 7, 9), "cotovelo_dir": (6, 8, 10),
    "ombro_esq": (11, 5, 7), "ombro_dir": (12, 6, 8),
}
PARES = {"joelho": ("joelho_esq", "joelho_dir"), "quadril": ("quadril_esq", "quadril_dir"),
         "cotovelo": ("cotovelo_esq", "cotovelo_dir"), "ombro": ("ombro_esq", "ombro_dir")}


def _ponto(pontos: np.ndarray, indice: int) -> np.ndarray:
    xy = pontos[:, indice, :2].astype(float).copy()
    xy[pontos[:, indice, 2] < CONFIANCA_MINIMA] = np.nan
    return xy


def _media(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))


def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angulo em b (graus) formado por a-b-c, quadro a quadro."""
    v1, v2 = a - b, c - b
    cosseno = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cosseno, -1, 1)))


def interpolar(serie: np.ndarray, lacuna_maxima: int = 8) -> np.ndarray:
    """Preenche lacunas curtas de NaN por interpolacao linear."""
    saida = serie.astype(float).copy()
    validos = ~np.isnan(saida)
    if validos.sum() < 2:
        return saida
    indices = np.arange(len(saida))
    preenchido = np.interp(indices, indices[validos], saida[validos])
    inicio = None
    for i, valido in enumerate(np.append(validos, True)):
        if not valido and inicio is None:
            inicio = i
        elif valido and inicio is not None:
            if i - inicio <= lacuna_maxima and inicio > 0 and i < len(saida):
                saida[inicio:i] = preenchido[inicio:i]
            inicio = None
    return saida


def suavizar(serie: np.ndarray, janela: int = 5) -> np.ndarray:
    """Mediana movel (robusta a saltos de um quadro), ignorando NaN."""
    if janela <= 1:
        return serie
    meio = janela // 2
    saida = np.full(len(serie), np.nan)
    for i in range(len(serie)):
        trecho = serie[max(0, i - meio):i + meio + 1]
        trecho = trecho[~np.isnan(trecho)]
        if len(trecho):
            saida[i] = np.median(trecho)
    return saida


@dataclass
class Cinematica:
    tempos: np.ndarray
    fps: float
    angulo_tronco: np.ndarray
    razao_caixa: np.ndarray
    altura_caixa: np.ndarray
    quadril_y: np.ndarray
    pe: np.ndarray
    articulacoes: dict[str, np.ndarray]
    presente: np.ndarray
    razao_ombros: float = float("nan")
    razao_ombros_quadro: np.ndarray | None = None
    confianca_articulacoes: dict[str, np.ndarray] | None = None
    comprimentos: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def perfil(self) -> bool:
        """Camera de lado: ombros quase sobrepostos em relacao ao comprimento do tronco."""
        return bool(np.isfinite(self.razao_ombros) and self.razao_ombros < 0.35)


def calcular(sequencia: SequenciaPose, janela: int = 5) -> Cinematica:
    p = sequencia.pontos
    ombros = _media(_ponto(p, 5), _ponto(p, 6))
    quadris = _media(_ponto(p, 11), _ponto(p, 12))
    tornozelos = _media(_ponto(p, 15), _ponto(p, 16))

    tronco = ombros - quadris
    angulo_tronco = np.degrees(np.arctan2(np.abs(tronco[:, 0]), -tronco[:, 1]))
    caixas = sequencia.caixas
    largura = caixas[:, 2] - caixas[:, 0]
    altura = caixas[:, 3] - caixas[:, 1]
    pe = np.where(np.isnan(tornozelos),
                  np.stack([(caixas[:, 0] + caixas[:, 2]) / 2, caixas[:, 3]], axis=1), tornozelos)

    articulacoes, comprimentos = {}, {}
    for nome, (a, b, c) in ARTICULACOES.items():
        pa, pb, pc = _ponto(p, a), _ponto(p, b), _ponto(p, c)
        articulacoes[nome] = suavizar(interpolar(angulo(pa, pb, pc)), janela)
        comprimentos[nome] = (np.linalg.norm(pa - pb, axis=1), np.linalg.norm(pc - pb, axis=1))
    return Cinematica(
        tempos=sequencia.tempos, fps=sequencia.fps,
        angulo_tronco=suavizar(interpolar(angulo_tronco), janela),
        razao_caixa=suavizar(interpolar(largura / altura), janela),
        altura_caixa=suavizar(interpolar(altura), janela),
        quadril_y=suavizar(interpolar(quadris[:, 1]), janela),
        pe=pe, articulacoes=articulacoes, presente=~np.isnan(caixas[:, 0]),
        razao_ombros=_mediana(_razoes_ombros(p, tronco)),
        razao_ombros_quadro=_razoes_ombros(p, tronco),
        confianca_articulacoes={nome: np.min(p[:, list(indices), 2], axis=1)
                                for nome, indices in ARTICULACOES.items()},
        comprimentos=comprimentos,
    )


def _razoes_ombros(pontos: np.ndarray, tronco: np.ndarray) -> np.ndarray:
    largura = np.abs(_ponto(pontos, 5)[:, 0] - _ponto(pontos, 6)[:, 0])
    return largura / np.linalg.norm(tronco, axis=1)


def _mediana(valores: np.ndarray) -> float:
    validos = valores[np.isfinite(valores)]
    return float(np.median(validos)) if len(validos) else float("nan")
