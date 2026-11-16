"""Analise de sessoes de exercicio e fisioterapia.

1. A articulacao principal e a de maior amplitude util (p95 - p5) entre joelhos,
   quadris, cotovelos e ombros.
2. As repeticoes sao os extremos do angulo, longe da postura de repouso
   (``scipy.signal.find_peaks`` com proeminencia e distancia minimas).
3. Em cada repeticao: amplitude de movimento (ADM), lado contralateral,
   inclinacao maxima do tronco e duracao.

Desvios, sempre em relacao a propria pessoa (primeiras repeticoes):

- queda progressiva de amplitude ao longo da serie (sugestiva de fadiga);
- assimetria entre lados (no exercicio bilateral, repeticao a repeticao; no
  alternado, media do lado esquerdo x direito). So entram repeticoes com os
  ombros de frente e pontos confiaveis dos dois lados; o limiar (20 graus) fica
  acima do erro tipico de angulo da pose 2D (cerca de 10 graus por articulacao).
  Membro escorcado (bem mais curto na imagem que o do outro lado) tira a
  repeticao da comparacao;
- compensacao de tronco (inclinacao bem maior que nas primeiras repeticoes);
- repeticoes incompletas e ritmo irregular.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cinematica import PARES, Cinematica

PARCEIRO = {lado: outro for a, b in PARES.values() for lado, outro in ((a, b), (b, a))}
NOMES = {"joelho_esq": "joelho esquerdo", "joelho_dir": "joelho direito",
         "quadril_esq": "quadril esquerdo", "quadril_dir": "quadril direito",
         "cotovelo_esq": "cotovelo esquerdo", "cotovelo_dir": "cotovelo direito",
         "ombro_esq": "ombro esquerdo", "ombro_dir": "ombro direito"}


@dataclass(frozen=True)
class ParametrosExercicio:
    amplitude_minima: float = 25.0
    proeminencia_relativa: float = 0.35
    distancia_minima_s: float = 0.6
    cobertura_minima: float = 0.6
    assimetria_moderada: float = 20.0
    assimetria_alta: float = 30.0
    queda_amplitude: float = 0.2
    compensacao_tronco: float = 15.0
    fracao_compensadas: float = 0.3
    incompleta: float = 0.6
    irregularidade_ritmo: float = 0.35
    razao_ombros_frontal: float = 0.35
    confianca_lados: float = 0.5
    minimo_comparaveis: int = 3
    razao_bilateral: float = 0.35
    escorco_maximo: float = 0.8


@dataclass
class Repeticao:
    numero: int
    inicio_s: float
    extremo_s: float
    fim_s: float
    amplitude: float
    amplitude_contralateral: float | None
    tronco_max: float | None
    comparavel: bool = True

    def como_dicionario(self) -> dict[str, Any]:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Desvio:
    tipo: str
    severidade: str
    descricao: str
    evidencias: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnaliseExercicio:
    articulacao: str | None
    modo: str
    repeticoes: list[Repeticao]
    desvios: list[Desvio]
    resumo: dict[str, Any]

    def como_dicionario(self) -> dict[str, Any]:
        return {"articulacao": self.articulacao, "modo": self.modo, "resumo": self.resumo,
                "repeticoes": [r.como_dicionario() for r in self.repeticoes],
                "desvios": [d.__dict__ for d in self.desvios]}


def _amplitude_util(serie: np.ndarray) -> float:
    validos = serie[~np.isnan(serie)]
    return float(np.percentile(validos, 95) - np.percentile(validos, 5)) if len(validos) > 10 else 0.0


def escolher_articulacao(cinematica: Cinematica, p: ParametrosExercicio) -> str | None:
    candidatas = {}
    presente = cinematica.presente if cinematica.presente.any() else np.ones(len(cinematica.tempos), bool)
    for nome, serie in cinematica.articulacoes.items():
        if np.mean(~np.isnan(serie[presente])) >= p.cobertura_minima:
            candidatas[nome] = _amplitude_util(serie)
    if not candidatas:
        return None
    melhor = max(candidatas, key=candidatas.get)
    return melhor if candidatas[melhor] >= p.amplitude_minima else None


def segmentar(serie: np.ndarray, tempos: np.ndarray, fps: float,
              p: ParametrosExercicio) -> list[tuple[int, int, int]]:
    """(inicio, extremo, fim) de cada repeticao, em indices de quadro."""
    from scipy.signal import find_peaks

    validos = ~np.isnan(serie)
    if validos.sum() < 10:
        return []
    preenchida = np.interp(np.arange(len(serie)), np.flatnonzero(validos), serie[validos])
    repouso = float(np.median(preenchida))
    baixo, alto = np.percentile(preenchida, [5, 95])
    sinal = -preenchida if (repouso - baixo) > (alto - repouso) else preenchida
    amplitude = alto - baixo
    extremos, _ = find_peaks(sinal, prominence=p.proeminencia_relativa * amplitude,
                             distance=max(int(p.distancia_minima_s * fps), 1))
    if len(extremos) == 0:
        return []
    limites = [0] + [int((a + b) / 2) for a, b in zip(extremos, extremos[1:], strict=False)] + [len(serie) - 1]
    return [(limites[k], int(e), limites[k + 1]) for k, e in enumerate(extremos)]


def _comparavel(cinematica: Cinematica, articulacao: str, a: int, e: int, b: int,
                p: ParametrosExercicio) -> bool:
    """Os dois lados sao comparaveis nesta repeticao.

    Exige ombros de frente, pontos confiaveis e, no ponto extremo da repeticao,
    segmentos (coxa e perna, braco e antebraco) com comprimento aparente parecido
    dos dois lados: um segmento bem mais curto na imagem esta escorcado
    (apontando para a camera) e o angulo 2D dele nao representa o angulo real.
    """
    trecho = slice(a, b + 1)
    if cinematica.razao_ombros_quadro is not None:
        razoes = cinematica.razao_ombros_quadro[trecho]
        razoes = razoes[np.isfinite(razoes)]
        if not len(razoes) or np.median(razoes) < p.razao_ombros_frontal:
            return False
    parceira = PARCEIRO.get(articulacao)
    if cinematica.comprimentos is not None and parceira in cinematica.comprimentos:
        extremo = slice(max(e - 5, 0), e + 6)
        for proprio_seg, outro_seg in zip(cinematica.comprimentos[articulacao],
                                          cinematica.comprimentos[parceira], strict=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                proprio = np.nanmedian(proprio_seg[extremo])
                outro = np.nanmedian(outro_seg[extremo])
            if not (np.isfinite(proprio) and np.isfinite(outro)) or min(proprio, outro) < (
                    p.escorco_maximo * max(proprio, outro)):
                return False
    if cinematica.confianca_articulacoes is not None:
        for nome in (articulacao, PARCEIRO.get(articulacao)):
            confiancas = cinematica.confianca_articulacoes.get(nome)
            if confiancas is not None and np.nanmean(confiancas[trecho]) < p.confianca_lados:
                return False
    return True


def analisar(cinematica: Cinematica, parametros: ParametrosExercicio | None = None) -> AnaliseExercicio:
    p = parametros or ParametrosExercicio()
    articulacao = escolher_articulacao(cinematica, p)
    if articulacao is None:
        return AnaliseExercicio(None, "sem_movimento", [], [], {
            "repeticoes": 0, "observacao": "sem movimento articular relevante"})
    serie = cinematica.articulacoes[articulacao]
    parceira = cinematica.articulacoes.get(PARCEIRO.get(articulacao, ""))
    t = cinematica.tempos

    repeticoes = []
    for numero, (a, e, b) in enumerate(segmentar(serie, t, cinematica.fps, p), start=1):
        janela = serie[a:b + 1]
        amplitude = float(np.nanmax(janela) - np.nanmin(janela))
        contralateral = None
        if parceira is not None and np.mean(~np.isnan(parceira[a:b + 1])) > 0.5:
            contralateral = float(np.nanmax(parceira[a:b + 1]) - np.nanmin(parceira[a:b + 1]))
        tronco = cinematica.angulo_tronco[a:b + 1]
        repeticoes.append(Repeticao(numero, float(t[a]), float(t[e]), float(t[b]), amplitude,
                                    contralateral,
                                    float(np.nanmax(tronco)) if np.isfinite(tronco).any() else None,
                                    _comparavel(cinematica, articulacao, a, e, b, p)))
    repeticoes = [r for r in repeticoes if r.amplitude >= p.amplitude_minima / 2]
    for numero, repeticao in enumerate(repeticoes, start=1):
        repeticao.numero = numero
    if not repeticoes:
        return AnaliseExercicio(articulacao, "sem_repeticoes", [], [], {
            "repeticoes": 0, "articulacao": NOMES.get(articulacao, articulacao),
            "amplitude_maxima_graus": round(_amplitude_util(serie), 1),
            "observacao": "movimento unico ou lento, sem ciclos repetidos"})

    amplitudes = np.array([r.amplitude for r in repeticoes])
    pares = [(r.amplitude, r.amplitude_contralateral) for r in repeticoes
             if r.amplitude_contralateral is not None and r.comparavel]
    if len(pares) < p.minimo_comparaveis:
        pares = []
    razoes = [min(a, b) / max(a, b) for a, b in pares if max(a, b) > 0]
    if not pares:
        modo = "lados_nao_comparaveis"
    else:
        modo = "bilateral" if np.median(razoes) >= p.razao_bilateral else (
            "alternado" if any(b > a for a, b in pares) else "unilateral")
    duracoes = np.array([r.fim_s - r.inicio_s for r in repeticoes])
    resumo = {
        "repeticoes": len(repeticoes),
        "articulacao": NOMES.get(articulacao, articulacao),
        "amplitude_media_graus": round(float(amplitudes.mean()), 1),
        "cadencia_por_minuto": round(60 / float(np.median(duracoes)), 1) if len(duracoes) else None,
        "modo": modo,
        "vista": "perfil (lados nao comparados)" if cinematica.perfil else "frontal",
        "repeticoes_comparaveis": len(pares),
    }

    desvios: list[Desvio] = []
    if len(repeticoes) >= 6:
        terco = len(repeticoes) // 3
        inicio, fim = float(amplitudes[:terco].mean()), float(amplitudes[-terco:].mean())
        if fim < (1 - p.queda_amplitude) * inicio:
            desvios.append(Desvio(
                "queda_de_amplitude", "moderada",
                f"Amplitude do {NOMES[articulacao]} caiu de {inicio:.0f} para {fim:.0f} graus ao "
                "longo da serie (sugestivo de fadiga)",
                {"inicio_graus": round(inicio, 1), "fim_graus": round(fim, 1)}))

    if cinematica.perfil:
        modo = "perfil"
    if modo == "bilateral":
        diferencas = [abs(a - b) for a, b in pares]
        assimetria = float(np.median(diferencas))
    elif modo == "alternado":
        lado_a = [a for a, b in pares if a >= b]
        lado_b = [b for a, b in pares if b > a]
        assimetria = abs(float(np.mean(lado_a)) - float(np.mean(lado_b))) if lado_a and lado_b else 0.0
    else:
        assimetria = 0.0
    resumo["assimetria_graus"] = round(assimetria, 1)
    if assimetria >= p.assimetria_moderada:
        severidade = "alta" if assimetria >= p.assimetria_alta else "moderada"
        desvios.append(Desvio(
            "assimetria", severidade,
            f"Assimetria de {assimetria:.0f} graus entre {NOMES[articulacao]} e "
            f"{NOMES[PARCEIRO[articulacao]]} ({modo})",
            {"assimetria_graus": round(assimetria, 1), "modo": modo}))

    troncos = [r.tronco_max for r in repeticoes if r.tronco_max is not None]
    if len(troncos) >= 4:
        base = float(np.median(troncos[:2]))
        compensadas = [r.numero for r in repeticoes
                       if r.tronco_max is not None and r.tronco_max - base > p.compensacao_tronco]
        if len(compensadas) / len(repeticoes) >= p.fracao_compensadas:
            desvios.append(Desvio(
                "compensacao_tronco", "moderada",
                f"Inclinacao de tronco acima do padrao inicial em {len(compensadas)} de "
                f"{len(repeticoes)} repeticoes",
                {"tronco_base_graus": round(base, 1), "repeticoes": compensadas}))

    mediana = float(np.median(amplitudes))
    incompletas = [r.numero for r in repeticoes if r.amplitude < p.incompleta * mediana]
    if len(incompletas) >= 2:
        desvios.append(Desvio(
            "repeticoes_incompletas", "baixa",
            f"{len(incompletas)} repeticoes com menos de {p.incompleta:.0%} da amplitude tipica",
            {"repeticoes": incompletas, "amplitude_tipica": round(mediana, 1)}))
    if len(duracoes) >= 4 and duracoes.mean() > 0:
        variacao = float(duracoes.std() / duracoes.mean())
        resumo["variacao_ritmo"] = round(variacao, 2)
        if variacao > p.irregularidade_ritmo:
            desvios.append(Desvio("ritmo_irregular", "baixa",
                                  f"Ritmo irregular entre repeticoes (variacao de {variacao:.0%})",
                                  {"coeficiente_variacao": round(variacao, 2)}))
    return AnaliseExercicio(articulacao, modo, repeticoes, desvios, resumo)
