"""Medidas acusticas da fala, calculadas localmente com o Praat (parselmouth).

Temporais (de Jong & Wempe, 2009): silencio e o que fica 25 dB abaixo do pico
de intensidade; pausa e silencio de pelo menos 0,3 s entre trechos de fala;
nucleo silabico e um pico de intensidade sonoro (com F0 definido) separado do
anterior por um vale de pelo menos 2 dB. Dai saem taxa de fala, taxa de
articulacao (sem as pausas), silabas por grupo respiratorio e irregularidade do
ritmo.

Prosodicas: F0 medio e sua variacao em semitons (fala monotona tem pouca
variacao), intensidade media.

Qualidade vocal: jitter e shimmer locais e HNR, pelos algoritmos do Praat.
Em fala encadeada sao aproximacoes; em vogal sustentada (VOICED) sao a medida
classica.

Nenhum audio sai da maquina nesta etapa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LIMIAR_SILENCIO_DB = 25.0
PAUSA_MINIMA_S = 0.3
VALE_MINIMO_DB = 2.0
INTENSIDADE_MINIMA_DB = 35.0


@dataclass
class MedidasVoz:
    duracao_s: float
    tempo_fala_s: float
    pausas: int
    pausa_media_s: float
    fracao_pausa: float
    pausas_por_minuto: float
    silabas: int
    taxa_fala: float
    taxa_articulacao: float
    silabas_por_grupo: float
    irregularidade_ritmo: float
    f0_media_hz: float | None
    f0_variacao_st: float | None
    intensidade_media_db: float
    jitter_local: float | None
    shimmer_local: float | None
    hnr_db: float | None
    intervalos_pausa: list[tuple[float, float]] = field(default_factory=list, repr=False)

    def como_dicionario(self) -> dict[str, Any]:
        dados = asdict(self)
        dados.pop("intervalos_pausa")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dados.items()}


def carregar_som(origem: str | Path | np.ndarray, taxa: int | None = None):
    import parselmouth

    if isinstance(origem, np.ndarray):
        if taxa is None:
            raise ValueError("Informe a taxa de amostragem do sinal.")
        return parselmouth.Sound(origem.astype(np.float64), sampling_frequency=taxa)
    return parselmouth.Sound(str(origem))


def _segmentos(ativo: np.ndarray, tempos: np.ndarray) -> list[tuple[float, float]]:
    segmentos = []
    inicio = None
    passo = float(tempos[1] - tempos[0]) if len(tempos) > 1 else 0.01
    for t, a in zip(tempos, ativo, strict=True):
        if a and inicio is None:
            inicio = t
        elif not a and inicio is not None:
            segmentos.append((float(inicio), float(t)))
            inicio = None
    if inicio is not None:
        segmentos.append((float(inicio), float(tempos[-1] + passo)))
    return segmentos


def _nucleos_silabicos(intensidade: np.ndarray, tempos: np.ndarray, limiar: float,
                       vozeado) -> list[float]:
    picos = [i for i in range(1, len(intensidade) - 1)
             if intensidade[i] >= intensidade[i - 1] and intensidade[i] > intensidade[i + 1]
             and intensidade[i] > limiar]
    nucleos: list[float] = []
    anterior = None
    for i in picos:
        if anterior is not None:
            vale = float(np.min(intensidade[anterior:i + 1]))
            if intensidade[i] - vale < VALE_MINIMO_DB:
                if intensidade[i] > intensidade[anterior] and nucleos and vozeado(tempos[i]):
                    nucleos[-1] = float(tempos[i])
                    anterior = i
                continue
        if vozeado(tempos[i]):
            nucleos.append(float(tempos[i]))
            anterior = i
    return nucleos


def _qualidade_vocal(som) -> tuple[float | None, float | None, float | None]:
    from parselmouth.praat import call

    try:
        pontos = call(som, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pontos, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([som, pontos], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicidade = call(som, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicidade, "Get mean", 0, 0)
    except Exception:  # sinal sem trechos periodicos suficientes
        return None, None, None

    def limpo(valor: float) -> float | None:
        return None if valor is None or not np.isfinite(valor) else float(valor)

    return limpo(jitter), limpo(shimmer), limpo(hnr)


def medir(origem: str | Path | np.ndarray, taxa: int | None = None) -> MedidasVoz:
    som = carregar_som(origem, taxa)
    intensidade_obj = som.to_intensity(minimum_pitch=75, time_step=0.01)
    intensidade = intensidade_obj.values[0]
    tempos = intensidade_obj.xs()
    pitch = som.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0_tempos = pitch.xs()

    def vozeado(t: float) -> bool:
        if len(f0_tempos) == 0:
            return False
        indice = int(np.argmin(np.abs(f0_tempos - t)))
        return bool(f0[indice] > 0)

    intensidade = np.nan_to_num(intensidade, nan=0.0, neginf=0.0)
    if float(np.max(intensidade)) < INTENSIDADE_MINIMA_DB:
        raise ValueError("Audio sem fala audivel (intensidade maxima muito baixa).")
    limiar = float(np.percentile(intensidade, 99)) - LIMIAR_SILENCIO_DB
    trechos = _segmentos(intensidade > limiar, tempos)
    if not trechos:
        raise ValueError("Nenhum trecho de fala detectado no audio.")

    pausas = [(fim, proximo) for (_, fim), (proximo, _) in zip(trechos, trechos[1:], strict=False)
              if proximo - fim >= PAUSA_MINIMA_S]
    inicio_fala, fim_fala = trechos[0][0], trechos[-1][1]
    duracao = max(fim_fala - inicio_fala, 1e-6)
    tempo_pausa = sum(b - a for a, b in pausas)
    tempo_fala = max(duracao - tempo_pausa, 1e-6)

    nucleos = _nucleos_silabicos(intensidade, tempos, limiar, vozeado)
    grupos = len(pausas) + 1
    intervalos = np.diff(nucleos) if len(nucleos) > 2 else np.array([])
    dentro = [d for d, (a, b) in zip(intervalos, zip(nucleos, nucleos[1:], strict=False), strict=False)
              if not any(p0 <= b and a <= p1 for p0, p1 in pausas)]
    irregularidade = float(np.std(dentro) / np.mean(dentro)) if len(dentro) > 2 else 0.0

    vozeados = f0[f0 > 0]
    if len(vozeados) > 5:
        f0_media = float(np.mean(vozeados))
        semitons = 12 * np.log2(vozeados / np.median(vozeados))
        f0_variacao = float(np.std(semitons))
    else:
        f0_media = f0_variacao = None

    jitter, shimmer, hnr = _qualidade_vocal(som)
    falando = intensidade > limiar
    return MedidasVoz(
        duracao_s=duracao,
        tempo_fala_s=tempo_fala,
        pausas=len(pausas),
        pausa_media_s=tempo_pausa / len(pausas) if pausas else 0.0,
        fracao_pausa=tempo_pausa / duracao,
        pausas_por_minuto=len(pausas) / (duracao / 60),
        silabas=len(nucleos),
        taxa_fala=len(nucleos) / duracao,
        taxa_articulacao=len(nucleos) / tempo_fala,
        silabas_por_grupo=len(nucleos) / grupos,
        irregularidade_ritmo=irregularidade,
        f0_media_hz=f0_media,
        f0_variacao_st=f0_variacao,
        intensidade_media_db=float(np.mean(intensidade[falando])) if falando.any() else 0.0,
        jitter_local=jitter,
        shimmer_local=shimmer,
        hnr_db=hnr,
        intervalos_pausa=pausas,
    )
