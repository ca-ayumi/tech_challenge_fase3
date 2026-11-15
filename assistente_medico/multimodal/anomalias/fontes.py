"""Leitura dos datasets publicos como fluxo de ``Leitura``."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

from ...config import CAMINHOS
from ..tempo_real import Leitura

INICIO_PADRAO = datetime(2026, 9, 15, 8, 0, 0)


def _numero(texto: str) -> float | None:
    try:
        valor = float(texto)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(valor) else valor


def registros_bidmc(pasta: Path | None = None) -> list[Path]:
    pasta = Path(pasta or CAMINHOS.multimodal / "bidmc")
    return sorted(pasta.glob("bidmc_*_Numerics.csv"))


def ler_bidmc(caminho: Path, prontuario: str | None = None,
              inicio: datetime = INICIO_PADRAO) -> list[Leitura]:
    """Registro BIDMC (1 Hz): HR, RESP e SpO2 viram fc, fr e saturacao."""
    prontuario = prontuario or caminho.stem.replace("_Numerics", "").upper()
    leituras = []
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo, skipinitialspace=True):
            linha = {chave.strip(): valor for chave, valor in linha.items()}
            segundos = _numero(linha.get("Time [s]"))
            if segundos is None:
                continue
            leituras.append(Leitura(
                prontuario=prontuario, modalidade="sinais_vitais",
                instante=inicio + timedelta(seconds=segundos),
                valores={"fc": _numero(linha.get("HR")), "fr": _numero(linha.get("RESP")),
                         "saturacao": _numero(linha.get("SpO2"))},
            ))
    return leituras


def registros_urfall(pasta: Path | None = None) -> list[Path]:
    pasta = Path(pasta or CAMINHOS.multimodal / "urfall")
    return sorted(pasta.glob("*-acc.csv"))


def ler_urfall(caminho: Path, prontuario: str | None = None,
               inicio: datetime = INICIO_PADRAO) -> list[Leitura]:
    """Acelerometro do UR Fall: tempo em ms, magnitude total e eixos, em g.

    O eixo y e o vertical do sensor preso a cintura: perto de 1 g em pe, perto
    de 0 deitado.
    """
    prontuario = prontuario or caminho.stem.replace("-acc", "").upper()
    leituras = []
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        for campos in csv.reader(arquivo):
            if len(campos) < 5:
                continue
            tempo, total, ax, ay, az = (_numero(c) for c in campos[:5])
            if tempo is None or total is None:
                continue
            leituras.append(Leitura(
                prontuario=prontuario, modalidade="movimentacao",
                instante=inicio + timedelta(milliseconds=tempo),
                valores={"aceleracao_g": total, "ax": ax, "ay": ay, "az": az,
                         "fonte": "acelerometro"},
            ))
    leituras.sort(key=lambda leitura: leitura.instante)
    return leituras
