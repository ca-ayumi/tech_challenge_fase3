"""Avaliacao da deteccao de queda por video no UR Fall.

Gabarito: rotulos quadro a quadro do dataset (-1 nao deitado, 0 transicao,
1 deitado). O inicio da queda e o primeiro quadro diferente de -1. Uma sequencia
de queda conta como detectada se houver evento ``queda``; uma atividade comum
com evento ``queda`` e falso alarme. Sequencias impares calibram, pares testam.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...config import CAMINHOS
from . import cinematica as cin
from . import quedas
from .extracao import PASTA_CACHE, SequenciaPose

PASTA_URFALL = CAMINHOS.multimodal / "urfall_video"


def gabarito(pasta: Path | None = None) -> dict[str, dict[str, Any]]:
    pasta = Path(pasta or PASTA_URFALL)
    rotulos: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for arquivo in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        caminho = pasta / arquivo
        if not caminho.exists():
            continue
        for linha in csv.reader(caminho.open(encoding="utf-8")):
            rotulos[linha[0]].append((int(linha[1]), int(linha[2])))
    saida = {}
    for nome, quadros in rotulos.items():
        inicio = next((q for q, r in sorted(quadros) if r != -1), None)
        saida[nome] = {"queda": nome.startswith("fall"), "quadro_inicio": inicio,
                       "deita": any(r == 1 for _, r in quadros)}
    return saida


def sequencias_em_cache(pasta_cache: Path | None = None) -> dict[str, SequenciaPose]:
    pasta = Path(pasta_cache or PASTA_CACHE)
    return {c.stem.replace("-cam0", ""): SequenciaPose.carregar(c)
            for c in sorted(pasta.glob("*-cam0.npz"))}


def avaliar(parametros: quedas.ParametrosQueda | None = None,
            sequencias: dict[str, SequenciaPose] | None = None,
            rotulos: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    parametros = parametros or quedas.ParametrosQueda()
    sequencias = sequencias if sequencias is not None else sequencias_em_cache()
    rotulos = rotulos if rotulos is not None else gabarito()
    resultado: dict[str, Any] = {"dataset": "UR Fall Detection (video RGB, cam0)",
                                 "parametros": parametros.__dict__, "particoes": {}}
    for particao, resto in (("calibracao (impares)", 1), ("teste (pares)", 0)):
        vp = fn = fp = vn = 0
        latencias, erros = [], []
        deitadas_sem_alarme = deitadas = 0
        for nome, sequencia in sequencias.items():
            if nome not in rotulos or int(nome.split("-")[1]) % 2 != resto:
                continue
            eventos = quedas.detectar(sequencia, cin.calcular(sequencia), parametros)
            detectou = any(e.tipo == "queda" for e in eventos)
            rotulo = rotulos[nome]
            if rotulo["queda"]:
                vp += detectou
                fn += not detectou
                if detectou and rotulo["quadro_inicio"]:
                    primeira = next(e for e in eventos if e.tipo == "queda")
                    latencias.append(primeira.fim_s - (rotulo["quadro_inicio"] - 1) / sequencia.fps)
            else:
                fp += detectou
                vn += not detectou
                if rotulo["deita"]:
                    deitadas += 1
                    deitadas_sem_alarme += not detectou
            if detectou != rotulo["queda"]:
                erros.append(nome)
        resultado["particoes"][particao] = {
            "quedas": vp + fn, "atividades": fp + vn,
            "sensibilidade": round(vp / max(vp + fn, 1), 3),
            "especificidade": round(vn / max(vn + fp, 1), 3),
            "atividades_deitando_sem_alarme": f"{deitadas_sem_alarme}/{deitadas}",
            "latencia_mediana_s": round(statistics.median(latencias), 2) if latencias else None,
            "vp": vp, "fn": fn, "fp": fp, "vn": vn, "erros": erros,
        }
    return resultado


def varrer(sequencias: dict[str, SequenciaPose], rotulos: dict[str, dict[str, Any]],
           grade: dict[str, list[float]]) -> list[dict[str, Any]]:
    """Busca em grade, avaliada so na particao de calibracao."""
    from itertools import product

    calibracao = {n: s for n, s in sequencias.items() if int(n.split("-")[1]) % 2 == 1}
    linhas = []
    chaves = list(grade)
    for valores in product(*grade.values()):
        parametros = replace(quedas.ParametrosQueda(), **dict(zip(chaves, valores, strict=True)))
        dados = avaliar(parametros, calibracao, rotulos)["particoes"]["calibracao (impares)"]
        linhas.append({**dict(zip(chaves, valores, strict=True)),
                       "sensibilidade": dados["sensibilidade"],
                       "especificidade": dados["especificidade"]})
    return sorted(linhas, key=lambda d: (-(d["sensibilidade"] + d["especificidade"]),
                                         -d["especificidade"]))
