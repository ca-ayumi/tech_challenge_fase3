"""Areas criticas e objetos de risco, com as deteccoes do YOLOv8.

A configuracao fica num JSON por camera: poligonos em coordenadas relativas
(0 a 1) da imagem, com nome, severidade e tempo minimo de permanencia, e a
lista de classes do COCO consideradas de risco perto do paciente. Exemplo:

    {"zonas": [{"nome": "porta", "poligono": [[0.7, 0.1], [0.85, 0.1],
                [0.85, 0.75], [0.7, 0.75]], "severidade": "alta", "permanencia_s": 1.0}],
     "objetos_risco": ["knife", "scissors"]}

A posicao da pessoa e o ponto medio dos tornozelos (ou a base da caixa): e o
que toca o chao, e nao muda quando a pessoa levanta os bracos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cinematica import Cinematica
from .extracao import SequenciaPose

OBJETOS_RISCO_PADRAO = ("knife", "scissors")


@dataclass(frozen=True)
class Zona:
    nome: str
    poligono: tuple[tuple[float, float], ...]
    severidade: str = "alta"
    permanencia_s: float = 1.0


@dataclass
class ConfigAreas:
    zonas: list[Zona] = field(default_factory=list)
    objetos_risco: tuple[str, ...] = OBJETOS_RISCO_PADRAO

    @classmethod
    def carregar(cls, caminho: str | Path | None) -> ConfigAreas:
        if not caminho:
            return cls()
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
        return cls(
            zonas=[Zona(z["nome"], tuple(tuple(p) for p in z["poligono"]),
                        z.get("severidade", "alta"), float(z.get("permanencia_s", 1.0)))
                   for z in dados.get("zonas", [])],
            objetos_risco=tuple(dados.get("objetos_risco", OBJETOS_RISCO_PADRAO)),
        )


def dentro(ponto: tuple[float, float], poligono: tuple[tuple[float, float], ...]) -> bool:
    """Ray casting: ponto dentro do poligono."""
    x, y = ponto
    resultado = False
    for (x1, y1), (x2, y2) in zip(poligono, poligono[1:] + poligono[:1], strict=True):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            resultado = not resultado
    return resultado


@dataclass
class EventoArea:
    tipo: str
    severidade: str
    descricao: str
    inicio_s: float
    fim_s: float
    evidencias: dict[str, Any] = field(default_factory=dict)


def detectar(sequencia: SequenciaPose, cinematica: Cinematica,
             config: ConfigAreas) -> list[EventoArea]:
    eventos: list[EventoArea] = []
    t = cinematica.tempos
    relativos = cinematica.pe / np.array([sequencia.largura, sequencia.altura])
    for zona in config.zonas:
        dentro_zona = np.array([not np.isnan(p).any() and dentro((p[0], p[1]), zona.poligono)
                                for p in relativos])
        inicio = None
        for i, presente in enumerate(np.append(dentro_zona, False)):
            if presente and inicio is None:
                inicio = i
            elif not presente and inicio is not None:
                fim = i - 1
                if t[fim] - t[inicio] >= zona.permanencia_s:
                    eventos.append(EventoArea(
                        "area_critica", zona.severidade,
                        f"Paciente na area critica '{zona.nome}' por {t[fim] - t[inicio]:.1f} s",
                        float(t[inicio]), float(t[fim]), {"zona": zona.nome}))
                inicio = None

    contagem: dict[str, list[int]] = {}
    for amostra in sequencia.objetos:
        for objeto in amostra["objetos"]:
            if objeto["classe"] in config.objetos_risco:
                contagem.setdefault(objeto["classe"], []).append(amostra["quadro"])
    for classe, quadros in contagem.items():
        if len(quadros) >= 2:
            eventos.append(EventoArea(
                "objeto_risco", "alta",
                f"Objeto de risco ({classe}) visivel em {len(quadros)} amostras do video",
                float(t[min(quadros[0], len(t) - 1)]), float(t[min(quadros[-1], len(t) - 1)]),
                {"classe": classe, "amostras": len(quadros)}))
    return eventos


def objetos_vistos(sequencia: SequenciaPose, confianca_minima: float = 0.5) -> dict[str, int]:
    """Quantas amostras contem cada classe (contexto para o relatorio)."""
    vistos: dict[str, int] = {}
    for amostra in sequencia.objetos:
        for classe in {o["classe"] for o in amostra["objetos"] if o["confianca"] >= confianca_minima}:
            vistos[classe] = vistos.get(classe, 0) + 1
    return dict(sorted(vistos.items(), key=lambda item: -item[1]))
