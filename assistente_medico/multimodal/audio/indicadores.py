"""Indicadores vocais de fadiga, dispneia e disartria.

Cada indicador e um conjunto de criterios sobre as medidas acusticas,
comparadas a linha de base do proprio paciente (gravacao anterior, por
exemplo na admissao). Sem linha de base, usa faixas de referencia para fala
espontanea de adultos. Os criterios seguem o que a literatura descreve:

- fadiga: voz mais fraca (obrigatorio), com fala mais lenta, pausas longas ou
  menos variacao de tom;
- dispneia: frases curtas entre respiracoes e mais pausas, com a articulacao
  preservada (a pessoa fala no ritmo normal, mas precisa parar para respirar);
- disartria: articulacao muito lenta (obrigatorio), com fala fragmentada ou
  ritmo irregular entre silabas.

Limiares calibrados em tres vozes sinteticas e avaliados em outras tres.

O resultado e uma suspeita com os criterios que a sustentam, para avaliacao
da equipe; nao e diagnostico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .acustica import MedidasVoz

REFERENCIA = {
    "taxa_articulacao": 4.5,
    "silabas_por_grupo": 9.0,
    "pausas_por_minuto": 14.0,
    "pausa_media_s": 0.6,
    "f0_variacao_st": 3.5,
    "irregularidade_ritmo": 0.35,
    "intensidade_media_db": None,
}

CONDICOES = {
    "fadiga": ("Fala sugestiva de fadiga", "moderada"),
    "dispneia": ("Fala sugestiva de dificuldade respiratoria", "alta"),
    "disartria": ("Fala sugestiva de disartria", "alta"),
}


@dataclass
class Indicador:
    condicao: str
    escore: float
    criterios: list[str] = field(default_factory=list)
    suspeita: bool = False

    def como_dicionario(self) -> dict[str, Any]:
        return {"condicao": self.condicao, "escore": round(self.escore, 2),
                "suspeita": self.suspeita, "criterios": self.criterios}


def _variacao(atual: float | None, base: float | None) -> float | None:
    if atual is None or base is None or base == 0:
        return None
    return (atual - base) / abs(base)


def comparar(atual: MedidasVoz, base: MedidasVoz | None = None) -> dict[str, float | None]:
    """Variacao relativa de cada medida (ou diferenca em dB para intensidade)."""
    referencia = {k: (getattr(base, k) if base else v) for k, v in REFERENCIA.items()}
    variacoes = {k: _variacao(getattr(atual, k), referencia[k])
                 for k in REFERENCIA if k != "intensidade_media_db"}
    variacoes["intensidade_media_db"] = (atual.intensidade_media_db - base.intensidade_media_db
                                         if base else None)
    return variacoes


def avaliar(atual: MedidasVoz, base: MedidasVoz | None = None) -> list[Indicador]:
    v = comparar(atual, base)

    def abaixo(chave: str, limite: float) -> bool:
        return v.get(chave) is not None and v[chave] <= limite

    def acima(chave: str, limite: float) -> bool:
        return v.get(chave) is not None and v[chave] >= limite

    def pct(chave: str) -> str:
        return f"{v[chave]:+.0%}"

    fadiga = Indicador("fadiga", 0.0)
    voz_fraca = abaixo("intensidade_media_db", -3.0)
    if voz_fraca:
        fadiga.criterios.append(f"voz mais fraca ({v['intensidade_media_db']:+.1f} dB)")
    if abaixo("taxa_articulacao", -0.12):
        fadiga.criterios.append(f"articulacao mais lenta ({pct('taxa_articulacao')})")
    if acima("pausa_media_s", 0.4):
        fadiga.criterios.append(f"pausas mais longas ({pct('pausa_media_s')})")
    if abaixo("f0_variacao_st", -0.25):
        fadiga.criterios.append(f"menos variacao de tom ({pct('f0_variacao_st')})")
    fadiga.escore = len(fadiga.criterios) / 4
    fadiga.suspeita = voz_fraca and len(fadiga.criterios) >= 2

    dispneia = Indicador("dispneia", 0.0)
    if abaixo("silabas_por_grupo", -0.55):
        dispneia.criterios.append(f"frases curtas entre respiracoes ({pct('silabas_por_grupo')})")
    if acima("pausas_por_minuto", 1.0):
        dispneia.criterios.append(f"mais pausas por minuto ({pct('pausas_por_minuto')})")
    if v.get("taxa_articulacao") is None or v["taxa_articulacao"] > -0.25:
        dispneia.criterios.append("articulacao preservada")
    dispneia.escore = len(dispneia.criterios) / 3
    dispneia.suspeita = len(dispneia.criterios) == 3

    disartria = Indicador("disartria", 0.0)
    lenta = abaixo("taxa_articulacao", -0.3)
    if lenta:
        disartria.criterios.append(f"articulacao muito lenta ({pct('taxa_articulacao')})")
    fragmentada = abaixo("silabas_por_grupo", -0.4)
    if fragmentada:
        disartria.criterios.append(f"fala fragmentada ({pct('silabas_por_grupo')})")
    irregular = acima("irregularidade_ritmo", 0.3)
    if irregular:
        disartria.criterios.append(f"ritmo silabico irregular ({pct('irregularidade_ritmo')})")
    disartria.escore = len(disartria.criterios) / 3
    disartria.suspeita = lenta and (fragmentada or irregular)

    return [fadiga, dispneia, disartria]
