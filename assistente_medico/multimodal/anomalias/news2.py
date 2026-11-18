"""National Early Warning Score 2 (Royal College of Physicians, 2017).

Escala 1 de SpO2 (sem hipercapnia cronica). O paciente em ar ambiente e o
padrao, porque o dado de oxigenio suplementar nem sempre vem no fluxo.
Quando faltam parametros, o escore e *parcial* e assim e rotulado: ele
subestima o risco, nunca o superestima.
"""

from __future__ import annotations

from collections.abc import Mapping

PARAMETROS = ("fr", "saturacao", "pas", "fc", "temperatura", "glasgow")


def _faixa(valor: float, faixas: list[tuple[float, int]], acima: int) -> int:
    for limite_superior, pontos in faixas:
        if valor <= limite_superior:
            return pontos
    return acima


def subescore(parametro: str, valor: float | None) -> int | None:
    if valor is None:
        return None
    if parametro == "fr":
        return _faixa(valor, [(8, 3), (11, 1), (20, 0), (24, 2)], 3)
    if parametro == "saturacao":
        return _faixa(valor, [(91, 3), (93, 2), (95, 1)], 0)
    if parametro == "pas":
        return _faixa(valor, [(90, 3), (100, 2), (110, 1), (219, 0)], 3)
    if parametro == "fc":
        return _faixa(valor, [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2)], 3)
    if parametro == "temperatura":
        return _faixa(valor, [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1)], 2)
    if parametro == "glasgow":
        return 0 if valor >= 15 else 3
    return None


def calcular(valores: Mapping[str, float | None]) -> dict:
    """Escore agregado, subescores e faixa de risco."""
    subescores = {p: subescore(p, valores.get(p)) for p in PARAMETROS}
    presentes = {p: s for p, s in subescores.items() if s is not None}
    total = sum(presentes.values())
    if total >= 7:
        faixa = "alto"
    elif total >= 5:
        faixa = "medio"
    elif any(s == 3 for s in presentes.values()):
        faixa = "baixo_medio"
    else:
        faixa = "baixo"
    return {
        "total": total,
        "faixa": faixa,
        "subescores": presentes,
        "parcial": len(presentes) < len(PARAMETROS),
        "parametros_ausentes": [p for p in PARAMETROS if p not in presentes],
    }
