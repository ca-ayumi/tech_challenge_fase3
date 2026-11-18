"""Relogio do monitoramento.

Em producao, o "agora" e a hora do sistema. Na demonstracao, os prontuarios
sinteticos tem uma linha do tempo propria (gerada junto com eles); se audio e
video fossem carimbados com a hora real, cairiam semanas depois dos sinais
vitais e nenhuma correlacao entre modalidades seria possivel. Com
``RELOGIO=cenario`` (padrao), o "agora" e o instante mais recente registrado no
prontuario: a ultima afericao de sinais vitais.
"""

from __future__ import annotations

from datetime import datetime

from ..config import MONITORAMENTO


def agora(banco=None) -> datetime:
    if MONITORAMENTO.relogio == "cenario" and banco is not None:
        linhas = banco.consultar_somente_leitura(
            "SELECT max(aferido_em) AS ultimo FROM sinais_vitais")
        if linhas and linhas[0]["ultimo"]:
            return datetime.fromisoformat(linhas[0]["ultimo"])
    return datetime.now().replace(microsecond=0)
