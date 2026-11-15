"""Transforma checagens ponto a ponto em episodios de anomalia.

Um detector online avalia cada leitura, mas a equipe precisa de um aviso por
*episodio*, nao por leitura. O gerenciador:

- exige persistencia: a condicao precisa se manter por um tempo minimo antes de
  abrir o episodio (um pico isolado de artefato nao abre);
- emite na abertura e de novo apenas se a severidade piorar;
- fecha o episodio depois de ``tolerancia_normal`` checagens normais seguidas.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..tipos import mais_grave


@dataclass
class Episodio:
    inicio: datetime
    severidade: str
    ultimo: datetime
    normais: int = 0


class GerenciadorEpisodios:
    def __init__(self, tolerancia_normal: int = 3) -> None:
        self.tolerancia_normal = tolerancia_normal
        self._pendentes: dict[Hashable, datetime] = {}
        self._ativos: dict[Hashable, Episodio] = {}

    def ativo(self, chave: Hashable) -> bool:
        return chave in self._ativos

    def positivo(self, chave: Hashable, instante: datetime, severidade: str,
                 persistencia: timedelta = timedelta(0)) -> datetime | None:
        """Registra a condicao anormal. Devolve o inicio do episodio se for hora de emitir."""
        episodio = self._ativos.get(chave)
        if episodio is not None:
            episodio.ultimo = instante
            episodio.normais = 0
            if mais_grave(severidade, episodio.severidade):
                episodio.severidade = severidade
                return episodio.inicio
            return None

        inicio = self._pendentes.setdefault(chave, instante)
        if instante - inicio < persistencia:
            return None
        del self._pendentes[chave]
        self._ativos[chave] = Episodio(inicio=inicio, severidade=severidade, ultimo=instante)
        return inicio

    def negativo(self, chave: Hashable) -> None:
        """Registra checagem normal: cancela pendencia e conta para fechar o episodio."""
        self._pendentes.pop(chave, None)
        episodio = self._ativos.get(chave)
        if episodio is None:
            return
        episodio.normais += 1
        if episodio.normais >= self.tolerancia_normal:
            del self._ativos[chave]
