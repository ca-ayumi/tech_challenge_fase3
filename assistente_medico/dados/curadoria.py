"""Curadoria do corpus de treinamento.

Aplica, nesta ordem: filtros de qualidade, verificacao de PII residual,
deduplicacao exata e aproximada, e divisao estratificada em treino, validacao e
teste. Todo exemplo descartado e contabilizado com o motivo, e o relatorio final
vai para o relatorio tecnico.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..finetuning.formato import segue_formato

# Sinais de que a anonimizacao falhou e sobrou identificador direto no texto.
PADROES_PII_RESIDUAL = [
    ("cpf", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("cns", re.compile(r"\b\d{3}\s\d{4}\s\d{4}\s\d{4}\b")),
    ("telefone", re.compile(r"\(\d{2}\)\s?9?\d{4}-\d{4}")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
]

TAMANHO_MINIMO_PERGUNTA = 12
TAMANHO_MINIMO_RESPOSTA = 60
TAMANHO_MAXIMO_RESPOSTA = 4000


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto.lower())
    return re.sub(r"\s+", " ", texto).strip()


def _shingles(texto: str, tamanho: int = 5) -> set[str]:
    tokens = _normalizar(texto).split()
    if len(tokens) < tamanho:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + tamanho]) for i in range(len(tokens) - tamanho + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersecao = len(a & b)
    return intersecao / (len(a) + len(b) - intersecao)


@dataclass
class RelatorioCuradoria:
    """Contagens de tudo que entrou, saiu e por que saiu."""

    total_entrada: int = 0
    total_saida: int = 0
    descartados: Counter = field(default_factory=Counter)
    por_categoria: Counter = field(default_factory=Counter)
    duplicatas_exatas: int = 0
    duplicatas_aproximadas: int = 0
    pii_residual: Counter = field(default_factory=Counter)
    tamanhos: dict[str, float] = field(default_factory=dict)

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "total_entrada": self.total_entrada,
            "total_saida": self.total_saida,
            "descartados": dict(self.descartados),
            "por_categoria": dict(self.por_categoria),
            "duplicatas_exatas": self.duplicatas_exatas,
            "duplicatas_aproximadas": self.duplicatas_aproximadas,
            "pii_residual": dict(self.pii_residual),
            "tamanhos": self.tamanhos,
        }


def detectar_pii_residual(texto: str) -> list[str]:
    """Lista os tipos de identificador direto que sobraram no texto."""
    return [tipo for tipo, padrao in PADROES_PII_RESIDUAL if padrao.search(texto or "")]


def _motivo_descarte(exemplo: dict[str, Any], exigir_formato: bool) -> str | None:
    pergunta = exemplo.get("pergunta", "")
    resposta = exemplo.get("resposta", "")

    if len(pergunta.strip()) < TAMANHO_MINIMO_PERGUNTA:
        return "pergunta_curta"
    if len(resposta.strip()) < TAMANHO_MINIMO_RESPOSTA:
        return "resposta_curta"
    if len(resposta) > TAMANHO_MAXIMO_RESPOSTA:
        return "resposta_longa"
    if exigir_formato and not segue_formato(resposta):
        return "formato_invalido"
    if not exemplo.get("fontes") and exemplo.get("categoria") not in {"fora_escopo"}:
        return "sem_fonte"
    return None


def curar(
    exemplos: Iterable[dict[str, Any]],
    limiar_similaridade: float = 0.92,
    exigir_formato: bool = True,
) -> tuple[list[dict[str, Any]], RelatorioCuradoria]:
    """Filtra, verifica PII e remove duplicatas. Devolve o corpus limpo e o relatorio."""
    relatorio = RelatorioCuradoria()
    aprovados: list[dict[str, Any]] = []
    vistos_exatos: set[str] = set()
    assinaturas: list[tuple[set[str], str]] = []

    for exemplo in exemplos:
        relatorio.total_entrada += 1

        motivo = _motivo_descarte(exemplo, exigir_formato)
        if motivo:
            relatorio.descartados[motivo] += 1
            continue

        texto_completo = f"{exemplo.get('pergunta','')}\n{exemplo.get('resposta','')}"
        tipos_pii = detectar_pii_residual(texto_completo)
        if tipos_pii:
            for tipo in tipos_pii:
                relatorio.pii_residual[tipo] += 1
            relatorio.descartados["pii_residual"] += 1
            continue

        chave = hashlib.sha256(_normalizar(texto_completo).encode("utf-8")).hexdigest()
        if chave in vistos_exatos:
            relatorio.duplicatas_exatas += 1
            relatorio.descartados["duplicata_exata"] += 1
            continue

        # Deduplicacao aproximada dentro da mesma categoria, para nao penalizar
        # categorias que sao naturalmente repetitivas entre si.
        categoria = exemplo.get("categoria", "geral")
        shingles = _shingles(texto_completo)
        duplicada = False
        for assinatura, categoria_existente in assinaturas:
            if categoria_existente != categoria:
                continue
            if jaccard(shingles, assinatura) >= limiar_similaridade:
                duplicada = True
                break
        if duplicada:
            relatorio.duplicatas_aproximadas += 1
            relatorio.descartados["duplicata_aproximada"] += 1
            continue

        vistos_exatos.add(chave)
        assinaturas.append((shingles, categoria))
        relatorio.por_categoria[categoria] += 1
        aprovados.append(exemplo)

    relatorio.total_saida = len(aprovados)
    if aprovados:
        tamanhos_resposta = [len(e.get("resposta", "")) for e in aprovados]
        tamanhos_pergunta = [len(e.get("pergunta", "")) for e in aprovados]
        relatorio.tamanhos = {
            "resposta_media": round(sum(tamanhos_resposta) / len(tamanhos_resposta), 1),
            "resposta_maxima": max(tamanhos_resposta),
            "pergunta_media": round(sum(tamanhos_pergunta) / len(tamanhos_pergunta), 1),
        }
    return aprovados, relatorio


def dividir(
    exemplos: list[dict[str, Any]],
    proporcao_validacao: float = 0.1,
    proporcao_teste: float = 0.1,
    semente: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Divide o corpus mantendo a proporcao de categorias em cada particao."""
    aleatorio = random.Random(semente)
    por_categoria: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exemplo in exemplos:
        por_categoria[exemplo.get("categoria", "geral")].append(exemplo)

    particoes: dict[str, list[dict[str, Any]]] = {"treino": [], "validacao": [], "teste": []}
    for categoria in sorted(por_categoria):
        grupo = por_categoria[categoria][:]
        aleatorio.shuffle(grupo)
        total = len(grupo)
        n_validacao = max(1, round(total * proporcao_validacao)) if total >= 8 else 0
        n_teste = max(1, round(total * proporcao_teste)) if total >= 8 else 0
        particoes["validacao"].extend(grupo[:n_validacao])
        particoes["teste"].extend(grupo[n_validacao:n_validacao + n_teste])
        particoes["treino"].extend(grupo[n_validacao + n_teste:])

    for particao in particoes.values():
        aleatorio.shuffle(particao)
    return particoes
