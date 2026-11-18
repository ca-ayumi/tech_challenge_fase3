"""Termos criticos ditos na consulta.

O catalogo define o que a equipe precisa saber de imediato quando aparece na
fala do paciente. Ele e usado de dois jeitos:

- para classificar as entidades do Text Analytics for Health (por texto ou
  por codigo UMLS), respeitando a assercao do servico: "nego dor no peito" vem
  marcado como negado e nao vira alerta;
- como detector local de reserva, quando a Azure nao esta disponivel, com uma
  regra simples de negacao (ate quatro palavras antes do termo).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from ..azure.linguagem import EntidadeSaude


@dataclass(frozen=True)
class TermoCritico:
    chave: str
    rotulo: str
    severidade: str
    padroes: tuple[str, ...]
    umls: tuple[str, ...] = ()


CATALOGO = (
    TermoCritico("dor_toracica", "dor toracica", "critica",
                 ("dor no peito", "dor toracica", "aperto no peito", "peito apertado",
                  "dor precordial", "dor no coracao"), ("C0008031", "C2926613")),
    TermoCritico("deficit_neurologico", "sinal neurologico focal", "critica",
                 ("fala enrolada", "fala arrastada", "boca torta", "fraqueza no braco",
                  "braco fraco", "braco direito fraco", "braco esquerdo fraco", "perna fraca",
                  "dormencia de um lado", "disartria", "hemiparesia", "nao consigo mexer"),
                 ("C0013362", "C0018989", "C0018991")),
    TermoCritico("sincope", "desmaio", "critica",
                 ("desmaio", "desmaiei", "sincope", "perdi a consciencia", "apaguei"),
                 ("C0039070",)),
    TermoCritico("ideacao_suicida", "ideacao suicida", "critica",
                 ("vontade de morrer", "tirar minha vida", "me matar", "acabar com tudo"),
                 ("C0424000",)),
    TermoCritico("convulsao", "convulsao", "critica",
                 ("convulsao", "convulsionei", "crise convulsiva", "ataque epileptico"),
                 ("C0036572", "C0014544")),
    TermoCritico("dispneia", "falta de ar", "alta",
                 ("falta de ar", "dispneia", "dificuldade para respirar", "sem folego",
                  "ofegante", "nao consigo respirar", "cansaco para respirar",
                  "cansado para respirar", "cansada para respirar"), ("C0013404",)),
    TermoCritico("sangramento", "sangramento", "alta",
                 ("sangramento", "sangrando", "vomitei sangue", "sangue nas fezes",
                  "fezes pretas", "tossindo sangue", "hemoptise"), ("C0019080", "C0019079")),
    TermoCritico("confusao", "confusao mental", "alta",
                 ("confuso", "confusa", "confusao", "desorientado", "desorientada",
                  "nao sei onde estou"), ("C0009676",)),
    TermoCritico("febre", "febre", "moderada",
                 ("febre", "febril", "calafrio", "calafrios"), ("C0015967",)),
    TermoCritico("fadiga", "cansaco", "moderada",
                 ("cansaco", "cansado", "cansada", "exausto", "exausta", "sem forca",
                  "sem energia", "fadiga", "fraqueza"), ("C0015672", "C0004093")),
)

NEGACOES = {"nao", "nega", "nego", "negou", "sem", "nenhum", "nenhuma", "nunca", "nem"}
JANELA_NEGACAO = 4


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto or "")
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"[a-z0-9]+", sem_acento))


INTERCALADAS = 2


def _expressao(padrao: str) -> re.Pattern[str]:
    """Padrao com ate duas palavras intercaladas: 'fala enrolada' casa 'fala estava enrolada'."""
    ligacao = rf"(?:\s+[a-z0-9]+){{0,{INTERCALADAS}}}\s+"
    return re.compile(r"\b" + ligacao.join(re.escape(p) for p in padrao.split()) + r"\b")


EXPRESSOES = {termo.chave: [(_expressao(p), p) for p in termo.padroes] for termo in CATALOGO}


def _casa(texto_normalizado: str, termo: TermoCritico) -> bool:
    if any(expressao.search(texto_normalizado) for expressao, _ in EXPRESSOES[termo.chave]):
        return True
    # Entidade parcial ("dor no" de "dor no peito"): so com duas palavras ou mais, para que
    # "cansado" nao case com "cansado para respirar".
    return len(texto_normalizado.split()) >= 2 and any(
        p.startswith(texto_normalizado) for p in termo.padroes)


@dataclass(frozen=True)
class Mencao:
    termo: TermoCritico
    trecho: str
    negada: bool
    condicional: bool = False
    fonte: str = "local"

    def como_dicionario(self) -> dict:
        return {"chave": self.termo.chave, "rotulo": self.termo.rotulo,
                "severidade": self.termo.severidade, "trecho": self.trecho,
                "negada": self.negada, "condicional": self.condicional, "fonte": self.fonte}


def classificar_entidades(entidades: Iterable[EntidadeSaude]) -> list[Mencao]:
    """Associa as entidades do Text Analytics for Health ao catalogo."""
    mencoes = []
    for entidade in entidades:
        if entidade.categoria not in {"SymptomOrSign", "Diagnosis", "ConditionQualifier"}:
            continue
        normalizado = normalizar(entidade.texto)
        codigos = {c.split(":", 1)[-1] for c in entidade.codigos}
        for termo in CATALOGO:
            if _casa(normalizado, termo) or codigos & set(termo.umls):
                mencoes.append(Mencao(termo, entidade.texto, entidade.negada,
                                      entidade.condicionalidade is not None, "azure_health"))
                break
    return mencoes


def detectar_local(texto: str) -> list[Mencao]:
    """Detector de reserva: busca os padroes do catalogo e aplica a regra de negacao."""
    palavras = normalizar(texto).split()
    normalizado = " ".join(palavras)
    mencoes = []
    vistos = set()
    ocupados: list[tuple[int, int]] = []
    for termo in CATALOGO:
        for expressao, _ in EXPRESSOES[termo.chave]:
            for achado in expressao.finditer(normalizado):
                if any(achado.start() < fim and inicio < achado.end() for inicio, fim in ocupados):
                    continue
                ocupados.append(achado.span())
                indice = normalizado[:achado.start()].count(" ")
                anteriores = palavras[max(0, indice - JANELA_NEGACAO):indice]
                negada = any(p in NEGACOES for p in anteriores)
                chave = (termo.chave, negada)
                if chave in vistos:
                    continue
                vistos.add(chave)
                mencoes.append(Mencao(termo, achado.group(0), negada, fonte="local"))
    return mencoes


def consolidar(mencoes: Iterable[Mencao]) -> list[Mencao]:
    """Uma mencao por termo: a presente prevalece sobre a negada; Azure antes da local.

    A Azure as vezes devolve a entidade cortada ("dor no" de "dor no peito"); se o
    detector local achou um trecho mais completo do mesmo termo, ele e usado.
    """
    mencoes = list(mencoes)
    escolhidas: dict[str, Mencao] = {}
    for mencao in mencoes:
        atual = escolhidas.get(mencao.termo.chave)
        if atual is None or (atual.negada and not mencao.negada):
            escolhidas[mencao.termo.chave] = mencao
    for chave, escolhida in escolhidas.items():
        mais_completa = max((m for m in mencoes if m.termo.chave == chave
                             and m.negada == escolhida.negada), key=lambda m: len(m.trecho))
        if len(mais_completa.trecho) > len(escolhida.trecho):
            escolhidas[chave] = Mencao(escolhida.termo, mais_completa.trecho, escolhida.negada,
                                       escolhida.condicional, escolhida.fonte)
    return list(escolhidas.values())
