"""Anonimizacao e pseudonimizacao de dados pessoais em textos clinicos.

Duas estrategias sao aplicadas, conforme o tipo de identificador:

- ``remover``: identificadores diretos que nao tem utilidade analitica
  (CPF, CNS, RG, telefone, e-mail, endereco, CEP, datas exatas) sao
  substituidos por um rotulo generico, por exemplo ``[CPF]``.
- ``pseudonimizar``: identificadores que precisam manter consistencia entre
  trechos (nome do paciente, numero de prontuario, CRM do profissional) sao
  substituidos por um codigo estavel derivado por HMAC-SHA256 com sal secreto,
  por exemplo ``[PACIENTE_7F3A]``. O mesmo valor gera sempre o mesmo codigo,
  o que preserva a coerencia do texto sem permitir reidentificacao direta.

Referencia institucional: PROT-GOV-010 §8 (protecao de dados).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

Estrategia = Literal["remover", "pseudonimizar", "manter"]

SAL_PADRAO = "sal-de-desenvolvimento-trocar-em-producao"

ROTULOS = {
    "cpf": "CPF",
    "cns": "CNS",
    "rg": "RG",
    "telefone": "TELEFONE",
    "email": "EMAIL",
    "cep": "CEP",
    "endereco": "ENDERECO",
    "data": "DATA",
    "nome_paciente": "PACIENTE",
    "nome_profissional": "PROFISSIONAL",
    "prontuario": "PRONTUARIO",
    "crm": "CRM",
}

ESTRATEGIA_PADRAO: dict[str, Estrategia] = {
    "cpf": "remover",
    "cns": "remover",
    "rg": "remover",
    "telefone": "remover",
    "email": "remover",
    "cep": "remover",
    "endereco": "remover",
    "data": "remover",
    "nome_paciente": "pseudonimizar",
    "nome_profissional": "pseudonimizar",
    "prontuario": "pseudonimizar",
    "crm": "pseudonimizar",
}

PADROES: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("cns", re.compile(r"\b\d{3}[\s.]?\d{4}[\s.]?\d{4}[\s.]?\d{4}\b")),
    ("cpf", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("crm", re.compile(r"\bCRM[-/\s]?[A-Z]{2}\s?\d{2,3}\.?\d{3}\b", re.IGNORECASE)),
    ("telefone", re.compile(r"\(?\b\d{2}\)?[\s.-]?9?\d{4}[-.\s]?\d{4}\b")),
    ("cep", re.compile(r"\bCEP\s?\d{5}-?\d{3}\b|\b\d{5}-\d{3}\b", re.IGNORECASE)),
    ("rg", re.compile(r"\b\d{2}\.\d{3}\.\d{3}-[0-9Xx]\b")),
    (
        "endereco",
        re.compile(
            r"\b(?:Rua|Avenida|Av\.|Alameda|Travessa|Praca|Praça|Rodovia|Estrada)\s+"
            r"[A-Za-zÀ-ÿ0-9'.\s]{2,50},?\s*\d{1,5}[^\n,;]{0,40}",
            re.IGNORECASE,
        ),
    ),
    ("data", re.compile(r"\b\d{2}/\d{2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")),
    (
        "prontuario",
        re.compile(
            r"\b(?:prontu[aá]rio|registro hospitalar|registro)\s*(?:n[ºo°]?\.?\s*)?(\d{4,8})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "nome_profissional",
        re.compile(
            r"\b(?:Dr\.?|Dra\.?|Prof\.?|Enf\.?)\s+"
            r"(?:[A-ZÀ-Ý][a-zà-ÿ']+(?:\s+(?:d[aeoi]s?|e)\s+|\s+)){1,4}[A-ZÀ-Ý][a-zà-ÿ']+"
        ),
    ),
]

NAO_SAO_NOMES = {
    "pronto", "socorro", "unidade", "terapia", "intensiva", "enfermaria", "hospital",
    "protocolo", "paciente", "medico", "medica", "escala", "glasgow", "sao", "santo",
    "santa", "janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho",
    "agosto", "setembro", "outubro", "novembro", "dezembro", "segunda", "terca",
    "quarta", "quinta", "sexta", "sabado", "domingo", "leito", "exame", "resultado",
    "lei", "geral", "resolucao", "comissao", "controle", "infeccao", "hospitalar",
}


def _normalizar(texto: str) -> str:
    sem_acentos = unicodedata.normalize("NFKD", texto)
    sem_acentos = "".join(c for c in sem_acentos if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acentos).strip().lower()


@dataclass(frozen=True)
class Ocorrencia:
    """Um identificador encontrado no texto."""

    tipo: str
    texto: str
    inicio: int
    fim: int
    estrategia: Estrategia
    substituto: str


@dataclass
class ResultadoAnonimizacao:
    """Texto anonimizado e o rastro do que foi alterado."""

    texto: str
    ocorrencias: list[Ocorrencia] = field(default_factory=list)

    @property
    def contagem_por_tipo(self) -> dict[str, int]:
        contagem: dict[str, int] = {}
        for ocorrencia in self.ocorrencias:
            contagem[ocorrencia.tipo] = contagem.get(ocorrencia.tipo, 0) + 1
        return contagem

    @property
    def total(self) -> int:
        return len(self.ocorrencias)


class Anonimizador:
    """Remove ou pseudonimiza dados pessoais em textos livres.

    Args:
        sal: segredo usado no HMAC da pseudonimizacao. Em producao vem da
            variavel de ambiente ``SAL_ANONIMIZACAO``.
        nomes_conhecidos: nomes do cadastro (pacientes, profissionais) que
            devem ser reconhecidos mesmo sem titulo antecedendo.
        estrategias: sobrescreve a estrategia padrao por tipo.
    """

    def __init__(
        self,
        sal: str | None = None,
        nomes_conhecidos: Iterable[str] = (),
        estrategias: dict[str, Estrategia] | None = None,
    ) -> None:
        self.sal = (sal or os.getenv("SAL_ANONIMIZACAO") or SAL_PADRAO).encode("utf-8")
        self.estrategias = {**ESTRATEGIA_PADRAO, **(estrategias or {})}
        self._nomes_conhecidos: dict[str, str] = {}
        self._padrao_nomes: re.Pattern[str] | None = None
        self.registrar_nomes(nomes_conhecidos)

    def registrar_nomes(self, nomes: Iterable[str], tipo: str = "nome_paciente") -> None:
        """Adiciona nomes do cadastro a serem reconhecidos no texto livre."""
        novos = {n.strip(): tipo for n in nomes if n and len(n.strip()) > 3}
        if not novos:
            self._compilar_nomes()
            return
        self._nomes_conhecidos.update(novos)
        self._compilar_nomes()

    def _compilar_nomes(self) -> None:
        if not self._nomes_conhecidos:
            self._padrao_nomes = None
            return
        alternativas = sorted(self._nomes_conhecidos, key=len, reverse=True)
        partes = []
        for nome in alternativas:
            partes.append(re.escape(nome))
            tokens = nome.split()
            if len(tokens) >= 2:
                partes.append(re.escape(f"{tokens[0]} {tokens[-1]}"))
        self._padrao_nomes = re.compile(r"\b(?:" + "|".join(partes) + r")\b")

    def pseudonimo(self, tipo: str, valor: str) -> str:
        """Codigo estavel para um valor: o mesmo valor gera sempre o mesmo codigo."""
        digest = hmac.new(self.sal, _normalizar(valor).encode("utf-8"), hashlib.sha256)
        return f"[{ROTULOS.get(tipo, tipo.upper())}_{digest.hexdigest()[:4].upper()}]"

    def _substituto(self, tipo: str, valor: str) -> tuple[Estrategia, str]:
        estrategia = self.estrategias.get(tipo, "remover")
        if estrategia == "manter":
            return estrategia, valor
        if estrategia == "pseudonimizar":
            return estrategia, self.pseudonimo(tipo, valor)
        return estrategia, f"[{ROTULOS.get(tipo, tipo.upper())}]"

    def _detectar(self, texto: str) -> list[Ocorrencia]:
        brutas: list[tuple[str, int, int, str]] = []

        for tipo, padrao in PADROES:
            for achado in padrao.finditer(texto):
                if tipo == "prontuario" and achado.groups():
                    inicio, fim = achado.span(1)
                else:
                    inicio, fim = achado.span()
                brutas.append((tipo, inicio, fim, texto[inicio:fim]))

        if self._padrao_nomes is not None:
            for achado in self._padrao_nomes.finditer(texto):
                valor = achado.group()
                tipo = self._nomes_conhecidos.get(valor, "nome_paciente")
                brutas.append((tipo, achado.start(), achado.end(), valor))

        brutas.extend(self._detectar_nomes_heuristicos(texto))

        brutas.sort(key=lambda item: (item[1], -(item[2] - item[1])))
        selecionadas: list[tuple[str, int, int, str]] = []
        ultimo_fim = -1
        for tipo, inicio, fim, valor in brutas:
            if inicio < ultimo_fim:
                continue
            selecionadas.append((tipo, inicio, fim, valor))
            ultimo_fim = fim

        ocorrencias = []
        for tipo, inicio, fim, valor in selecionadas:
            estrategia, substituto = self._substituto(tipo, valor)
            ocorrencias.append(Ocorrencia(tipo, valor, inicio, fim, estrategia, substituto))
        return ocorrencias

    def _detectar_nomes_heuristicos(self, texto: str) -> list[tuple[str, int, int, str]]:
        """Captura sequencias de nomes proprios apos marcadores como 'paciente' ou 'Sr.'."""
        padrao = re.compile(
            r"\b(?i:paciente|acompanhante|responsavel|respons[aá]vel|filho|filha|irm[aã]o|"
            r"irm[aã]|esposa|marido|m[aã]e|pai|vizinho|sr\.?|sra\.?)\s+"
            r"((?:[A-ZÀ-Ý][a-zà-ÿ']+(?:\s+(?:d[aeoi]s?|e)\s+|\s+)){0,4}[A-ZÀ-Ý][a-zà-ÿ']+)"
        )
        achados = []
        for achado in padrao.finditer(texto):
            valor = achado.group(1).strip()
            tokens = [t for t in valor.split() if len(t) > 2]
            if not tokens:
                continue
            if all(_normalizar(t) in NAO_SAO_NOMES for t in tokens):
                continue
            if len(tokens) == 1 and _normalizar(tokens[0]) in NAO_SAO_NOMES:
                continue
            achados.append(("nome_paciente", achado.start(1), achado.end(1), valor))
        return achados

    def anonimizar(self, texto: str) -> ResultadoAnonimizacao:
        """Retorna o texto anonimizado e a lista de identificadores tratados."""
        if not texto:
            return ResultadoAnonimizacao(texto="", ocorrencias=[])

        ocorrencias = self._detectar(texto)
        partes: list[str] = []
        cursor = 0
        for ocorrencia in ocorrencias:
            partes.append(texto[cursor:ocorrencia.inicio])
            partes.append(ocorrencia.substituto)
            cursor = ocorrencia.fim
        partes.append(texto[cursor:])
        return ResultadoAnonimizacao(texto="".join(partes), ocorrencias=ocorrencias)

    def anonimizar_texto(self, texto: str) -> str:
        """Atalho que devolve apenas o texto tratado."""
        return self.anonimizar(texto).texto

    def anonimizar_estrutura(self, dado, campos_sensiveis: set[str] | None = None):
        """Percorre dicionarios e listas anonimizando todos os textos encontrados.

        Campos listados em ``campos_sensiveis`` sao substituidos integralmente,
        mesmo que o valor nao case com nenhum padrao (por exemplo, o campo
        ``nome`` de um cadastro).
        """
        campos_sensiveis = campos_sensiveis or {
            "nome", "cpf", "cns", "rg", "telefone", "email", "endereco",
            "data_nascimento", "medico_responsavel", "autor",
        }
        if isinstance(dado, dict):
            eh_cadastro_de_pessoa = bool(
                {"cpf", "cns", "rg", "data_nascimento", "prontuario"} & set(dado)
            )
            resultado = {}
            for chave, valor in dado.items():
                if chave == "nome" and not eh_cadastro_de_pessoa:
                    resultado[chave] = self.anonimizar_estrutura(valor, campos_sensiveis)
                    continue
                if chave in campos_sensiveis and isinstance(valor, str) and valor:
                    tipo = {
                        "nome": "nome_paciente",
                        "medico_responsavel": "nome_profissional",
                        "autor": "nome_profissional",
                    }.get(chave)
                    if tipo:
                        resultado[chave] = self.pseudonimo(tipo, valor)
                    else:
                        resultado[chave] = self.anonimizar_texto(valor) if any(
                            padrao.search(valor) for _, padrao in PADROES
                        ) else f"[{ROTULOS.get(chave, chave.upper())}]"
                else:
                    resultado[chave] = self.anonimizar_estrutura(valor, campos_sensiveis)
            return resultado
        if isinstance(dado, list):
            return [self.anonimizar_estrutura(item, campos_sensiveis) for item in dado]
        if isinstance(dado, str):
            return self.anonimizar_texto(dado)
        return dado


def mascarar_para_log(texto: str, anonimizador: Anonimizador | None = None) -> str:
    """Mascara PII antes de gravar qualquer coisa em log de auditoria."""
    return (anonimizador or Anonimizador()).anonimizar_texto(texto)
