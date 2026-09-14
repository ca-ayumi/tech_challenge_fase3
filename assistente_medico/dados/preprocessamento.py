"""Leitura e fatiamento dos documentos institucionais.

Cada protocolo e cada modelo de documento e quebrado em trechos que
correspondem as secoes numeradas do arquivo. O identificador do trecho
(``PROT-SEP-001 §4``) e a unidade de citacao usada em todo o sistema: e o que
aparece no bloco "Fontes" das respostas e no registro de auditoria.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import CAMINHOS

PADRAO_SECAO = re.compile(r"^##\s+(?:(\d+)\.\s*)?(.+)$", re.MULTILINE)


def normalizar_espacos(texto: str) -> str:
    """Colapsa espacos e normaliza a forma unicode, preservando paragrafos."""
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace(" ", " ")
    linhas = [re.sub(r"[ \t]+", " ", linha).rstrip() for linha in texto.splitlines()]
    texto = "\n".join(linhas)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def remover_marcacao(texto: str) -> str:
    """Remove ruido de markdown que nao agrega para recuperacao nem para treino."""
    texto = re.sub(r"^>\s?", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"\*\*(.+?)\*\*", r"\1", texto)
    texto = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", texto)
    return normalizar_espacos(texto)


@dataclass
class Documento:
    """Documento institucional completo, com seus metadados de cabecalho."""

    identificador: str
    titulo: str
    caminho: str
    tipo: str
    metadados: dict[str, str] = field(default_factory=dict)
    corpo: str = ""

    @property
    def versao(self) -> str:
        return self.metadados.get("versao", "").strip('"')

    @property
    def especialidade(self) -> str:
        return self.metadados.get("especialidade", self.metadados.get("tipo", ""))


@dataclass
class Trecho:
    """Uma secao citavel de um documento."""

    referencia: str
    documento: str
    titulo_documento: str
    secao: str
    titulo_secao: str
    texto: str
    tipo: str
    versao: str

    def como_contexto(self) -> str:
        return f"[{self.referencia}] {self.titulo_documento} — {self.titulo_secao}\n{self.texto}"


def _extrair_frontmatter(conteudo: str) -> tuple[dict[str, str], str]:
    """Le o cabecalho YAML simples (chave: valor) do inicio do arquivo."""
    if not conteudo.startswith("---"):
        return {}, conteudo
    fim = conteudo.find("\n---", 3)
    if fim == -1:
        return {}, conteudo
    cabecalho = conteudo[3:fim]
    corpo = conteudo[fim + 4:]
    metadados: dict[str, str] = {}
    for linha in cabecalho.splitlines():
        if ":" not in linha or linha.strip().startswith("#"):
            continue
        chave, _, valor = linha.partition(":")
        metadados[chave.strip()] = valor.strip().strip('"')
    return metadados, corpo


def carregar_documento(caminho: Path, tipo: str) -> Documento:
    conteudo = caminho.read_text(encoding="utf-8")
    metadados, corpo = _extrair_frontmatter(conteudo)
    identificador = metadados.get("id") or caminho.stem.split("_")[0]
    titulo = metadados.get("titulo") or identificador
    return Documento(
        identificador=identificador,
        titulo=titulo,
        caminho=str(caminho),
        tipo=metadados.get("tipo", tipo),
        metadados=metadados,
        corpo=normalizar_espacos(corpo),
    )


def carregar_documentos(
    dir_protocolos: Path | None = None,
    dir_modelos: Path | None = None,
) -> list[Documento]:
    """Carrega protocolos e modelos de documento do diretorio de dados brutos."""
    dir_protocolos = dir_protocolos or CAMINHOS.protocolos
    dir_modelos = dir_modelos or CAMINHOS.modelos_documentos

    documentos: list[Documento] = []
    for caminho in sorted(dir_protocolos.glob("*.md")):
        documentos.append(carregar_documento(caminho, tipo="protocolo"))
    if dir_modelos.exists():
        for caminho in sorted(dir_modelos.glob("*.md")):
            documentos.append(carregar_documento(caminho, tipo="modelo_documento"))
    return documentos


def dividir_em_trechos(documento: Documento) -> list[Trecho]:
    """Quebra o corpo do documento nas secoes marcadas por ``## n. Titulo``."""
    corpo = documento.corpo
    marcadores = list(PADRAO_SECAO.finditer(corpo))
    trechos: list[Trecho] = []

    for indice, marcador in enumerate(marcadores):
        numero = marcador.group(1)
        titulo_secao = marcador.group(2).strip()
        inicio = marcador.end()
        fim = marcadores[indice + 1].start() if indice + 1 < len(marcadores) else len(corpo)
        texto = remover_marcacao(corpo[inicio:fim])
        if not texto or len(texto) < 40:
            continue

        secao = numero if numero else str(indice + 1)
        referencia = f"{documento.identificador} §{secao}"
        trechos.append(
            Trecho(
                referencia=referencia,
                documento=documento.identificador,
                titulo_documento=documento.titulo,
                secao=secao,
                titulo_secao=titulo_secao,
                texto=texto,
                tipo=documento.tipo,
                versao=documento.versao,
            )
        )
    return trechos


def construir_trechos(documentos: list[Documento] | None = None) -> list[Trecho]:
    documentos = documentos if documentos is not None else carregar_documentos()
    trechos: list[Trecho] = []
    for documento in documentos:
        trechos.extend(dividir_em_trechos(documento))
    return trechos


def salvar_trechos(trechos: list[Trecho], destino: Path | None = None) -> Path:
    destino = destino or CAMINHOS.dados_processados / "trechos_documentos.jsonl"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as arquivo:
        for trecho in trechos:
            arquivo.write(json.dumps(asdict(trecho), ensure_ascii=False) + "\n")
    return destino


def carregar_trechos(origem: Path | None = None) -> list[Trecho]:
    origem = origem or CAMINHOS.dados_processados / "trechos_documentos.jsonl"
    if not origem.exists():
        trechos = construir_trechos()
        salvar_trechos(trechos, origem)
        return trechos
    with origem.open(encoding="utf-8") as arquivo:
        return [Trecho(**json.loads(linha)) for linha in arquivo if linha.strip()]


def indexar_por_referencia(trechos: list[Trecho]) -> dict[str, Trecho]:
    return {trecho.referencia: trecho for trecho in trechos}
