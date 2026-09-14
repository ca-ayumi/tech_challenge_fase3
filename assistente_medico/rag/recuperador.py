"""Recuperacao de trechos de protocolo (RAG lexical).

A base institucional e pequena (dezenas de documentos, uma centena de secoes) e
o vocabulario e altamente tecnico e padronizado. Nesse regime, BM25 sobre texto
normalizado recupera melhor que embeddings genericos e, sobretudo, e
explicavel: da para mostrar ao auditor exatamente quais termos casaram. Por
isso a recuperacao padrao e lexical, com dois reforcos:

- consulta que cita um identificador de documento (``PROT-SEP-001``) ou uma
  secao (``§4``) recebe boost para aquele documento;
- sinonimos institucionais expandem a consulta (``ATB`` -> ``antimicrobiano``).

``RecuperadorHibrido`` permite somar embeddings quando eles estiverem
disponiveis, sem tornar isso obrigatorio.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from ..config import RECUPERACAO
from ..dados.preprocessamento import Trecho, carregar_trechos

PALAVRAS_VAZIAS = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "sem", "sobre", "que", "e", "ou", "se", "ao",
    "aos", "qual", "quais", "quando", "como", "onde", "quanto", "quantos", "ser",
    "esta", "este", "essa", "esse", "isso", "ja", "mais", "menos", "muito", "meu",
    "minha", "seu", "sua", "nao", "sim", "the", "of", "posso", "devo", "preciso", "tem",
    "ha", "fazer", "faco", "eu", "voce", "me", "mim", "aqui", "hospital",
}

SINONIMOS = {
    "atb": ["antimicrobiano", "antibiotico"],
    "antibiotico": ["antimicrobiano"],
    "antibiotics": ["antimicrobiano"],
    "iam": ["infarto", "sindrome", "coronariana"],
    "sca": ["sindrome", "coronariana", "aguda"],
    "avc": ["acidente", "vascular", "cerebral"],
    "ecg": ["eletrocardiograma"],
    "tc": ["tomografia"],
    "uti": ["terapia", "intensiva"],
    "tev": ["tromboembolismo", "venoso", "profilaxia"],
    "cad": ["cetoacidose", "diabetica"],
    "pac": ["pneumonia", "adquirida", "comunidade"],
    "pa": ["pressao", "arterial"],
    "pam": ["pressao", "arterial", "media"],
    "fc": ["frequencia", "cardiaca"],
    "fr": ["frequencia", "respiratoria"],
    "itu": ["infeccao", "trato", "urinario"],
    "ccih": ["comissao", "controle", "infeccao"],
    "alta": ["alta", "hospitalar", "criterios"],
    "dose": ["dose", "posologia", "prescricao"],
    "receita": ["receituario", "prescricao"],
    "laudo": ["laudo", "exame", "imagem"],
    "ia": ["inteligencia", "artificial", "assistente"],
    "prescrever": ["prescricao", "proibidas", "assistente", "limites"],
    "prescricao": ["prescrever", "proibidas", "assistente"],
    "sozinho": ["proibidas", "validacao", "humana"],
    "pode": ["permitidas", "proibidas"],
    "trombolise": ["trombolitico", "janelas", "terapeuticas"],
}

PADRAO_DOCUMENTO = re.compile(r"\b((?:PROT|MOD|POP)-[A-Z]{2,10}-\d{3})\b", re.IGNORECASE)
PADRAO_SECAO = re.compile(r"§\s*(\d+)")


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.lower()


def tokenizar(texto: str, expandir: bool = False) -> list[str]:
    """Tokeniza removendo acentos, pontuacao e palavras vazias."""
    bruto = re.findall(r"[a-z0-9]+", normalizar(texto))
    tokens = [t for t in bruto if len(t) > 1 and t not in PALAVRAS_VAZIAS]
    if expandir:
        expandidos = list(tokens)
        for token in tokens:
            expandidos.extend(SINONIMOS.get(token, []))
        return expandidos
    return tokens


@dataclass
class TrechoRecuperado:
    """Trecho recuperado com o score e o motivo da recuperacao."""

    trecho: Trecho
    score: float
    motivo: str

    @property
    def referencia(self) -> str:
        return self.trecho.referencia

    def como_dicionario(self) -> dict:
        return {
            "referencia": self.trecho.referencia,
            "documento": self.trecho.documento,
            "titulo_secao": self.trecho.titulo_secao,
            "score": round(self.score, 3),
            "motivo": self.motivo,
            "versao": self.trecho.versao,
        }


class RecuperadorProtocolos:
    """Indice BM25 sobre as secoes dos documentos institucionais."""

    def __init__(self, trechos: Iterable[Trecho] | None = None) -> None:
        self.trechos: list[Trecho] = list(trechos) if trechos is not None else carregar_trechos()
        self._corpus = [
            tokenizar(f"{t.titulo_documento} {t.titulo_secao} {t.texto}") for t in self.trechos
        ]
        self._bm25 = self._construir_indice()

    def _construir_indice(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:  # pragma: no cover - fallback sem dependencia
            return None
        return BM25Okapi(self._corpus)

    def _scores_bm25(self, consulta: list[str]) -> list[float]:
        if self._bm25 is not None:
            return list(self._bm25.get_scores(consulta))
        conjunto = set(consulta)
        return [
            len(conjunto & set(documento)) / (len(conjunto) or 1) for documento in self._corpus
        ]

    def recuperar(self, consulta: str, top_k: int | None = None,
                  minimo_score: float | None = None,
                  documentos_preferidos: Iterable[str] = ()) -> list[TrechoRecuperado]:
        """Recupera as secoes mais relevantes para a consulta."""
        top_k = top_k or RECUPERACAO.top_k
        minimo_score = RECUPERACAO.minimo_score if minimo_score is None else minimo_score

        tokens = tokenizar(consulta, expandir=True)
        if not tokens:
            return []

        scores = self._scores_bm25(tokens)

        citados = {d.upper() for d in PADRAO_DOCUMENTO.findall(consulta)}
        preferidos = {d.upper() for d in documentos_preferidos}
        secoes_citadas = set(PADRAO_SECAO.findall(consulta))

        tokens_consulta = set(tokens)
        resultados: list[TrechoRecuperado] = []
        for indice, trecho in enumerate(self.trechos):
            score = float(scores[indice])
            motivos = ["bm25"]

            if "referencias internas" in normalizar(trecho.titulo_secao):
                score *= 0.2

            overlap = tokens_consulta & set(tokenizar(trecho.titulo_secao))
            if overlap:
                score += 2.5 * len(overlap)
                motivos.append(f"titulo da secao ({', '.join(sorted(overlap))})")

            if trecho.documento.upper() in citados:
                score += 8.0
                motivos.append("documento citado na pergunta")
                if trecho.secao in secoes_citadas:
                    score += 20.0
                    motivos.append("secao citada na pergunta")
            elif trecho.documento.upper() in preferidos:
                score *= 1.8
                motivos.append("protocolo ativo do paciente")

            if score > 0:
                resultados.append(TrechoRecuperado(trecho, score, " + ".join(motivos)))

        resultados.sort(key=lambda item: item.score, reverse=True)
        selecionados = [r for r in resultados if r.score >= minimo_score][:top_k]
        if not selecionados and resultados:
            melhor = resultados[0]
            selecionados = [TrechoRecuperado(melhor.trecho, melhor.score, melhor.motivo + " (abaixo do limiar)")]
        return selecionados

    def por_referencia(self, referencia: str) -> Trecho | None:
        for trecho in self.trechos:
            if trecho.referencia == referencia:
                return trecho
        return None

    def secoes_do_documento(self, documento: str) -> list[Trecho]:
        return [t for t in self.trechos if t.documento.upper() == documento.upper()]

    @staticmethod
    def formatar_contexto(recuperados: list[TrechoRecuperado], limite_caracteres: int = 4000) -> str:
        """Monta o bloco de contexto que vai para o prompt."""
        blocos: list[str] = []
        total = 0
        for item in recuperados:
            bloco = item.trecho.como_contexto()
            if total + len(bloco) > limite_caracteres:
                break
            blocos.append(bloco)
            total += len(bloco)
        return "\n\n".join(blocos)


@lru_cache(maxsize=1)
def recuperador_padrao() -> RecuperadorProtocolos:
    """Instancia unica reutilizada pelo fluxo e pelas ferramentas."""
    return RecuperadorProtocolos()
