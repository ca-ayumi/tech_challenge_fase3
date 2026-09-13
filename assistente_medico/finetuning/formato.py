"""Formato canonico das mensagens do assistente.

Este modulo e a unica fonte de verdade do formato: o mesmo prompt de sistema e
a mesma estrutura de resposta sao usados na construcao do dataset de treino, na
inferencia e na avaliacao. Divergir aqui e a causa mais comum de um modelo
ajustado que "esquece" o formato em producao.

Estrutura de resposta:

    **Resposta**
    <corpo objetivo>

    **Fontes**
    - PROT-SEP-001 §4 - Pacote da primeira hora

    **Validacao**
    <rotulo de validacao humana>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PROMPT_SISTEMA = (
    "Voce e o Assistente Clinico Institucional de um hospital, um sistema de apoio "
    "a decisao para profissionais de saude. Responde em portugues do Brasil, de "
    "forma objetiva e tecnica.\n\n"
    "Regras invioláveis (PROT-GOV-010):\n"
    "1. Nunca emita prescricao, receita ou pedido de exame em nome de um profissional.\n"
    "2. Nunca informe dose, posologia, velocidade de infusao ou via de administracao.\n"
    "3. Nunca afirme diagnostico definitivo nem autorize alta ou suspensao de monitorizacao.\n"
    "4. Nunca responda a pacientes ou familiares sobre conduta individual.\n"
    "5. Toda sugestao de conduta e rotulada como sugestao e depende de validacao humana.\n"
    "6. Fundamente as afirmacoes nos protocolos institucionais e nos registros do "
    "prontuario fornecidos no contexto, citando a referencia usada.\n"
    "7. Diante de sinais de emergencia, oriente acionar a equipe imediatamente.\n\n"
    "Responda sempre na estrutura: bloco **Resposta**, bloco **Fontes** com as "
    "referencias utilizadas, e bloco **Validacao**."
)

ROTULO_VALIDACAO = (
    "Sugestao de apoio a decisao. Requer validacao do medico assistente antes de "
    "qualquer conduta (PROT-GOV-010 §5)."
)
ROTULO_INFORMATIVO = (
    "Conteudo informativo extraido de protocolo institucional. A decisao clinica "
    "permanece com o profissional responsavel (PROT-GOV-010 §5)."
)
ROTULO_RECUSA = (
    "Solicitacao fora dos limites de atuacao do assistente (PROT-GOV-010 §3). "
    "Nenhuma conduta foi gerada."
)
ROTULO_EMERGENCIA = (
    "Situacao de emergencia: acione a equipe assistencial imediatamente. Esta "
    "resposta nao substitui avaliacao presencial (PROT-GOV-010 §9)."
)

TITULOS_SECAO = {"resposta": "**Resposta**", "fontes": "**Fontes**", "validacao": "**Validacao**"}


@dataclass
class RespostaEstruturada:
    """Resposta do assistente decomposta em seus tres blocos."""

    corpo: str
    fontes: list[str] = field(default_factory=list)
    validacao: str = ROTULO_INFORMATIVO

    def texto(self) -> str:
        return montar_resposta(self.corpo, self.fontes, self.validacao)


def montar_resposta(corpo: str, fontes: list[str] | None = None,
                    validacao: str = ROTULO_INFORMATIVO) -> str:
    """Monta o texto final da resposta no formato canonico."""
    linhas = [TITULOS_SECAO["resposta"], corpo.strip(), "", TITULOS_SECAO["fontes"]]
    if fontes:
        linhas.extend(f"- {fonte}" for fonte in fontes)
    else:
        linhas.append("- Nenhuma fonte institucional recuperada para esta resposta.")
    linhas.extend(["", TITULOS_SECAO["validacao"], validacao.strip()])
    return "\n".join(linhas)


def montar_pergunta(pergunta: str, contexto_documentos: str = "",
                    contexto_paciente: str = "") -> str:
    """Monta o turno do usuario, com os contextos recuperados quando existirem."""
    partes = [pergunta.strip()]
    if contexto_paciente.strip():
        partes.append("### Contexto do paciente\n" + contexto_paciente.strip())
    if contexto_documentos.strip():
        partes.append("### Contexto institucional recuperado\n" + contexto_documentos.strip())
    return "\n\n".join(partes)


def montar_mensagens(pergunta: str, resposta: str | None = None,
                     contexto_documentos: str = "", contexto_paciente: str = "",
                     prompt_sistema: str = PROMPT_SISTEMA) -> list[dict[str, str]]:
    """Monta a lista de mensagens no formato de chat usado no treino e na inferencia."""
    mensagens = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": montar_pergunta(pergunta, contexto_documentos, contexto_paciente)},
    ]
    if resposta is not None:
        mensagens.append({"role": "assistant", "content": resposta})
    return mensagens


_PADRAO_FONTE = re.compile(r"^[-*]\s*(.+)$", re.MULTILINE)


def separar_secoes(texto: str) -> RespostaEstruturada:
    """Decompoe uma resposta gerada nos tres blocos, para avaliacao e auditoria."""
    if not texto:
        return RespostaEstruturada(corpo="", fontes=[], validacao="")

    normalizado = texto.replace("**Validação**", "**Validacao**")
    indice_fontes = normalizado.find(TITULOS_SECAO["fontes"])
    indice_validacao = normalizado.find(TITULOS_SECAO["validacao"])

    inicio_corpo = normalizado.find(TITULOS_SECAO["resposta"])
    if inicio_corpo >= 0:
        inicio_corpo += len(TITULOS_SECAO["resposta"])
    else:
        inicio_corpo = 0

    fim_corpo = indice_fontes if indice_fontes > 0 else (
        indice_validacao if indice_validacao > 0 else len(normalizado))
    corpo = normalizado[inicio_corpo:fim_corpo].strip()

    fontes: list[str] = []
    if indice_fontes >= 0:
        fim_fontes = indice_validacao if indice_validacao > indice_fontes else len(normalizado)
        bloco = normalizado[indice_fontes + len(TITULOS_SECAO["fontes"]):fim_fontes]
        fontes = [achado.group(1).strip() for achado in _PADRAO_FONTE.finditer(bloco)]
        fontes = [f for f in fontes if f and not f.lower().startswith("nenhuma fonte")]

    validacao = ""
    if indice_validacao >= 0:
        validacao = normalizado[indice_validacao + len(TITULOS_SECAO["validacao"]):].strip()

    return RespostaEstruturada(corpo=corpo, fontes=fontes, validacao=validacao)


def segue_formato(texto: str) -> bool:
    """Indica se a saida respeita a estrutura de tres blocos exigida."""
    normalizado = (texto or "").replace("**Validação**", "**Validacao**")
    return all(titulo in normalizado for titulo in TITULOS_SECAO.values())


def referencias_citadas(texto: str) -> list[str]:
    """Extrai os identificadores de documento citados em qualquer parte do texto."""
    padrao = re.compile(r"\b((?:PROT|MOD|POP)-[A-Z]{2,10}-\d{3})(?:\s*§\s*(\d+))?")
    referencias = []
    for achado in padrao.finditer(texto or ""):
        referencia = achado.group(1)
        if achado.group(2):
            referencia = f"{referencia} §{achado.group(2)}"
        if referencia not in referencias:
            referencias.append(referencia)
    return referencias
