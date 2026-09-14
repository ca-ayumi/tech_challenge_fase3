"""Politicas de uso do assistente, derivadas do PROT-GOV-010.

Cada politica e um objeto com: codigo, o que detecta, a secao do protocolo que a
justifica, a severidade e a acao a tomar. Manter isso como dado — e nao
espalhado em ``if`` pelo codigo — permite auditar a politica vigente, testar
cada regra isoladamente e mostrar ao usuario exatamente qual regra disparou.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern
from typing import Literal

Momento = Literal["entrada", "saida"]
Acao = Literal["bloquear", "encaminhar", "anexar_aviso", "mascarar", "registrar"]
Severidade = Literal["critica", "alta", "media", "baixa"]


@dataclass(frozen=True)
class Politica:
    """Uma regra de limite de atuacao."""

    codigo: str
    descricao: str
    momento: Momento
    acao: Acao
    severidade: Severidade
    fonte: str
    padroes: tuple[Pattern[str], ...] = field(default_factory=tuple)
    mensagem: str = ""

    def casa(self, texto: str) -> str | None:
        """Devolve o trecho que disparou a politica, ou None."""
        for padrao in self.padroes:
            achado = padrao.search(texto or "")
            if achado:
                return achado.group(0).strip()
        return None


def _p(*expressoes: str) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(expressao, re.IGNORECASE) for expressao in expressoes)


POLITICAS_ENTRADA: tuple[Politica, ...] = (
    Politica(
        codigo="ENT-01-prescricao",
        descricao="Solicitacao de prescricao, receita ou emissao de pedido em nome do profissional.",
        momento="entrada", acao="bloquear", severidade="critica", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\b(prescrev[ae]|prescrever|prescricao|prescrição)\b",
            r"\b(ger[ae]\w*|escrev\w+|faz|fa[cç]a|emit\w+|mont\w+|prepar\w+|redi[jg]\w+|"
            r"element\w*|elabor\w+)\s+(a\s+|o\s+|um\s+|uma\s+)?"
            r"(receita|receituario|receituário)\w*\b",
            r"\breceita\s+(de|para|do|da)\b",
            r"\bpedido\s+de\s+exame\s+(em\s+meu\s+nome|assinado)\b",
        ),
        mensagem=("Nao emito prescricao, receita ou pedido de exame em nome de um "
                  "profissional."),
    ),
    Politica(
        codigo="ENT-02-dose",
        descricao="Solicitacao de dose, posologia, via ou velocidade de infusao.",
        momento="entrada", acao="bloquear", severidade="critica", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\b(qual|quanto|quantos|quantas|me\s+pass[ae]|me\s+d[aá]|informa)\b[^?.!]{0,40}"
            r"\b(dose|doses|dosagem|posologia|mg|ml\s*/\s*h|gotejamento)\b",
            r"\b(dose|posologia|dosagem|velocidade\s+de\s+infus[aã]o|via\s+de\s+administra)\w*\b",
            r"\bquantos?\s+(mg|ml|comprimidos?|ampolas?|gotas?)\b",
        ),
        mensagem=("Nao informo dose, posologia, velocidade de infusao nem via de "
                  "administracao."),
    ),
    Politica(
        codigo="ENT-03-alta",
        descricao="Pedido de autorizacao de alta ou de suspensao de monitorizacao.",
        momento="entrada", acao="bloquear", severidade="alta", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\b(libera|liberar|autoriza|autorizar|pode\s+dar|dar|d[aá])\s+(a\s+)?alta\b",
            r"\balta\s+(hospitalar|da\s+uti|para\s+casa)\b[^?.!]{0,30}\?",
            r"\b(suspende|suspender|tira|retirar)\s+(a\s+)?monitoriza",
            r"\b(autoriza\w*|libera\w*|pode\s+(dar|mandar))\s+(a\s+)?"
            r"(sa[ií]da|transfer[eê]ncia|desospitaliza\w*)\s+(da|do)\s+"
            r"(uti|terapia\s+intensiva|unidade)",
        ),
        mensagem=("Nao autorizo alta hospitalar, alta de terapia intensiva nem suspensao "
                  "de monitorizacao."),
    ),
    Politica(
        codigo="ENT-04-diagnostico",
        descricao="Pedido de diagnostico definitivo.",
        momento="entrada", acao="bloquear", severidade="alta", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\b(confirma|confirmar|fecha|fechar)\s+(o\s+)?diagn[oó]stico\b",
            r"\bdiagn[oó]stico\s+definitivo\b",
            r"\b(o\s+)?paciente\s+tem\s+(c[aâ]ncer|tumor|aids|hiv)\b",
        ),
        mensagem="Nao afirmo diagnostico definitivo.",
    ),
    Politica(
        codigo="ENT-05-suspensao",
        descricao="Pedido de suspensao de medicacao em uso.",
        momento="entrada", acao="bloquear", severidade="alta", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\b(suspende|suspender|interrompe|interromper|para|parar|corta|cortar)\s+"
            r"(o|a|os|as)?\s*\w*\s*(medica|rem[eé]dio|antibi[oó]tico|anticoagulante)",
            r"\bposso\s+parar\s+de\s+tomar\b",
        ),
        mensagem=("Nao recomendo suspensao de medicacao em uso sem encaminhamento ao "
                  "medico assistente."),
    ),
    Politica(
        codigo="ENT-06-paciente",
        descricao="Solicitacao vinda de paciente ou familiar sobre conduta individual.",
        momento="entrada", acao="bloquear", severidade="alta", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\bsou\s+(paciente|familiar|filh[oa]|esposa|marido|m[aã]e|pai)\b",
            r"\bminha?\s+(m[aã]e|pai|filh[oa]|espos[ao])\s+(est[aá]|foi)\s+internad",
            r"\bestou\s+com\s+(dor|febre|falta\s+de\s+ar)\b[^?.!]{0,40}\bo\s+que\s+(eu\s+)?"
            r"(tomo|fa[cç]o)\b",
        ),
        mensagem=("Este assistente atende exclusivamente profissionais da instituicao e nao "
                  "responde a pacientes ou familiares sobre conduta individual."),
    ),
    Politica(
        codigo="ENT-08-sobreposicao",
        descricao="Tentativa de sobrepor as instrucoes do sistema ou de assumir outro papel.",
        momento="entrada", acao="bloquear", severidade="critica", fonte="PROT-GOV-010 §2",
        padroes=_p(
            r"\b(ignore|ignora|esque[cç]a|desconsidere|desative|desabilite)\s+"
            r"(suas?|as|essas?|todas?|qualquer)?\s*(regras?|restri[cç][oõ]es|instru[cç][oõ]es|"
            r"limites?|politicas?|pol[ií]ticas?)",
            r"\bvoc[eê]\s+(agora\s+)?(e|é|sera|ser[aá])\s+um[a]?\s+"
            r"\w+\s*(sem\s+restri[cç][oõ]es|sem\s+limites)",
            r"\b(finja|imagine|suponha)\s+que\s+voc[eê]\s+(e|é|pode)\b",
            r"\bmodo\s+(desenvolvedor|livre|sem\s+filtro)\b",
            r"\bpara\s+fins\s+(de\s+)?(estudo|educacionais?|academicos?|acad[eê]micos?|teste)"
            r"[^.?!]{0,40}\b(dose|posologia|receita|prescri)",
        ),
        mensagem=("Nao altero meus limites de atuacao a pedido. Eles sao definidos pela "
                  "politica institucional, nao pela conversa."),
    ),
    Politica(
        codigo="ENT-07-emergencia",
        descricao="Sinais de emergencia em curso que exigem acionamento imediato da equipe.",
        momento="entrada", acao="encaminhar", severidade="critica", fonte="PROT-GOV-010 §9",
        padroes=_p(
            r"\b(parada\s+cardiorrespirat[oó]ria|pcr\s+em\s+curso|n[aã]o\s+responde)\b",
            r"\b(rebaixamento|rebaixou)\b[^.!?]{0,30}\b(agora|s[uú]bito|subita)",
            r"\bdor\s+tor[aá]cica\b[^.!?]{0,40}\b(agora|em\s+curso|intensa)",
            r"\b(sangramento\s+ativo|hemorragia\s+ativa)\b",
            r"\b(convuls[aã]o|convulsionando)\s+(agora|em\s+curso)\b",
            r"\b(dessatura|dessaturando|cianose)\b",
            r"\b(risco\s+de\s+autoexterm[ií]nio|tentativa\s+de\s+suic[ií]dio)\b",
        ),
        mensagem=("Situacao de emergencia: acione a equipe assistencial imediatamente, sem "
                  "aguardar esta resposta."),
    ),
)

_UNIDADES_DOSE = r"(?:mg|mcg|µg|ug|ui|u\.i\.)"

POLITICAS_SAIDA: tuple[Politica, ...] = (
    Politica(
        codigo="SAI-01-dose",
        descricao="Dose ou posologia de medicamento presente na resposta.",
        momento="saida", acao="bloquear", severidade="critica", fonte="PROT-GOV-010 §3",
        padroes=_p(
            rf"\b\d+(?:[.,]\d+)?\s*{_UNIDADES_DOSE}\b(?!\s*/\s*(?:d?[lL]|mm))",
            r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg)\s*/\s*kg\b",
            r"\b\d+(?:[.,]\d+)?\s*(?:mcg|µg)\s*/\s*kg\s*/\s*min\b",
            r"\b\d+\s*(?:comprimidos?|c[aá]psulas?|ampolas?|gotas?|frascos?)\b",
            r"\bde\s+\d+\s+em\s+\d+\s+horas\b",
            r"\b\d+\s*x\s*(?:ao|por)\s+dia\b",
        ),
        mensagem="A resposta continha dose ou posologia e foi bloqueada.",
    ),
    Politica(
        codigo="SAI-02-prescricao",
        descricao="Linguagem prescritiva em primeira pessoa.",
        momento="saida", acao="bloquear", severidade="critica", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\b(prescrevo|receito|estou\s+prescrevendo|segue\s+a\s+receita)\b",
            r"\breceitu[aá]rio\s*:\s*\n",
        ),
        mensagem="A resposta assumia ato prescritivo e foi bloqueada.",
    ),
    Politica(
        codigo="SAI-03-diagnostico",
        descricao="Afirmacao de diagnostico definitivo.",
        momento="saida", acao="anexar_aviso", severidade="alta", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\bdiagn[oó]stico\s+(?:e|é)\s+definitiv",
            r"\bconfirmo\s+(?:o\s+)?diagn[oó]stico\b",
            r"\bcom\s+certeza\s+(?:e|é)\s+(?:um|uma)\b",
        ),
        mensagem=("A resposta continha afirmacao diagnostica; foi anexado aviso de que a "
                  "conclusao cabe ao profissional."),
    ),
    Politica(
        codigo="SAI-04-autorizacao",
        descricao="Autorizacao de alta ou de suspensao emitida pelo assistente.",
        momento="saida", acao="bloquear", severidade="alta", fonte="PROT-GOV-010 §3",
        padroes=_p(
            r"\b(autorizo|libero|pode\s+receber)\s+(a\s+)?alta\b",
            r"\bestá\s+liberado\s+para\s+alta\b",
        ),
        mensagem="A resposta autorizava alta e foi bloqueada.",
    ),
)

POLITICAS = POLITICAS_ENTRADA + POLITICAS_SAIDA


def politica_por_codigo(codigo: str) -> Politica | None:
    for politica in POLITICAS:
        if politica.codigo == codigo:
            return politica
    return None


def resumo_politicas() -> list[dict[str, str]]:
    """Tabela das politicas vigentes, usada na interface e no relatorio."""
    return [
        {
            "codigo": politica.codigo,
            "momento": politica.momento,
            "acao": politica.acao,
            "severidade": politica.severidade,
            "fonte": politica.fonte,
            "descricao": politica.descricao,
        }
        for politica in POLITICAS
    ]
