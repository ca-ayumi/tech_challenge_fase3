"""Cadeias LangChain (LCEL) do assistente.

O grafo do LangGraph e o caminho completo, com governanca e efeitos colaterais.
Este modulo oferece a composicao declarativa equivalente em LCEL, util para
tres coisas: compor o assistente dentro de outras cadeias, avaliar o modelo sem
o overhead do grafo, e deixar visivel a estrutura do pipeline —
recuperacao -> contexto -> prompt -> LLM customizada -> validacao.

    cadeia = criar_cadeia_consulta()
    resultado = cadeia.invoke({"pergunta": "...", "prontuario": "884521"})
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

from .dados.banco import BancoHospital, preparar_banco
from .dados.contexto_paciente import montar_contexto
from .explicabilidade import montar_explicacao
from .finetuning.formato import PROMPT_SISTEMA
from .llm.langchain_llm import ChatAssistenteMedico, criar_modelo_chat
from .rag.recuperador import RecuperadorProtocolos, TrechoRecuperado, recuperador_padrao
from .seguranca.guardrails import Guardrails

MODELO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{prompt_sistema}"),
    ("human",
     "{pergunta}\n\n"
     "### Contexto do paciente\n{contexto_paciente}\n\n"
     "### Contexto institucional recuperado\n{contexto_documentos}"),
])


def _recuperar(recuperador: RecuperadorProtocolos):
    def executar(entrada: dict[str, Any]) -> list[TrechoRecuperado]:
        return recuperador.recuperar(entrada.get("pergunta", ""))
    return RunnableLambda(executar)


def _contexto_do_paciente(banco: BancoHospital):
    def executar(entrada: dict[str, Any]) -> str:
        prontuario = entrada.get("prontuario")
        if not prontuario:
            return "Nenhum paciente informado nesta consulta."
        contexto = montar_contexto(banco, str(prontuario))
        return contexto.como_texto() if contexto else "Prontuario nao encontrado."
    return RunnableLambda(executar)


def criar_cadeia_recuperacao(
    recuperador: RecuperadorProtocolos | None = None,
    banco: BancoHospital | None = None,
) -> Runnable:
    """Cadeia que monta os contextos (documentos e paciente) em paralelo."""
    recuperador = recuperador or recuperador_padrao()
    banco = banco or preparar_banco()

    return RunnableParallel(
        pergunta=RunnableLambda(lambda e: e.get("pergunta", "")),
        prontuario=RunnableLambda(lambda e: e.get("prontuario")),
        recuperados=_recuperar(recuperador),
        contexto_paciente=_contexto_do_paciente(banco),
        prompt_sistema=RunnableLambda(lambda _: PROMPT_SISTEMA),
    ) | RunnableLambda(
        lambda e: {
            **e,
            "contexto_documentos": RecuperadorProtocolos.formatar_contexto(e["recuperados"])
            or "Nenhum trecho institucional recuperado.",
        }
    )


def criar_cadeia_consulta(
    modelo: ChatAssistenteMedico | None = None,
    recuperador: RecuperadorProtocolos | None = None,
    banco: BancoHospital | None = None,
    guardrails: Guardrails | None = None,
    com_explicacao: bool = True,
) -> Runnable:
    """Monta a cadeia completa: recuperacao -> prompt -> LLM customizada -> guardrail.

    A saida e um dicionario com ``resposta``, ``fontes``, ``violacoes`` e, quando
    ``com_explicacao``, o rastro de explicabilidade.
    """
    modelo = modelo or criar_modelo_chat()
    recuperador = recuperador or recuperador_padrao()
    banco = banco or preparar_banco()
    guardrails = guardrails or Guardrails()

    preparo = criar_cadeia_recuperacao(recuperador, banco)
    geracao = MODELO_PROMPT | modelo | StrOutputParser()

    def finalizar(entrada: dict[str, Any]) -> dict[str, Any]:
        recuperados: list[TrechoRecuperado] = entrada["recuperados"]
        bruta: str = entrada["resposta_bruta"]
        resultado = guardrails.avaliar_saida(
            bruta, fontes_esperadas=[r.referencia for r in recuperados][:3]
        )
        saida: dict[str, Any] = {
            "pergunta": entrada["pergunta"],
            "resposta": resultado.texto,
            "resposta_bruta": bruta,
            "fontes": [r.como_dicionario() for r in recuperados],
            "violacoes": [v.como_dicionario() for v in resultado.violacoes],
            "avisos": resultado.avisos,
            "aprovado": resultado.aprovado,
        }
        if com_explicacao:
            saida["explicacao"] = montar_explicacao(
                resultado.texto, recuperados=recuperados
            ).como_dicionario()
        return saida

    return (
        preparo
        | RunnableLambda(lambda e: {**e, "resposta_bruta": geracao.invoke(e)})
        | RunnableLambda(finalizar)
    )


def criar_cadeia_simples(modelo: ChatAssistenteMedico | None = None) -> Runnable:
    """Cadeia minima (prompt -> LLM -> texto), usada na avaliacao do modelo base.

    Sem recuperacao e sem guardrail: e a linha de base contra a qual se mede o
    ganho do fine-tuning e do restante do pipeline.
    """
    modelo = modelo or criar_modelo_chat()
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_SISTEMA),
        ("human", "{pergunta}"),
    ])
    return prompt | modelo | StrOutputParser()
