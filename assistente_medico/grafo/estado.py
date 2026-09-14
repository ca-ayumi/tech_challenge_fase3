"""Estado compartilhado pelos nos do fluxo.

O estado e um dicionario tipado que atravessa o grafo inteiro. Cada no le o que
precisa e escreve apenas os campos que produz — o LangGraph funde as escritas.
Manter tudo em um estado unico e explicito e o que permite auditar a decisao
depois: a trilha de auditoria e, na pratica, a serializacao deste estado em
cada etapa.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _substituir(_antigo: Any, novo: Any) -> Any:
    """Redutor padrao: o valor mais recente vence."""
    return novo


def _concatenar(antigo: list | None, novo: list | None) -> list:
    """Redutor de listas acumulativas (alertas, violacoes, eventos)."""
    return list(antigo or []) + list(novo or [])


class EstadoAssistente(TypedDict, total=False):
    """Tudo que o fluxo conhece sobre uma interacao."""

    interacao: str
    pergunta: str
    perfil: str
    usuario: str
    prontuario_informado: str | None

    categoria: str
    permitido: bool
    intencao: str

    prontuario: str | None
    contexto_paciente: str
    campos_prontuario: list[str]
    exames_pendentes: list[dict[str, Any]]

    contexto_documentos: str
    recuperados: list[dict[str, Any]]

    resposta_bruta: str
    resposta: str
    metadados_modelo: dict[str, Any]

    violacoes: Annotated[list[dict[str, Any]], _concatenar]
    avisos: Annotated[list[str], _concatenar]
    alertas: Annotated[list[dict[str, Any]], _concatenar]
    alertas_registrados: list[int]
    validacao_id: int | None
    explicacao: dict[str, Any]

    etapas: Annotated[list[str], _concatenar]
    erro: str | None


def estado_inicial(pergunta: str, interacao: str, perfil: str = "medico",
                   usuario: str = "nao identificado",
                   prontuario: str | None = None) -> EstadoAssistente:
    return EstadoAssistente(
        interacao=interacao,
        pergunta=pergunta,
        perfil=perfil,
        usuario=usuario,
        prontuario_informado=prontuario,
        prontuario=None,
        categoria="",
        permitido=True,
        intencao="",
        contexto_paciente="",
        campos_prontuario=[],
        exames_pendentes=[],
        contexto_documentos="",
        recuperados=[],
        resposta_bruta="",
        resposta="",
        metadados_modelo={},
        violacoes=[],
        avisos=[],
        alertas=[],
        alertas_registrados=[],
        validacao_id=None,
        explicacao={},
        etapas=[],
        erro=None,
    )
