"""Montagem do fluxo de decisao no LangGraph.

Topologia:

    START
      v
    triagem ------------------------------> recusar --> concluir --> END
      | (permitido)                (bloqueada na entrada)
      +-- com paciente --> contexto_paciente --> exames_pendentes --> avaliar_regras --+
      |                                                                                |
      +-- sem paciente ----------------------------------------------------------------+
                                                                                       v
                                                                          recuperar_protocolos
                                                                                       v
                                                                             gerar_resposta
                                                                                       v
                                                                              validar_saida
                                                                                       v
                                                                              emitir_alertas
                                                                                       v
                                                                                  explicar
                                                                                       v
                                                                             abrir_validacao
                                                                                       v
                                                                                  concluir --> END

A bifurcacao depois da triagem e o ponto central: uma pergunta que envolve um
paciente aciona a leitura do prontuario, a verificacao de exames pendentes e as
regras de vigilancia antes de o modelo ser chamado; uma duvida geral de
protocolo vai direto para a recuperacao. Em ambos os casos, guardrail de saida,
explicabilidade, validacao humana e auditoria sao obrigatorios.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..auditoria import Auditoria
from .estado import EstadoAssistente, estado_inicial
from .nos import Dependencias, criar_nos


def rota_apos_triagem(estado: EstadoAssistente) -> str:
    """Decide o caminho depois da triagem."""
    if not estado.get("permitido", True):
        return "recusar"
    if estado.get("prontuario"):
        return "carregar_contexto_paciente"
    return "recuperar_protocolos"


def construir_grafo(dependencias: Dependencias | None = None, compilar: bool = True):
    """Monta (e opcionalmente compila) o grafo do assistente."""
    dependencias = dependencias or Dependencias()
    nos = criar_nos(dependencias)

    grafo = StateGraph(EstadoAssistente)
    for nome, funcao in nos.items():
        grafo.add_node(nome, funcao)

    grafo.add_edge(START, "triagem")
    grafo.add_conditional_edges(
        "triagem",
        rota_apos_triagem,
        {
            "recusar": "recusar",
            "carregar_contexto_paciente": "carregar_contexto_paciente",
            "recuperar_protocolos": "recuperar_protocolos",
        },
    )

    grafo.add_edge("carregar_contexto_paciente", "verificar_exames_pendentes")
    grafo.add_edge("verificar_exames_pendentes", "avaliar_regras")
    grafo.add_edge("avaliar_regras", "recuperar_protocolos")

    grafo.add_edge("recuperar_protocolos", "gerar_resposta")
    grafo.add_edge("gerar_resposta", "validar_saida")
    grafo.add_edge("validar_saida", "emitir_alertas")
    grafo.add_edge("emitir_alertas", "explicar")
    grafo.add_edge("explicar", "abrir_validacao")
    grafo.add_edge("abrir_validacao", "concluir")

    grafo.add_edge("recusar", "concluir")
    grafo.add_edge("concluir", END)

    return grafo.compile() if compilar else grafo


class AssistenteMedico:
    """Fachada do assistente: recebe a pergunta, devolve a resposta auditada."""

    def __init__(self, dependencias: Dependencias | None = None) -> None:
        self.dependencias = dependencias or Dependencias()
        self.grafo = construir_grafo(self.dependencias)
        self.auditoria: Auditoria = self.dependencias.auditoria  # type: ignore[assignment]

    def responder(self, pergunta: str, perfil: str = "medico",
                  usuario: str = "nao identificado",
                  prontuario: str | None = None,
                  interacao: str | None = None) -> dict[str, Any]:
        """Executa o fluxo completo para uma pergunta."""
        identificador = interacao or Auditoria.nova_interacao()
        estado = estado_inicial(
            pergunta=pergunta, interacao=identificador, perfil=perfil,
            usuario=usuario, prontuario=prontuario,
        )
        final = self.grafo.invoke(estado)
        return dict(final)

    def transmitir(self, pergunta: str, **parametros: Any):
        """Executa o fluxo emitindo o estado a cada no — usado para acompanhar a execucao."""
        identificador = parametros.pop("interacao", None) or Auditoria.nova_interacao()
        estado = estado_inicial(
            pergunta=pergunta, interacao=identificador,
            perfil=parametros.get("perfil", "medico"),
            usuario=parametros.get("usuario", "nao identificado"),
            prontuario=parametros.get("prontuario"),
        )
        return self.grafo.stream(estado)

    def trilha(self, interacao: str) -> list[dict[str, Any]]:
        return self.auditoria.trilha(interacao)

    def diagrama_mermaid(self) -> str:
        """Diagrama do fluxo em Mermaid, para o relatorio e o README."""
        try:
            return self.grafo.get_graph().draw_mermaid()
        except Exception:  # pragma: no cover - depende de versao do langgraph
            return ""
