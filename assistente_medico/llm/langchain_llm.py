"""Integracao da LLM customizada com o LangChain.

``ChatAssistenteMedico`` encapsula o modelo ajustado (MLX, Transformers ou o
backend deterministico) em um ``BaseChatModel``. A partir dai, o modelo treinado
neste projeto pode ser usado em qualquer construcao do LangChain — LCEL, agentes,
grafos do LangGraph — sem que o restante do codigo saiba qual backend esta por
tras.

O ultimo resultado bruto fica acessivel em ``ultima_resposta``, porque o fluxo
precisa dos metadados (backend, adaptador, tokens, duracao) para a auditoria, e
eles nao cabem no contrato de mensagens do LangChain.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field

from ..config import MODELO
from .modelo_local import RespostaModelo, carregar_modelo

PAPEL_POR_TIPO = {"system": "system", "human": "user", "ai": "assistant"}


def mensagens_para_dicionarios(mensagens: Sequence[BaseMessage]) -> list[dict[str, str]]:
    """Converte mensagens do LangChain para o formato de chat dos backends."""
    convertidas: list[dict[str, str]] = []
    for mensagem in mensagens:
        papel = PAPEL_POR_TIPO.get(mensagem.type, "user")
        conteudo = mensagem.content
        if isinstance(conteudo, list):
            conteudo = " ".join(str(parte) for parte in conteudo)
        convertidas.append({"role": papel, "content": str(conteudo)})
    return convertidas


class ChatAssistenteMedico(BaseChatModel):
    """Modelo de chat do LangChain servido pela LLM customizada do hospital."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: str = Field(default_factory=lambda: MODELO.backend)
    temperatura: float = Field(default_factory=lambda: MODELO.temperatura)
    max_tokens: int = Field(default_factory=lambda: MODELO.max_tokens)
    top_p: float = Field(default_factory=lambda: MODELO.top_p)
    usar_adaptador: bool = Field(default_factory=lambda: MODELO.usar_adaptador)
    modelo: Any = None
    ultima_resposta: Any = None

    def model_post_init(self, __contexto: Any) -> None:
        if self.modelo is None:
            self.modelo = carregar_modelo(
                backend=self.backend, usar_adaptador=self.usar_adaptador
            )
        self.backend = getattr(self.modelo, "backend", self.backend)

    @property
    def _llm_type(self) -> str:
        return f"assistente-medico-{self.backend}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "modelo": getattr(self.modelo, "nome", "desconhecido"),
            "adaptador": getattr(self.modelo, "adaptador", None),
            "temperatura": self.temperatura,
            "max_tokens": self.max_tokens,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        resposta: RespostaModelo = self.modelo.gerar(
            mensagens_para_dicionarios(messages),
            temperatura=kwargs.get("temperatura", self.temperatura),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
        )
        self.ultima_resposta = resposta

        texto = resposta.texto
        for parada in stop or []:
            if parada in texto:
                texto = texto.split(parada)[0]

        mensagem = AIMessage(
            content=texto,
            response_metadata={
                "backend": resposta.backend,
                "modelo": resposta.modelo,
                "adaptador": resposta.adaptador,
                "tokens_gerados": resposta.tokens_gerados,
                "duracao_ms": resposta.duracao_ms,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=mensagem)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming simplificado: gera tudo e devolve em blocos.

        Os backends locais ja sao rapidos o suficiente para a demonstracao; um
        streaming token a token exigiria expor o gerador do MLX, o que
        complicaria a auditoria sem ganho pratico aqui.
        """
        resultado = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        texto = resultado.generations[0].message.content
        for inicio in range(0, len(texto), 120):
            yield ChatGenerationChunk(message=AIMessageChunk(content=texto[inicio:inicio + 120]))


def criar_modelo_chat(backend: str | None = None, **parametros: Any) -> ChatAssistenteMedico:
    """Instancia o modelo de chat do LangChain com o backend pedido."""
    return ChatAssistenteMedico(backend=backend or MODELO.backend, **parametros)
