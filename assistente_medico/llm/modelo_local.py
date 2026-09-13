"""Carregamento e execucao do modelo de linguagem customizado.

Tres backends implementam a mesma interface ``ModeloLinguagem``:

- ``ModeloMLX``: modelo quantizado rodando em Apple Silicon via MLX, com o
  adaptador LoRA treinado neste projeto. E o backend usado na demonstracao.
- ``ModeloTransformers``: mesmo modelo em PyTorch com PEFT, para ambientes com
  GPU NVIDIA (Colab) ou CPU.
- ``ModeloEco``: nao e um modelo neural. Monta a resposta a partir do contexto
  recuperado, por regra. Serve para rodar o fluxo completo (LangGraph,
  guardrails, auditoria) em teste automatizado e em maquina sem modelo baixado.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..config import CAMINHOS, MODELO
from ..finetuning.formato import (
    ROTULO_INFORMATIVO,
    montar_resposta,
    referencias_citadas,
)

logger = logging.getLogger(__name__)


class ModeloIndisponivel(RuntimeError):
    """O backend pedido nao pode ser carregado neste ambiente."""


@dataclass
class RespostaModelo:
    """Saida do modelo com os metadados necessarios para a auditoria."""

    texto: str
    backend: str
    modelo: str
    adaptador: str | None
    tokens_gerados: int
    duracao_ms: int


class ModeloLinguagem(Protocol):
    """Interface minima que o restante do sistema conhece."""

    nome: str
    backend: str

    def gerar(self, mensagens: list[dict[str, str]], **parametros: Any) -> RespostaModelo:
        ...


class ModeloMLX:
    """Modelo quantizado em Apple Silicon, opcionalmente com adaptador LoRA."""

    backend = "mlx"

    def __init__(self, modelo: str | None = None, adaptador: Path | None = None,
                 usar_adaptador: bool | None = None) -> None:
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as erro:  # pragma: no cover - depende do ambiente
            raise ModeloIndisponivel(
                "mlx-lm nao esta instalado. Use 'pip install -r requirements-treino.txt' "
                "em Apple Silicon, ou defina BACKEND_LLM=eco."
            ) from erro

        self._generate = generate
        self._make_sampler = make_sampler
        self.nome = modelo or MODELO.modelo_base_mlx

        usar = MODELO.usar_adaptador if usar_adaptador is None else usar_adaptador
        caminho_adaptador = Path(adaptador or CAMINHOS.adaptador)
        self.adaptador: str | None = None
        if usar and (caminho_adaptador / "adapters.safetensors").exists():
            self.adaptador = str(caminho_adaptador)
        elif usar:
            logger.warning(
                "Adaptador LoRA nao encontrado em %s; usando o modelo base sem ajuste.",
                caminho_adaptador,
            )

        inicio = time.perf_counter()
        self._modelo, self._tokenizador = load(self.nome, adapter_path=self.adaptador)
        logger.info(
            "Modelo %s carregado em %.1fs (adaptador: %s)",
            self.nome, time.perf_counter() - inicio, self.adaptador or "nenhum",
        )

    def gerar(self, mensagens: list[dict[str, str]], **parametros: Any) -> RespostaModelo:
        max_tokens = int(parametros.get("max_tokens", MODELO.max_tokens))
        temperatura = float(parametros.get("temperatura", MODELO.temperatura))
        top_p = float(parametros.get("top_p", MODELO.top_p))

        prompt = self._tokenizador.apply_chat_template(
            mensagens, tokenize=False, add_generation_prompt=True
        )
        amostrador = self._make_sampler(temp=temperatura, top_p=top_p)

        inicio = time.perf_counter()
        texto = self._generate(
            self._modelo, self._tokenizador, prompt=prompt,
            max_tokens=max_tokens, sampler=amostrador, verbose=False,
        )
        duracao = int((time.perf_counter() - inicio) * 1000)
        tokens = len(self._tokenizador.encode(texto))
        return RespostaModelo(
            texto=texto.strip(), backend=self.backend, modelo=self.nome,
            adaptador=self.adaptador, tokens_gerados=tokens, duracao_ms=duracao,
        )


class ModeloTransformers:
    """Mesmo modelo em PyTorch, com adaptador PEFT. Usado em GPU NVIDIA ou CPU."""

    backend = "transformers"

    def __init__(self, modelo: str | None = None, adaptador: Path | None = None,
                 usar_adaptador: bool | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as erro:  # pragma: no cover
            raise ModeloIndisponivel(
                "transformers/torch nao estao instalados. Veja requirements-treino-cuda.txt."
            ) from erro

        self._torch = torch
        self.nome = modelo or MODELO.modelo_base_hf
        self._tokenizador = AutoTokenizer.from_pretrained(self.nome)
        self._modelo = AutoModelForCausalLM.from_pretrained(
            self.nome,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        usar = MODELO.usar_adaptador if usar_adaptador is None else usar_adaptador
        caminho_adaptador = Path(adaptador or CAMINHOS.adaptador)
        self.adaptador = None
        if usar and (caminho_adaptador / "adapter_config.json").exists():
            from peft import PeftModel

            self._modelo = PeftModel.from_pretrained(self._modelo, str(caminho_adaptador))
            self.adaptador = str(caminho_adaptador)
        self._modelo.eval()

    def gerar(self, mensagens: list[dict[str, str]], **parametros: Any) -> RespostaModelo:
        max_tokens = int(parametros.get("max_tokens", MODELO.max_tokens))
        temperatura = float(parametros.get("temperatura", MODELO.temperatura))

        entrada = self._tokenizador.apply_chat_template(
            mensagens, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self._modelo.device)

        inicio = time.perf_counter()
        with self._torch.no_grad():
            saida = self._modelo.generate(
                entrada,
                max_new_tokens=max_tokens,
                do_sample=temperatura > 0,
                temperature=max(temperatura, 1e-5),
                top_p=float(parametros.get("top_p", MODELO.top_p)),
                pad_token_id=self._tokenizador.eos_token_id,
            )
        duracao = int((time.perf_counter() - inicio) * 1000)
        gerados = saida[0][entrada.shape[-1]:]
        texto = self._tokenizador.decode(gerados, skip_special_tokens=True)
        return RespostaModelo(
            texto=texto.strip(), backend=self.backend, modelo=self.nome,
            adaptador=self.adaptador, tokens_gerados=int(gerados.shape[-1]), duracao_ms=duracao,
        )


class ModeloEco:
    """Backend deterministico, sem rede neural.

    Monta a resposta a partir do contexto recebido no prompt. Nao gera texto
    novo: extrai o que foi recuperado e o apresenta no formato canonico. Existe
    para que o fluxo completo possa ser testado sem depender de um modelo
    baixado, e para servir de linha de base na avaliacao.
    """

    backend = "eco"
    nome = "eco-deterministico"
    adaptador = None

    def gerar(self, mensagens: list[dict[str, str]], **parametros: Any) -> RespostaModelo:
        inicio = time.perf_counter()
        pergunta = ""
        for mensagem in reversed(mensagens):
            if mensagem["role"] == "user":
                pergunta = mensagem["content"]
                break

        contexto_documentos = ""
        if "### Contexto institucional recuperado" in pergunta:
            contexto_documentos = pergunta.split("### Contexto institucional recuperado", 1)[1]
        contexto_paciente = ""
        if "### Contexto do paciente" in pergunta:
            contexto_paciente = pergunta.split("### Contexto do paciente", 1)[1].split("###")[0]

        partes = []
        if contexto_paciente.strip():
            partes.append("Situacao registrada no prontuario: "
                          + " ".join(contexto_paciente.strip().splitlines()[:4]))
        if contexto_documentos.strip():
            trechos = [linha for linha in contexto_documentos.strip().splitlines() if linha.strip()]
            partes.append("Orientacao institucional aplicavel: " + " ".join(trechos[:6]))
        if not partes:
            partes.append(
                "Nao ha contexto institucional recuperado para esta pergunta. "
                "Consulte o protocolo aplicavel ou reformule a solicitacao."
            )

        corpo = "\n\n".join(partes)
        fontes = referencias_citadas(contexto_documentos) or []
        texto = montar_resposta(corpo, fontes, ROTULO_INFORMATIVO)
        duracao = int((time.perf_counter() - inicio) * 1000)
        return RespostaModelo(
            texto=texto, backend=self.backend, modelo=self.nome, adaptador=None,
            tokens_gerados=len(texto.split()), duracao_ms=duracao,
        )


_CACHE: dict[tuple, ModeloLinguagem] = {}


def carregar_modelo(backend: str | None = None, modelo: str | None = None,
                    adaptador: Path | None = None, usar_adaptador: bool | None = None,
                    cache: bool = True) -> ModeloLinguagem:
    """Instancia o backend pedido, com cache por combinacao de parametros.

    Se o backend preferido falhar (por exemplo, MLX em maquina sem Apple
    Silicon), cai para o backend ``eco`` com aviso, para que o sistema continue
    auditavel em vez de quebrar.
    """
    escolhido = (backend or MODELO.backend).lower()
    chave = (escolhido, modelo, str(adaptador), usar_adaptador)
    if cache and chave in _CACHE:
        return _CACHE[chave]

    construtores = {
        "mlx": lambda: ModeloMLX(modelo, adaptador, usar_adaptador),
        "transformers": lambda: ModeloTransformers(modelo, adaptador, usar_adaptador),
        "eco": ModeloEco,
    }
    if escolhido not in construtores:
        raise ValueError(f"Backend desconhecido: {escolhido}")

    try:
        instancia = construtores[escolhido]()
    except (ModeloIndisponivel, Exception) as erro:
        if escolhido == "eco":
            raise
        logger.warning("Falha ao carregar backend '%s' (%s). Usando backend 'eco'.",
                       escolhido, erro)
        instancia = ModeloEco()

    if cache:
        _CACHE[chave] = instancia
    return instancia


def limpar_cache() -> None:
    _CACHE.clear()
