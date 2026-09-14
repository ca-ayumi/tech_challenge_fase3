"""Configuracao central do assistente medico.

Todos os caminhos derivam da raiz do repositorio, e todo parametro ajustavel
pode ser sobrescrito por variavel de ambiente (arquivo .env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv e opcional
    pass

RAIZ = Path(__file__).resolve().parents[1]


def _caminho(variavel: str, padrao: str) -> Path:
    return Path(os.getenv(variavel, str(RAIZ / padrao))).expanduser()


def _inteiro(variavel: str, padrao: int) -> int:
    try:
        return int(os.getenv(variavel, padrao))
    except (TypeError, ValueError):
        return padrao


def _decimal(variavel: str, padrao: float) -> float:
    try:
        return float(os.getenv(variavel, padrao))
    except (TypeError, ValueError):
        return padrao


def _booleano(variavel: str, padrao: bool) -> bool:
    valor = os.getenv(variavel)
    if valor is None:
        return padrao
    return valor.strip().lower() in {"1", "true", "yes", "sim", "on"}


@dataclass(frozen=True)
class Caminhos:
    raiz: Path = RAIZ
    dados_brutos: Path = field(default_factory=lambda: _caminho("DIR_DADOS_BRUTOS", "dados/brutos"))
    dados_processados: Path = field(
        default_factory=lambda: _caminho("DIR_DADOS_PROCESSADOS", "dados/processados")
    )
    protocolos: Path = field(default_factory=lambda: _caminho("DIR_PROTOCOLOS", "dados/brutos/protocolos"))
    modelos_documentos: Path = field(
        default_factory=lambda: _caminho("DIR_MODELOS_DOC", "dados/brutos/modelos_documentos")
    )
    banco: Path = field(default_factory=lambda: _caminho("CAMINHO_BANCO", "dados/hospital.db"))
    modelos: Path = field(default_factory=lambda: _caminho("DIR_MODELOS", "modelos"))
    adaptador: Path = field(default_factory=lambda: _caminho("DIR_ADAPTADOR", "modelos/adaptador-lora"))
    logs: Path = field(default_factory=lambda: _caminho("DIR_LOGS", "logs"))
    avaliacao: Path = field(default_factory=lambda: _caminho("DIR_AVALIACAO", "avaliacao/resultados"))

    def garantir(self) -> None:
        for destino in (self.dados_processados, self.modelos, self.logs, self.avaliacao):
            destino.mkdir(parents=True, exist_ok=True)
        self.banco.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ConfigModelo:
    """Parametros do modelo de linguagem usado pelo assistente."""

    backend: str = os.getenv("BACKEND_LLM", "mlx")
    modelo_base_mlx: str = os.getenv("MODELO_BASE_MLX", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    modelo_base_hf: str = os.getenv("MODELO_BASE_HF", "Qwen/Qwen2.5-1.5B-Instruct")
    usar_adaptador: bool = _booleano("USAR_ADAPTADOR", True)
    temperatura: float = _decimal("TEMPERATURA", 0.2)
    max_tokens: int = _inteiro("MAX_TOKENS", 700)
    top_p: float = _decimal("TOP_P", 0.9)


@dataclass(frozen=True)
class ConfigTreino:
    """Hiperparametros do fine-tuning por LoRA."""

    camadas_lora: int = _inteiro("LORA_CAMADAS", 8)
    rank: int = _inteiro("LORA_RANK", 16)
    escala: float = _decimal("LORA_ESCALA", 20.0)
    iteracoes: int = _inteiro("TREINO_ITERACOES", 400)
    lote: int = _inteiro("TREINO_LOTE", 2)
    taxa_aprendizado: float = _decimal("TREINO_LR", 1e-5)
    passos_validacao: int = _inteiro("TREINO_PASSOS_VAL", 50)
    max_tokens_sequencia: int = _inteiro("TREINO_MAX_SEQ", 1280)
    semente: int = _inteiro("SEMENTE", 42)


@dataclass(frozen=True)
class ConfigRecuperacao:
    """Parametros da recuperacao de trechos de protocolo (RAG)."""

    top_k: int = _inteiro("RAG_TOP_K", 4)
    tamanho_trecho: int = _inteiro("RAG_TAMANHO_TRECHO", 1200)
    minimo_score: float = _decimal("RAG_MIN_SCORE", 0.8)


@dataclass(frozen=True)
class ConfigSeguranca:
    """Limites de atuacao e politica de auditoria."""

    exigir_validacao_humana: bool = _booleano("EXIGIR_VALIDACAO_HUMANA", True)
    mascarar_pii_nos_logs: bool = _booleano("MASCARAR_PII_LOGS", True)
    perfis_autorizados: tuple[str, ...] = ("medico", "enfermeiro", "farmaceutico", "residente")
    retencao_auditoria_anos: int = _inteiro("RETENCAO_AUDITORIA_ANOS", 5)


CAMINHOS = Caminhos()
MODELO = ConfigModelo()
TREINO = ConfigTreino()
RECUPERACAO = ConfigRecuperacao()
SEGURANCA = ConfigSeguranca()

SEMENTE = TREINO.semente

NOME_ASSISTENTE = os.getenv("NOME_ASSISTENTE", "Assistente Clínico Institucional")
VERSAO_ASSISTENTE = os.getenv("VERSAO_ASSISTENTE", "1.0.0")
