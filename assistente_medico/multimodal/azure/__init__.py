"""Integracao com os servicos cognitivos da Azure (Speech e Language)."""

from .cliente import ErroAzure, RespostaAzure, requisitar
from .diagnostico import StatusServico, verificar_azure

__all__ = ["ErroAzure", "RespostaAzure", "StatusServico", "requisitar", "verificar_azure"]
