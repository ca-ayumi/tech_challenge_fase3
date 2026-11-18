"""Monitoramento multimodal (Fase 4): video, audio, sinais vitais, prescricoes e movimentacao.

Cada modalidade produz ``Anomalia``; a ``CentralAlertas`` decide o que vira
alerta para a equipe, e o ``Monitor`` avalia leituras em fluxo.
"""

from .alertas import CentralAlertas, NotificadorArquivo, NotificadorWebhook, ResultadoAlerta
from .tempo_real import Leitura, Monitor, ResumoMonitoramento
from .tipos import MODALIDADES, SEVERIDADES, Anomalia

__all__ = [
    "MODALIDADES", "SEVERIDADES", "Anomalia", "CentralAlertas", "Leitura", "Monitor",
    "NotificadorArquivo", "NotificadorWebhook", "ResultadoAlerta", "ResumoMonitoramento",
]
