"""Demonstracao ponta a ponta do monitoramento multimodal.

1. Fluxo das ultimas horas de internacao (sinais vitais, prescricoes, sensor de
   leito) com a queda real do UR Fall (acelerometro) na madrugada do ENF-221.
2. Video da *mesma* queda (UR Fall grava acelerometro e camera juntos),
   alinhado no tempo, analisado com YOLOv8.
3. Audio das consultas: voz, Azure Speech to Text, Azure Language.
4. Fusao: escore de risco por paciente e alertas consolidados.

Usada pelo CLI (``monitorar demo``) e pela aba Monitoramento do Streamlit.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import CAMINHOS
from . import fusao
from .alertas import CentralAlertas
from .anomalias import (
    DetectorMovimentacao,
    DetectorPrescricoes,
    DetectorSinaisVitais,
    ParametrosSinaisVitais,
    avaliacao,
    sintetico,
)
from .anomalias.fontes import ler_urfall, registros_urfall
from .tempo_real import Leitura, Monitor

PACIENTE_QUEDA = "875431"
SEQUENCIA_QUEDA = "fall-01"


def momento_queda(dados: dict[str, Any], prontuario: str = PACIENTE_QUEDA) -> datetime | None:
    """A queda e encaixada 7 min depois do inicio da saida noturna do leito."""
    saidas = [r for r in dados["rotulos"]
              if r["prontuario"] == prontuario and r["evento"] == "saida_noturna"]
    return datetime.fromisoformat(saidas[0]["inicio"]) + timedelta(minutes=7) if saidas else None


def leituras_queda(dados: dict[str, Any], prontuario: str = PACIENTE_QUEDA) -> list[Leitura]:
    momento = momento_queda(dados, prontuario)
    arquivos = [c for c in registros_urfall() if c.name.startswith(SEQUENCIA_QUEDA)]
    if momento is None or not arquivos:
        return []
    return ler_urfall(arquivos[0], prontuario=prontuario, inicio=momento)


def fluxo(banco, central: CentralAlertas, dados: dict[str, Any], horas: float = 24.0,
          prontuario: str | None = None, com_queda: bool = True,
          aceleracao: float | None = None,
          ao_processar: Callable | None = None, analise_id: int | None = None):
    leituras = avaliacao.leituras_internacao(dados)
    if com_queda:
        leituras = sorted(leituras + leituras_queda(dados), key=lambda item: item.instante)
    if prontuario:
        leituras = [leitura for leitura in leituras if leitura.prontuario == prontuario]
    corte = leituras[-1].instante - timedelta(hours=horas) if leituras else None
    detectores = [
        DetectorSinaisVitais(ParametrosSinaisVitais.horario()),
        DetectorPrescricoes.do_banco(banco),
        DetectorMovimentacao(risco_queda=dados["risco_queda"]),
    ]
    for leitura in leituras:
        if corte and leitura.instante < corte:
            for detector in detectores:
                if leitura.modalidade in detector.modalidades:
                    detector.atualizar(leitura)
    em_fluxo = [leitura for leitura in leituras if not corte or leitura.instante >= corte]
    monitor = Monitor(central, detectores, analise_id=analise_id)
    return monitor.reproduzir(em_fluxo, aceleracao=aceleracao, ao_processar=ao_processar)


def executar(banco, usar_azure: bool = True, usar_video: bool = True, horas: float = 24.0,
             progresso: Callable[[str], None] = print,
             ao_processar: Callable | None = None) -> dict[str, Any]:
    from .relogio import agora

    banco.limpar_monitoramento()
    dados = sintetico.preparar_monitoramento(banco)
    central = CentralAlertas(banco)
    resumo: dict[str, Any] = {"inicio": datetime.now().isoformat(timespec="seconds")}

    progresso(f"1/4 Fluxo das ultimas {horas:.0f} h: sinais vitais, prescricoes, movimentacao")
    analise = banco.abrir_analise("monitoramento", origem="demonstracao")
    resultado_fluxo = fluxo(banco, central, dados, horas=horas, ao_processar=ao_processar,
                            analise_id=analise)
    banco.concluir_analise(analise, resultado_fluxo.como_dicionario())
    resumo["fluxo"] = resultado_fluxo.como_dicionario()

    progresso("2/4 Video da mesma queda (YOLOv8-pose + YOLOv8)")
    resumo["video"] = None
    video = CAMINHOS.multimodal / "urfall_video" / f"{SEQUENCIA_QUEDA}-cam0.mp4"
    momento = momento_queda(dados)
    if usar_video and video.exists() and momento:
        try:
            from .video.analise import PASTA_RELATORIOS, analisar_video

            areas = Path(__file__).parent / "video" / "configuracoes" / "urfall_cam0.json"
            r = analisar_video(video, tipo="quarto", prontuario=PACIENTE_QUEDA, config_areas=areas,
                               banco=banco, central=central, pasta_saida=PASTA_RELATORIOS,
                               inicio_gravacao=momento)
            resumo["video"] = {"anomalias": [a.tipo for a in r.anomalias]}
        except ImportError as erro:
            progresso(f"    video ignorado (instale requirements-video.txt): {erro}")
    else:
        progresso("    video ignorado (rode scripts/baixar_datasets.py urfall_video)")

    progresso("3/4 Consultas de audio" + (" com Azure Speech e Language" if usar_azure else " (local)"))
    resumo["audio"] = []
    from .audio.consulta import analisar_consulta, salvar_relatorio
    from .audio.roteiros import PASTA_AUDIO

    referencia = agora(banco)
    for indice, arquivo in enumerate(sorted((PASTA_AUDIO / "consultas").glob("*_consulta.wav"))):
        r = analisar_consulta(arquivo, banco=banco, central=central, usar_azure=usar_azure,
                              instante=referencia - timedelta(minutes=30 - indice))
        salvar_relatorio(r, PASTA_AUDIO / "relatorios")
        resumo["audio"].append({"prontuario": r.prontuario, "anomalias": [a.tipo for a in r.anomalias]})
    if not resumo["audio"]:
        progresso("    sem consultas: rode 'audio gerar'")

    progresso("4/4 Fusao multimodal: risco por paciente e alertas consolidados")
    consolidados = fusao.consolidar(banco, central)
    resumo["fusao"] = [
        {**avaliacao_paciente.como_dicionario(),
         "alertas": [r.alerta_id for r in resultados if r.emitido]}
        for avaliacao_paciente, resultados in consolidados
    ]
    return resumo
