"""Envio de gravacoes pela equipe: salva o arquivo, analisa, alerta e reavalia a fusao.

E o caminho usado pela aba Monitoramento para processar um video (sessao de
fisioterapia, camera do quarto) ou um audio de consulta enviado pelo navegador.
O resultado passa pela mesma central de alertas e pela mesma fusao do fluxo continuo.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import CAMINHOS
from . import fusao
from .alertas import CentralAlertas

PASTA_ENVIOS = CAMINHOS.multimodal / "envios"
EXTENSOES_VIDEO = (".mp4", ".mov", ".avi", ".webm", ".ogv", ".mkv")
EXTENSOES_AUDIO = (".wav", ".mp3", ".flac")
LIMITE_BYTES = 200 * 1024 * 1024


class EnvioInvalido(ValueError):
    """Arquivo com extensao nao suportada, vazio ou grande demais."""


@dataclass
class ResultadoEnvio:
    midia: str
    arquivo: Path
    prontuario: str
    analise: Any
    relatorio: Path | None
    alertas_novos: list[dict[str, Any]] = field(default_factory=list)
    consolidados_novos: list[dict[str, Any]] = field(default_factory=list)


def midia_do_arquivo(nome: str) -> str | None:
    """'video' ou 'audio' pela extensao; None se o formato nao e suportado."""
    extensao = Path(nome).suffix.lower()
    if extensao in EXTENSOES_VIDEO:
        return "video"
    if extensao in EXTENSOES_AUDIO:
        return "audio"
    return None


def salvar_envio(nome: str, conteudo: bytes, midia: str, pasta: Path | None = None) -> Path:
    """Guarda o arquivo enviado com nome seguro e prefixo do conteudo (reenvio reaproveita o cache)."""
    extensoes = EXTENSOES_VIDEO if midia == "video" else EXTENSOES_AUDIO
    original = Path(nome)
    extensao = original.suffix.lower()
    if extensao not in extensoes:
        raise EnvioInvalido(f"Formato {extensao or 'sem extensao'} nao suportado; use "
                            + ", ".join(extensoes))
    if not conteudo:
        raise EnvioInvalido("Arquivo vazio.")
    if len(conteudo) > LIMITE_BYTES:
        raise EnvioInvalido("Arquivo acima de 200 MB.")
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", original.stem).strip("_")[:60] or midia
    resumo = hashlib.sha1(conteudo).hexdigest()[:10]
    destino_pasta = (pasta or PASTA_ENVIOS) / midia
    destino_pasta.mkdir(parents=True, exist_ok=True)
    destino = destino_pasta / f"{resumo}_{base}{extensao}"
    if not destino.exists():
        destino.write_bytes(conteudo)
    return destino


def _quadros(caminho: Path) -> int:
    try:
        import cv2
    except ImportError:
        return 0
    captura = cv2.VideoCapture(str(caminho))
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    captura.release()
    return total


def _consolidar(banco, central: CentralAlertas, prontuario: str,
                antes: set[int]) -> tuple[list[dict], list[dict]]:
    fusao.consolidar(banco, central)
    novos = [a for a in banco.listar_alertas(prontuario=prontuario, limite=500) if a["id"] not in antes]
    consolidados = [a for a in novos if a["categoria"] == "anomalia_multimodal"]
    individuais = [a for a in novos if a["categoria"] != "anomalia_multimodal"]
    return individuais, consolidados


def _ids_abertos(banco, prontuario: str) -> set[int]:
    return {a["id"] for a in banco.listar_alertas(prontuario=prontuario, limite=500)}


def analisar_video_enviado(banco, caminho: Path, prontuario: str, tipo: str = "auto",
                           areas: Path | None = None, anotar: bool = True,
                           central: CentralAlertas | None = None,
                           progresso: Callable[[float, str], None] | None = None,
                           pasta_saida: Path | None = None) -> ResultadoEnvio:
    from .video.analise import PASTA_RELATORIOS, analisar_video
    from .video.extracao import extrair

    central = central or CentralAlertas(banco)
    antes = _ids_abertos(banco, prontuario)
    total = _quadros(caminho)

    def avancar(indice: int) -> None:
        if progresso and total:
            progresso(min(indice / total, 1.0) * 0.85, f"Pose e objetos: quadro {indice}/{total}")

    if progresso:
        progresso(0.0, "Carregando YOLOv8-pose e YOLOv8...")
    sequencia = extrair(caminho, progresso=avancar)
    if progresso:
        progresso(0.88, "Detectando quedas, exercicio e areas criticas; gerando relatorio...")
    saida = pasta_saida or PASTA_RELATORIOS
    resultado = analisar_video(caminho, tipo=tipo, prontuario=prontuario, config_areas=areas,
                               banco=banco, central=central, pasta_saida=saida, anotar=anotar,
                               sequencia=sequencia)
    if progresso:
        progresso(0.97, "Reavaliando o risco do paciente (fusao)...")
    individuais, consolidados = _consolidar(banco, central, prontuario, antes)
    if progresso:
        progresso(1.0, "Concluido")
    return ResultadoEnvio("video", caminho, prontuario, resultado, saida / f"{caminho.stem}.md",
                          individuais, consolidados)


def basal_do_paciente(prontuario: str) -> Path | None:
    """Gravacao de linha de base ja registrada para o paciente, se houver.

    So faz sentido comparar com ela quando a gravacao enviada e da mesma voz.
    """
    from .audio.roteiros import PASTA_AUDIO

    candidato = PASTA_AUDIO / "consultas" / f"{prontuario}_basal.wav"
    return candidato if candidato.exists() else None


def analisar_audio_enviado(banco, caminho: Path, prontuario: str, basal: Path | None = None,
                           usar_azure: bool = True, central: CentralAlertas | None = None,
                           progresso: Callable[[float, str], None] | None = None,
                           pasta_relatorios: Path | None = None) -> ResultadoEnvio:
    from .audio.consulta import analisar_consulta, salvar_relatorio
    from .audio.roteiros import PASTA_AUDIO

    central = central or CentralAlertas(banco)
    antes = _ids_abertos(banco, prontuario)
    if progresso:
        progresso(0.1, "Medindo a voz (Praat)"
                  + (", transcrevendo e analisando o texto (Azure)..." if usar_azure else "..."))
    resultado = analisar_consulta(caminho, prontuario=prontuario,
                                  basal=basal, banco=banco,
                                  central=central, usar_azure=usar_azure)
    relatorio = salvar_relatorio(resultado, pasta_relatorios or PASTA_AUDIO / "relatorios")
    if progresso:
        progresso(0.95, "Reavaliando o risco do paciente (fusao)...")
    individuais, consolidados = _consolidar(banco, central, prontuario, antes)
    if progresso:
        progresso(1.0, "Concluido")
    return ResultadoEnvio("audio", caminho, prontuario, resultado, relatorio, individuais, consolidados)
