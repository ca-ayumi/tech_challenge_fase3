"""Azure Speech: sintese (Text to Speech) e transcricao (Speech to Text).

A transcricao usa a API REST de audio curto, a que a camada gratuita (F0)
oferece: ate 60 s por chamada, com tempo de inicio e duracao de cada palavra.
Audios mais longos sao cortados no meio das pausas, nunca no meio de uma
palavra, e os tempos sao reajustados para o audio inteiro.

A transcricao rapida (fast transcription), que tambem separa os falantes, e
exclusiva da camada paga; com ela, o mesmo pipeline isolaria a voz do paciente
num dialogo. Na F0 analisamos gravacoes do relato do paciente.
"""

from __future__ import annotations

import io
import time
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...config import AZURE, ConfigAzure
from .cliente import ErroAzure, requisitar

FORMATO_SAIDA = "riff-16khz-16bit-mono-pcm"
LIMITE_TRECHO_S = 55.0
TICKS_POR_SEGUNDO = 10_000_000


@dataclass(frozen=True)
class Palavra:
    texto: str
    inicio_s: float
    duracao_s: float


@dataclass
class Transcricao:
    texto: str
    fonte: str
    palavras: list[Palavra] = field(default_factory=list)
    confianca: float | None = None
    trechos: int = 0

    @property
    def palavras_por_minuto(self) -> float | None:
        if len(self.palavras) < 2:
            return None
        inicio = self.palavras[0].inicio_s
        fim = self.palavras[-1].inicio_s + self.palavras[-1].duracao_s
        return 60 * len(self.palavras) / max(fim - inicio, 1e-6)

    def como_dicionario(self) -> dict[str, Any]:
        return {"texto": self.texto, "fonte": self.fonte, "confianca": self.confianca,
                "trechos": self.trechos, "palavras": len(self.palavras),
                "palavras_por_minuto": (round(self.palavras_por_minuto, 1)
                                        if self.palavras_por_minuto else None),
                "tempos_palavras": [asdict(p) for p in self.palavras]}


def sintetizar(ssml: str, config: ConfigAzure | None = None) -> bytes:
    """Converte SSML em WAV PCM 16 kHz mono."""
    config = config or AZURE
    if not config.fala_configurada:
        raise ErroAzure("Azure Speech nao configurado (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION).")
    resposta = requisitar(
        "POST", f"https://{config.regiao_fala}.tts.speech.microsoft.com/cognitiveservices/v1",
        config.chave_fala, corpo=ssml.encode("utf-8"), timeout_s=config.timeout_s,
        cabecalhos={"Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": FORMATO_SAIDA,
                    "User-Agent": "assistente-medico"},
    )
    return resposta.corpo


def _ler_wav(conteudo: bytes) -> tuple[bytes, int, int, int]:
    with wave.open(io.BytesIO(conteudo)) as arquivo:
        return (arquivo.readframes(arquivo.getnframes()), arquivo.getframerate(),
                arquivo.getsampwidth(), arquivo.getnchannels())


def _montar_wav(quadros: bytes, taxa: int, largura: int, canais: int) -> bytes:
    saida = io.BytesIO()
    with wave.open(saida, "wb") as arquivo:
        arquivo.setnchannels(canais)
        arquivo.setsampwidth(largura)
        arquivo.setframerate(taxa)
        arquivo.writeframes(quadros)
    return saida.getvalue()


def wav_compativel(caminho: Path) -> bytes:
    """Devolve o audio como WAV PCM 16 bits mono 16 kHz, convertendo se preciso."""
    conteudo = Path(caminho).read_bytes()
    try:
        _, taxa, largura, canais = _ler_wav(conteudo)
        if taxa == 16000 and largura == 2 and canais == 1:
            return conteudo
    except wave.Error:
        pass
    import tempfile

    import parselmouth

    som = parselmouth.Sound(str(caminho)).convert_to_mono().resample(16000)
    with tempfile.TemporaryDirectory() as pasta:
        destino = Path(pasta) / "convertido.wav"
        som.save(str(destino), "WAV")
        return destino.read_bytes()


def cortes(duracao_s: float, pausas: Sequence[tuple[float, float]],
           limite_s: float = LIMITE_TRECHO_S) -> list[float]:
    """Pontos de corte no meio das pausas, de modo que nenhum trecho passe do limite."""
    pontos = []
    inicio = 0.0
    meios = sorted((a + b) / 2 for a, b in pausas)
    while duracao_s - inicio > limite_s:
        candidatos = [m for m in meios if inicio < m <= inicio + limite_s]
        corte = candidatos[-1] if candidatos else inicio + limite_s
        pontos.append(corte)
        inicio = corte
    return pontos


def _transcrever_trecho(conteudo: bytes, config: ConfigAzure) -> dict[str, Any]:
    resposta = requisitar(
        "POST",
        f"https://{config.regiao_fala}.stt.speech.microsoft.com/speech/recognition/conversation/"
        f"cognitiveservices/v1?language={config.idioma}&format=detailed&wordLevelTimestamps=true",
        config.chave_fala, corpo=conteudo, timeout_s=max(config.timeout_s, 60),
        cabecalhos={"Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                    "Accept": "application/json"},
    )
    return resposta.json() or {}


def transcrever(caminho: Path, pausas: Sequence[tuple[float, float]] = (),
                config: ConfigAzure | None = None) -> Transcricao:
    config = config or AZURE
    if not config.fala_configurada:
        raise ErroAzure("Azure Speech nao configurado (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION).")
    quadros, taxa, largura, canais = _ler_wav(wav_compativel(caminho))
    por_segundo = taxa * largura * canais
    duracao = len(quadros) / por_segundo
    limites = [0.0, *cortes(duracao, pausas), duracao]

    textos, palavras, confiancas = [], [], []
    for indice, (inicio, fim) in enumerate(zip(limites, limites[1:], strict=False)):
        a = int(inicio * taxa) * largura * canais
        b = int(fim * taxa) * largura * canais
        if indice:
            time.sleep(1.0)
        resultado = _transcrever_trecho(_montar_wav(quadros[a:b], taxa, largura, canais), config)
        if resultado.get("RecognitionStatus") != "Success" or not resultado.get("NBest"):
            continue
        melhor = resultado["NBest"][0]
        textos.append(melhor.get("Display", ""))
        if melhor.get("Confidence") is not None:
            confiancas.append(float(melhor["Confidence"]))
        for palavra in melhor.get("Words") or []:
            palavras.append(Palavra(
                texto=palavra["Word"],
                inicio_s=round(inicio + palavra["Offset"] / TICKS_POR_SEGUNDO, 3),
                duracao_s=round(palavra["Duration"] / TICKS_POR_SEGUNDO, 3),
            ))
    return Transcricao(
        texto=" ".join(t for t in textos if t).strip(), fonte="azure_speech",
        palavras=palavras, trechos=len(limites) - 1,
        confianca=round(sum(confiancas) / len(confiancas), 4) if confiancas else None,
    )
