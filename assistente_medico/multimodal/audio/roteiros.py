"""Consultas sinteticas em portugues, geradas com o Azure Text to Speech.

Cada paciente tem uma gravacao de linha de base (fala habitual, na admissao) e
o relato da consulta atual. A condicao vocal e simulada por SSML: velocidade,
tom, volume, amplitude de entonacao e pausas. O TTS reproduz bem o *ritmo* e a
*prosodia* dessas condicoes, mas nao a qualidade vocal (rouquidao, soprosidade);
por isso jitter, shimmer e HNR sao validados a parte, com vozes reais (VOICED).

Todo o conteudo e ficticio.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from ...config import CAMINHOS

PASTA_AUDIO = CAMINHOS.multimodal / "audio"

TEXTO_BASAL = (
    "Bom dia. Eu moro com a minha família e trabalhei muitos anos no comércio. "
    "Gosto de caminhar de manhã e de cuidar das plantas do quintal. "
    "Normalmente durmo bem e como de tudo. Vim para cá porque o médico do posto pediu "
    "que eu fizesse alguns exames."
)

TEXTO_NEUTRO = (
    "Hoje eu acordei cedo e fiquei sentado na cadeira perto da janela. "
    "A enfermeira veio medir a pressão e trouxe o café da manhã. "
    "Depois eu almocei um pouco e tentei caminhar no corredor com o meu filho. "
    "A minha família vem me visitar mais tarde e vai trazer umas roupas limpas."
)

VOZES_AVALIACAO = ("pt-BR-FranciscaNeural", "pt-BR-AntonioNeural", "pt-BR-BrendaNeural",
                   "pt-BR-DonatoNeural", "pt-BR-LeilaNeural", "pt-BR-FabioNeural")
CONDICOES = ("normal", "fadiga", "dispneia", "disartria")


@dataclass(frozen=True)
class Roteiro:
    prontuario: str
    voz: str
    condicao: str
    texto: str
    descricao: str


ROTEIROS = (
    Roteiro("117256", "pt-BR-ElzaNeural", "normal",
            "Estou me sentindo bem melhor, doutor. A dor no corte diminuiu bastante e ontem eu já "
            "consegui andar até o fim do corredor. Estou comendo direitinho e dormi a noite toda. "
            "Só queria saber quando vou poder ir para casa.",
            "pos-operatorio sem intercorrencia (controle)"),
    Roteiro("208731", "pt-BR-HumbertoNeural", "fadiga",
            "Eu estou muito cansado, doutor. Não tenho força nem para levantar da cama. "
            "Desde ontem eu não tenho vontade de comer e passo o dia todo deitado. "
            "Parece que o corpo pesa.",
            "fadiga na fala e no conteudo"),
    Roteiro("559120", "pt-BR-JulioNeural", "dispneia",
            "Doutor, eu estou com muita falta de ar desde ontem a noite. Qualquer coisa que eu "
            "faço eu fico cansado para respirar, até para falar. A tosse continua com catarro "
            "amarelado e eu tive febre de madrugada.",
            "dificuldade respiratoria na fala e no conteudo"),
    Roteiro("553119", "pt-BR-YaraNeural", "disartria",
            "Eu acordei e a minha fala estava enrolada. O meu braço direito está fraco e eu não "
            "consigo segurar o copo. A minha filha disse que a minha boca está torta.",
            "fala arrastada com sinais neurologicos focais (AVC)"),
    Roteiro("771203", "pt-BR-NicolauNeural", "normal",
            "Doutor, eu estou com uma dor no peito muito forte, apertando, que vai para o braço "
            "esquerdo. Começou faz uma hora e eu estou suando frio. Eu fiquei com muito medo.",
            "voz sem alteracao, conteudo critico (dor toracica)"),
    Roteiro("875431", "pt-BR-ValerioNeural", "normal",
            "Meu nome é {nome}. Não tenho dor no peito e nem falta de ar. A tosse está "
            "melhorando e o catarro diminuiu. Só dormi mal porque fiquei agitado de madrugada, "
            "mas agora estou tranquilo.",
            "negacoes (nao devem virar alerta) e nome citado (deve ser pseudonimizado)"),
)


def _sentencas(texto: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", texto) if s.strip()]


def ssml(texto: str, voz: str, condicao: str, semente: int = 7) -> str:
    """Monta o SSML que simula a condicao vocal sobre o texto."""
    gerador = random.Random(f"{semente}-{voz}-{condicao}")
    partes = []
    if condicao == "normal":
        partes.append(f'<prosody rate="0%">{escape(texto)}</prosody>')
    elif condicao == "fadiga":
        corpo = "".join(
            f'{escape(s)}<break time="{gerador.randint(900, 1400)}ms"/>' for s in _sentencas(texto))
        partes.append(f'<prosody rate="-25%" pitch="-10%" range="-50%" volume="-40%">{corpo}</prosody>')
    elif condicao == "dispneia":
        palavras = texto.split()
        corpo, contador, limite = [], 0, gerador.randint(2, 4)
        for palavra in palavras:
            corpo.append(escape(palavra))
            contador += 1
            if contador >= limite:
                corpo.append(f'<break time="{gerador.randint(450, 700)}ms"/>')
                contador, limite = 0, gerador.randint(2, 4)
        partes.append(f'<prosody rate="+5%" volume="-10%">{" ".join(corpo)}</prosody>')
    elif condicao == "disartria":
        corpo = []
        for palavra in texto.split():
            corpo.append(escape(palavra))
            if gerador.random() < 0.45:
                corpo.append(f'<break time="{gerador.randint(120, 380)}ms"/>')
        partes.append(f'<prosody rate="-45%" range="-60%">{" ".join(corpo)}</prosody>')
    else:
        raise ValueError(f"Condicao desconhecida: {condicao}")
    return (f'<speak version="1.0" xml:lang="pt-BR"><voice name="{voz}">'
            f'{"".join(partes)}</voice></speak>')


def _gravar(destino: Path, texto: str, voz: str, condicao: str,
            sintetizar: Callable[[str], bytes], metadados: dict) -> bool:
    if destino.exists():
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(sintetizar(ssml(texto, voz, condicao)))
    destino.with_suffix(".txt").write_text(texto, encoding="utf-8")
    destino.with_suffix(".json").write_text(
        json.dumps({**metadados, "voz": voz, "condicao": condicao}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    time.sleep(3.2)  # camada F0: no maximo 20 sinteses por minuto
    return True


def gerar_consultas(nomes: dict[str, str], sintetizar: Callable[[str], bytes],
                    pasta: Path | None = None) -> list[Path]:
    pasta = Path(pasta or PASTA_AUDIO) / "consultas"
    gerados = []
    for roteiro in ROTEIROS:
        texto = roteiro.texto.format(nome=nomes.get(roteiro.prontuario, "Jose da Silva"))
        meta = {k: v for k, v in asdict(roteiro).items() if k not in {"texto", "voz", "condicao"}}
        basal = pasta / f"{roteiro.prontuario}_basal.wav"
        consulta = pasta / f"{roteiro.prontuario}_consulta.wav"
        _gravar(basal, TEXTO_BASAL, roteiro.voz, "normal", sintetizar, {**meta, "tipo": "basal"})
        _gravar(consulta, texto, roteiro.voz, roteiro.condicao, sintetizar,
                {**meta, "tipo": "consulta"})
        gerados += [basal, consulta]
    return gerados


def gerar_avaliacao(sintetizar: Callable[[str], bytes], pasta: Path | None = None) -> list[Path]:
    """Mesmo texto neutro, quatro condicoes, seis vozes: so a forma de falar muda."""
    pasta = Path(pasta or PASTA_AUDIO) / "avaliacao"
    gerados = []
    for voz in VOZES_AVALIACAO:
        nome = voz.split("-")[2].replace("Neural", "").lower()
        _gravar(pasta / f"{nome}_basal.wav", TEXTO_BASAL, voz, "normal", sintetizar,
                {"tipo": "basal"})
        gerados.append(pasta / f"{nome}_basal.wav")
        for condicao in CONDICOES:
            destino = pasta / f"{nome}_{condicao}.wav"
            _gravar(destino, TEXTO_NEUTRO, voz, condicao, sintetizar, {"tipo": "avaliacao"})
            gerados.append(destino)
    return gerados
