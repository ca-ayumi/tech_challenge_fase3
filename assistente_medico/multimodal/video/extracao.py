"""Extracao de pose e objetos com YOLOv8.

- **YOLOv8-pose** (``yolov8s-pose``): 17 pontos do corpo no padrao COCO, o mesmo
  esqueleto de 17 pontos usado pelo OpenPose no modo COCO. Escolhido no lugar do
  OpenPose por rodar no Apple Silicon (MPS) e em CPU sem compilacao de CUDA/Caffe.
- **YOLOv8** (``yolov8n``): objetos do COCO (cama, cadeira, sofa, objetos
  cortantes...), que dao contexto (deitado na cama x no chao) e alimentam as
  areas criticas.

A inferencia roda uma vez por video e o resultado fica em cache (``.npz``); o
restante da analise trabalha sobre os pontos, sem depender do modelo.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...config import CAMINHOS

MODELO_POSE = "yolov8s-pose.pt"
MODELO_OBJETOS = "yolov8n.pt"
PASTA_CACHE = CAMINHOS.multimodal / "cache_video"
NOMES_PONTOS = ("nariz", "olho_esq", "olho_dir", "orelha_esq", "orelha_dir", "ombro_esq",
                "ombro_dir", "cotovelo_esq", "cotovelo_dir", "punho_esq", "punho_dir",
                "quadril_esq", "quadril_dir", "joelho_esq", "joelho_dir", "tornozelo_esq",
                "tornozelo_dir")


@dataclass
class SequenciaPose:
    """Pontos e caixas da pessoa principal, quadro a quadro, mais os objetos detectados."""

    origem: str
    fps: float
    largura: int
    altura: int
    tempos: np.ndarray
    pontos: np.ndarray
    caixas: np.ndarray
    confianca: np.ndarray
    objetos: list[dict] = field(default_factory=list)
    recorte: str | None = None
    todas_caixas: np.ndarray | None = None
    todas_confiancas: np.ndarray | None = None
    todos_pontos: np.ndarray | None = None

    @property
    def quadros(self) -> int:
        return len(self.tempos)

    def salvar(self, destino: Path) -> Path:
        destino.parent.mkdir(parents=True, exist_ok=True)
        extras = {}
        if self.todas_caixas is not None:
            extras = {"todas_caixas": self.todas_caixas, "todas_confiancas": self.todas_confiancas,
                      "todos_pontos": self.todos_pontos}
        np.savez_compressed(
            destino, tempos=self.tempos, pontos=self.pontos, caixas=self.caixas,
            confianca=self.confianca, **extras,
            meta=json.dumps({"origem": self.origem, "fps": self.fps, "largura": self.largura,
                             "altura": self.altura, "objetos": self.objetos,
                             "recorte": self.recorte}),
        )
        return destino

    @classmethod
    def carregar(cls, origem: Path) -> SequenciaPose:
        with np.load(origem, allow_pickle=False) as dados:
            meta = json.loads(str(dados["meta"]))
            extras = {chave: dados[chave] for chave in
                      ("todas_caixas", "todas_confiancas", "todos_pontos") if chave in dados}
            return cls(origem=meta["origem"], fps=meta["fps"], largura=meta["largura"],
                       altura=meta["altura"], tempos=dados["tempos"], pontos=dados["pontos"],
                       caixas=dados["caixas"], confianca=dados["confianca"],
                       objetos=meta["objetos"], recorte=meta.get("recorte"), **extras)


def recorte_padrao(caminho: Path, largura: int, altura: int) -> str | None:
    """Os MP4 do UR Fall trazem profundidade (esquerda) e RGB (direita) lado a lado."""
    if caminho.name.endswith("-cam0.mp4") and largura == 2 * int(altura * 4 / 3):
        return "metade_direita"
    return None


def _aplicar_recorte(quadro: np.ndarray, recorte: str | None) -> np.ndarray:
    if recorte == "metade_direita":
        return quadro[:, quadro.shape[1] // 2:]
    return quadro


def ler_quadros(caminho: Path, recorte: str | None = "auto"):
    """Gera (indice, tempo_s, quadro_bgr) ja recortado. Devolve tambem fps e dimensoes."""
    import cv2

    captura = cv2.VideoCapture(str(caminho))
    if not captura.isOpened():
        raise ValueError(f"Nao foi possivel abrir o video: {caminho}")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if recorte == "auto":
        recorte = recorte_padrao(caminho, largura, altura)

    def quadros():
        indice = 0
        try:
            while True:
                ok, quadro = captura.read()
                if not ok:
                    break
                yield indice, indice / fps, _aplicar_recorte(quadro, recorte)
                indice += 1
        finally:
            captura.release()

    if recorte == "metade_direita":
        largura //= 2
    return quadros(), fps, largura, altura, recorte


def _dispositivo() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


PESSOAS_POR_QUADRO = 3


def rastrear(caixas: np.ndarray, confiancas: np.ndarray, memoria: int = 15) -> np.ndarray:
    """Escolhe, quadro a quadro, a deteccao da pessoa principal.

    Associa as deteccoes em trilhas (vizinho mais proximo, em alturas de corpo) e
    elege a trilha que mais se desloca. Objetos parados confundidos com pessoa
    (cadeira com casaco, cabide) quase nao se movem; o paciente, sim. Depois de
    um corte de cena a pessoa vira outra trilha: trilhas com movimento que nao
    se sobrepoem no tempo a principal sao encadeadas a ela.
    Devolve o indice da deteccao escolhida em cada quadro (-1 sem pessoa).
    """
    quadros = len(caixas)
    trilhas: list[dict] = []
    for i in range(quadros):
        ordem = [k for k in np.argsort(-confiancas[i]) if confiancas[i, k] > 0]
        usadas: set[int] = set()
        for k in ordem:
            caixa = caixas[i, k]
            centro = (caixa[:2] + caixa[2:]) / 2
            altura = max(caixa[3] - caixa[1], 1.0)
            melhor, distancia_melhor = None, 1.0
            for indice, trilha in enumerate(trilhas):
                if indice in usadas or i - trilha["ultimo"] > memoria:
                    continue
                distancia = np.linalg.norm(centro - trilha["centro"]) / altura
                if distancia < distancia_melhor:
                    melhor, distancia_melhor = indice, distancia
            if melhor is None:
                trilhas.append({"membros": {}, "centro": centro, "ultimo": i, "caminho": 0.0})
                melhor = len(trilhas) - 1
            trilha = trilhas[melhor]
            trilha["caminho"] += float(np.linalg.norm(centro - trilha["centro"]) / altura)
            trilha["centro"], trilha["ultimo"] = centro, i
            trilha["membros"][i] = k
            usadas.add(melhor)

    escolha = np.full(quadros, -1)
    ordenadas = sorted(trilhas, key=lambda t: (t["caminho"], len(t["membros"])), reverse=True)
    ocupados: set[int] = set()
    for posicao, trilha in enumerate(ordenadas):
        quadros_trilha = set(trilha["membros"])
        if posicao and (len(quadros_trilha) < max(0.03 * quadros, 3) or trilha["caminho"] <= 0.5
                        or len(quadros_trilha & ocupados) > 0.05 * len(quadros_trilha)):
            continue
        for quadro, k in trilha["membros"].items():
            if quadro not in ocupados:
                escolha[quadro] = k
        ocupados |= quadros_trilha
    return escolha


def aplicar_rastreamento(sequencia: SequenciaPose) -> SequenciaPose:
    """Recalcula a pessoa principal a partir de todas as deteccoes guardadas."""
    if sequencia.todas_caixas is None:
        return sequencia
    escolha = rastrear(sequencia.todas_caixas, sequencia.todas_confiancas)
    linhas = np.arange(len(escolha))
    validos = escolha >= 0
    pontos = np.full((len(escolha), 17, 3), np.nan)
    caixas = np.full((len(escolha), 4), np.nan)
    confianca = np.zeros(len(escolha))
    pontos[validos] = sequencia.todos_pontos[linhas[validos], escolha[validos]]
    caixas[validos] = sequencia.todas_caixas[linhas[validos], escolha[validos]]
    confianca[validos] = sequencia.todas_confiancas[linhas[validos], escolha[validos]]
    sequencia.pontos, sequencia.caixas, sequencia.confianca = pontos, caixas, confianca
    return sequencia


def extrair(caminho: str | Path, recorte: str | None = "auto", passo_objetos: int = 5,
            usar_cache: bool = True, progresso: Callable[[int], None] | None = None,
            pasta_cache: Path | None = None) -> SequenciaPose:
    caminho = Path(caminho)
    cache = Path(pasta_cache or PASTA_CACHE) / f"{caminho.stem}.npz"
    if usar_cache and cache.exists():
        return SequenciaPose.carregar(cache)

    from ultralytics import YOLO

    modelos = CAMINHOS.modelos
    pose = YOLO(str(modelos / MODELO_POSE))
    detector = YOLO(str(modelos / MODELO_OBJETOS))
    dispositivo = _dispositivo()
    quadros, fps, largura, altura, recorte = ler_quadros(caminho, recorte)

    k = PESSOAS_POR_QUADRO
    tempos, pontos, caixas, confiancas, objetos = [], [], [], [], []
    for indice, tempo, quadro in quadros:
        resultado = pose.predict(quadro, verbose=False, device=dispositivo, conf=0.3)[0]
        tempos.append(tempo)
        pontos_q, caixas_q, conf_q = np.full((k, 17, 3), np.nan), np.full((k, 4), np.nan), np.zeros(k)
        if resultado.keypoints is not None and len(resultado.boxes):
            ordem = np.argsort(-resultado.boxes.conf.cpu().numpy())[:k]
            pontos_q[:len(ordem)] = resultado.keypoints.data.cpu().numpy()[ordem]
            caixas_q[:len(ordem)] = resultado.boxes.xyxy.cpu().numpy()[ordem]
            conf_q[:len(ordem)] = resultado.boxes.conf.cpu().numpy()[ordem]
        pontos.append(pontos_q)
        caixas.append(caixas_q)
        confiancas.append(conf_q)
        if indice % passo_objetos == 0:
            deteccao = detector.predict(quadro, verbose=False, device=dispositivo, conf=0.35)[0]
            objetos.append({"quadro": indice, "objetos": [
                {"classe": detector.names[int(c)], "confianca": round(float(p), 3),
                 "caixa": [round(float(v), 1) for v in caixa]}
                for c, p, caixa in zip(deteccao.boxes.cls.cpu().numpy(),
                                       deteccao.boxes.conf.cpu().numpy(),
                                       deteccao.boxes.xyxy.cpu().numpy(), strict=True)
            ]})
        if progresso:
            progresso(indice)

    todos_pontos = np.array(pontos).reshape(-1, k, 17, 3)
    todas_caixas = np.array(caixas).reshape(-1, k, 4)
    todas_confiancas = np.array(confiancas).reshape(-1, k)
    sequencia = aplicar_rastreamento(SequenciaPose(
        origem=str(caminho), fps=float(fps), largura=int(largura), altura=int(altura),
        tempos=np.array(tempos), pontos=todos_pontos[:, 0], caixas=todas_caixas[:, 0],
        confianca=todas_confiancas[:, 0], objetos=objetos, recorte=recorte,
        todas_caixas=todas_caixas, todas_confiancas=todas_confiancas, todos_pontos=todos_pontos,
    ))
    if usar_cache:
        sequencia.salvar(cache)
    return sequencia
