"""Pipeline de analise de video e relatorio automatico de desvios.

    video -> YOLOv8-pose + YOLOv8 (cache) -> cinematica
          -> queda / permanencia no chao (sempre)
          -> exercicio: repeticoes, amplitude, simetria, compensacao (sessao de exercicio)
          -> areas criticas e objetos de risco (se houver configuracao da camera)
          => anomalias (modalidade video) -> central de alertas
          => relatorio Markdown + grafico + quadros-chave anotados (+ video anotado)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ...config import CAMINHOS
from ..tipos import Anomalia
from . import areas as mod_areas
from . import cinematica as cin
from . import exercicio as mod_exercicio
from . import quedas as mod_quedas
from .extracao import SequenciaPose, extrair

PASTA_RELATORIOS = CAMINHOS.multimodal / "relatorios_video"
ESQUELETO = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))
DESCRICOES_QUEDA = {
    "queda": ("critica", "Queda detectada no video: descida rapida ate o chao"),
    "permanencia_no_chao": ("critica", "Paciente permanece no chao apos a queda"),
}


@dataclass
class ResultadoVideo:
    arquivo: str
    prontuario: str | None
    tipo: str
    duracao_s: float
    quadros: int
    quadros_com_pessoa: int
    objetos: dict[str, int]
    eventos_queda: list[mod_quedas.EventoQueda]
    exercicio: mod_exercicio.AnaliseExercicio | None
    eventos_area: list[mod_areas.EventoArea]
    anomalias: list[Anomalia]
    arquivos_gerados: dict[str, str] = field(default_factory=dict)
    analise_id: int | None = None

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "arquivo": self.arquivo, "prontuario": self.prontuario, "tipo": self.tipo,
            "duracao_s": round(self.duracao_s, 2), "quadros": self.quadros,
            "quadros_com_pessoa": self.quadros_com_pessoa, "objetos": self.objetos,
            "eventos_queda": [e.__dict__ for e in self.eventos_queda],
            "exercicio": self.exercicio.como_dicionario() if self.exercicio else None,
            "eventos_area": [e.__dict__ for e in self.eventos_area],
            "anomalias": [a.como_dicionario() for a in self.anomalias],
            "arquivos_gerados": self.arquivos_gerados, "analise_id": self.analise_id,
        }

    def relatorio(self) -> str:
        nome = Path(self.arquivo).name
        linhas = [f"# Relatorio automatico de video - {nome}", "",
                  f"Prontuario: {self.prontuario or '-'} | tipo de analise: {self.tipo} | "
                  f"duracao: {self.duracao_s:.1f} s | pessoa visivel em "
                  f"{self.quadros_com_pessoa}/{self.quadros} quadros", "",
                  "Modelos: YOLOv8s-pose (17 pontos COCO) e YOLOv8n (objetos COCO).", ""]
        if self.objetos:
            linhas += ["Objetos no ambiente (amostras com deteccao): "
                       + ", ".join(f"{k} ({v})" for k, v in list(self.objetos.items())[:8]), ""]
        linhas += ["## Desvios e falhas encontrados", ""]
        if not self.anomalias:
            linhas.append("Nenhum desvio relevante encontrado.")
        for anomalia in self.anomalias:
            inicio = anomalia.evidencias.get("inicio_video_s")
            momento = f" (em {inicio:.1f} s)" if inicio is not None else ""
            linhas.append(f"- **[{anomalia.severidade.upper()}]** {anomalia.descricao}{momento}")
        eventos_informativos = [e for e in self.eventos_queda if e.tipo not in DESCRICOES_QUEDA]
        if eventos_informativos:
            linhas += ["", "Eventos sem alerta (registrados para contexto):", ""]
            for e in eventos_informativos:
                rotulo = {"deitou_controlado": "deitou de forma controlada",
                          "deitou_em_movel": f"deitou em movel ({e.evidencias.get('movel')})"}[e.tipo]
                linhas.append(f"- {e.inicio_s:.1f}-{e.fim_s:.1f} s: {rotulo} "
                              f"(descida {e.evidencias.get('descida_quadril')}, velocidade "
                              f"{e.evidencias.get('velocidade_pico_alturas_s')} alturas/s)")
        if self.exercicio and not self.exercicio.repeticoes:
            r = self.exercicio.resumo
            linhas += ["", "## Sessao de exercicio", "",
                       f"{r.get('observacao', 'sem repeticoes')}. Articulacao com maior movimento: "
                       f"{r.get('articulacao', '-')}; amplitude maxima alcancada: "
                       f"{r.get('amplitude_maxima_graus', '-')} graus."]
        if self.exercicio and self.exercicio.repeticoes:
            r = self.exercicio.resumo
            linhas += ["", "## Sessao de exercicio", "",
                       f"Articulacao principal: {r['articulacao']} | modo: {r['modo']} | "
                       f"{r['repeticoes']} repeticoes | amplitude media {r['amplitude_media_graus']} "
                       f"graus | cadencia {r['cadencia_por_minuto']}/min | assimetria "
                       f"{r.get('assimetria_graus', 0)} graus ({r.get('repeticoes_comparaveis', 0)} "
                       "repeticoes com os dois lados comparaveis)", "",
                       "| Rep. | Inicio (s) | Amplitude (graus) | Contralateral (graus) | Tronco max (graus) |",
                       "|---|---|---|---|---|"]
            for rep in self.exercicio.repeticoes:
                contra = f"{rep.amplitude_contralateral:.0f}" if rep.amplitude_contralateral is not None else "-"
                tronco = f"{rep.tronco_max:.0f}" if rep.tronco_max is not None else "-"
                linhas.append(f"| {rep.numero} | {rep.inicio_s:.1f} | {rep.amplitude:.0f} | "
                              f"{contra} | {tronco} |")
        if self.arquivos_gerados:
            linhas += ["", "## Arquivos", ""]
            for rotulo, caminho in self.arquivos_gerados.items():
                linhas.append(f"- {rotulo}: `{Path(caminho).name}`")
                if caminho.endswith((".png", ".jpg")):
                    linhas.append(f"\n![{rotulo}]({Path(caminho).name})\n")
        return "\n".join(linhas) + "\n"


def _anomalias(eventos_queda, exercicio, eventos_area, prontuario: str | None,
               inicio_gravacao: datetime) -> list[Anomalia]:
    def iso(segundos: float) -> str:
        return (inicio_gravacao + timedelta(seconds=segundos)).isoformat(timespec="seconds")

    saida = []
    for evento in eventos_queda:
        if evento.tipo not in DESCRICOES_QUEDA:
            continue
        severidade, descricao = DESCRICOES_QUEDA[evento.tipo]
        if evento.tipo == "permanencia_no_chao":
            descricao += f" ha {evento.fim_s - evento.inicio_s:.0f} s"
        saida.append(Anomalia(
            modalidade="video", tipo=f"video_{evento.tipo}", severidade=severidade,
            descricao=descricao, prontuario=prontuario, inicio=iso(evento.inicio_s),
            fim=iso(evento.fim_s), detector="yolov8_pose_queda",
            evidencias={**evento.evidencias, "inicio_video_s": round(evento.inicio_s, 2)}))
    if exercicio:
        for desvio in exercicio.desvios:
            saida.append(Anomalia(
                modalidade="video", tipo=f"exercicio_{desvio.tipo}", severidade=desvio.severidade,
                descricao=f"Exercicio: {desvio.descricao}", prontuario=prontuario,
                inicio=iso(0), fim=iso(exercicio.repeticoes[-1].fim_s if exercicio.repeticoes else 0),
                detector="yolov8_pose_exercicio", evidencias=desvio.evidencias))
    for evento in eventos_area:
        saida.append(Anomalia(
            modalidade="video", tipo=f"video_{evento.tipo}", severidade=evento.severidade,
            descricao=evento.descricao, prontuario=prontuario, inicio=iso(evento.inicio_s),
            fim=iso(evento.fim_s), detector="yolov8_areas",
            evidencias={**evento.evidencias, "inicio_video_s": round(evento.inicio_s, 2)}))
    return saida


def _grafico(cinematica: cin.Cinematica, resultado: ResultadoVideo, destino: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, eixo = plt.subplots(figsize=(9, 3.2), dpi=110)
    eixo.plot(cinematica.tempos, cinematica.angulo_tronco, color="#4a5568", label="tronco x vertical")
    if resultado.exercicio and resultado.exercicio.articulacao:
        nome = resultado.exercicio.articulacao
        eixo.plot(cinematica.tempos, cinematica.articulacoes[nome], color="#2b6cb0",
                  label=mod_exercicio.NOMES[nome])
        for rep in resultado.exercicio.repeticoes:
            eixo.axvline(rep.extremo_s, color="#2b6cb0", alpha=0.25, linewidth=1)
    for evento in resultado.eventos_queda:
        cor = "#c53030" if evento.tipo == "queda" else "#718096"
        eixo.axvspan(evento.inicio_s, max(evento.fim_s, evento.inicio_s + 0.1), color=cor, alpha=0.2)
    eixo.set_xlabel("tempo (s)")
    eixo.set_ylabel("graus")
    eixo.legend(loc="upper right", fontsize=8, frameon=False)
    eixo.spines[["top", "right"]].set_visible(False)
    figura.tight_layout()
    figura.savefig(destino)
    plt.close(figura)
    return destino


def _desenhar(quadro: np.ndarray, pontos: np.ndarray, texto: str, cor=(0, 200, 255)) -> np.ndarray:
    import cv2

    saida = quadro.copy()
    if pontos is not None and not np.isnan(pontos[:, 0]).all():
        for a, b in ESQUELETO:
            if pontos[a, 2] >= 0.3 and pontos[b, 2] >= 0.3:
                cv2.line(saida, tuple(int(v) for v in pontos[a, :2]),
                         tuple(int(v) for v in pontos[b, :2]), cor, 2)
        for x, y, c in pontos:
            if c >= 0.3:
                cv2.circle(saida, (int(x), int(y)), 3, (255, 255, 255), -1)
    cv2.rectangle(saida, (0, 0), (saida.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(saida, texto, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)
    return saida


def _quadros_chave(sequencia: SequenciaPose, resultado: ResultadoVideo, pasta: Path,
                   base: str) -> dict[str, str]:
    import cv2

    from .extracao import ler_quadros

    alvos: dict[int, tuple[str, tuple[int, int, int]]] = {}
    for evento in resultado.eventos_queda:
        indice = int(min(evento.fim_s * sequencia.fps, sequencia.quadros - 1))
        cor = (0, 0, 255) if evento.tipo == "queda" else (200, 200, 200)
        alvos[indice] = (f"{evento.tipo} t={evento.fim_s:.1f}s", cor)
    if resultado.exercicio and resultado.exercicio.repeticoes:
        rep = max(resultado.exercicio.repeticoes, key=lambda r: r.amplitude)
        alvos[int(rep.extremo_s * sequencia.fps)] = (
            f"rep {rep.numero}: amplitude {rep.amplitude:.0f} graus", (0, 200, 255))
    gerados = {}
    if not alvos:
        return gerados
    quadros, *_ = ler_quadros(Path(sequencia.origem), sequencia.recorte)
    for indice, _, quadro in quadros:
        if indice in alvos:
            texto, cor = alvos[indice]
            destino = pasta / f"{base}_q{indice:05d}.jpg"
            cv2.imwrite(str(destino), _desenhar(quadro, sequencia.pontos[indice], texto, cor))
            gerados[f"quadro {texto}"] = str(destino)
        if indice > max(alvos):
            break
    return gerados


def anotar_video(sequencia: SequenciaPose, resultado: ResultadoVideo, destino: Path) -> Path:
    """Video com esqueleto e estado a cada quadro, para a demonstracao."""
    import cv2

    from .extracao import ler_quadros

    quadros, fps, largura, altura, _ = ler_quadros(Path(sequencia.origem), sequencia.recorte)
    escritor = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"avc1"), fps, (largura, altura))
    if not escritor.isOpened():  # sem codificador H.264: mp4v (abre no VLC, nao no navegador)
        escritor = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                   (largura, altura))
    estado = mod_quedas.estados(cin.calcular(sequencia), mod_quedas.ParametrosQueda())
    for indice, tempo, quadro in quadros:
        alerta = next((e for e in resultado.eventos_queda
                       if e.tipo == "queda" and e.inicio_s <= tempo), None)
        texto = f"t={tempo:4.1f}s  {estado[indice] if indice < len(estado) else ''}"
        cor = (0, 200, 255)
        if alerta:
            texto += "  ALERTA: QUEDA"
            cor = (0, 0, 255)
        pontos = sequencia.pontos[indice] if indice < sequencia.quadros else None
        escritor.write(_desenhar(quadro, pontos, texto, cor))
    escritor.release()
    return destino


def analisar_video(caminho: str | Path, tipo: str = "auto", prontuario: str | None = None,
                   config_areas: str | Path | None = None, banco=None, central=None,
                   pasta_saida: Path | None = None, anotar: bool = False,
                   sequencia: SequenciaPose | None = None,
                   inicio_gravacao: datetime | None = None) -> ResultadoVideo:
    caminho = Path(caminho)
    sequencia = sequencia or extrair(caminho)
    cinematica = cin.calcular(sequencia)
    eventos_queda = mod_quedas.detectar(sequencia, cinematica)
    if tipo == "exercicio":
        eventos_queda = [e for e in eventos_queda if e.tipo in DESCRICOES_QUEDA]

    analise_exercicio = None
    if tipo in {"auto", "exercicio"}:
        candidata = mod_exercicio.analisar(cinematica)
        if tipo == "exercicio" or len(candidata.repeticoes) >= 4:
            analise_exercicio = candidata
            tipo = "exercicio"
    if tipo == "auto":
        tipo = "quarto"
    eventos_area = mod_areas.detectar(sequencia, cinematica,
                                      mod_areas.ConfigAreas.carregar(config_areas))
    if inicio_gravacao is None:
        from ..relogio import agora

        inicio_gravacao = agora(banco)
    anomalias = _anomalias(eventos_queda, analise_exercicio, eventos_area, prontuario,
                           inicio_gravacao)
    resultado = ResultadoVideo(
        arquivo=str(caminho), prontuario=prontuario, tipo=tipo,
        duracao_s=float(sequencia.tempos[-1]) if sequencia.quadros else 0.0,
        quadros=sequencia.quadros, quadros_com_pessoa=int((sequencia.confianca > 0).sum()),
        objetos=mod_areas.objetos_vistos(sequencia), eventos_queda=eventos_queda,
        exercicio=analise_exercicio, eventos_area=eventos_area, anomalias=anomalias,
    )

    if pasta_saida is not None:
        pasta_saida.mkdir(parents=True, exist_ok=True)
        base = caminho.stem
        try:
            resultado.arquivos_gerados["grafico"] = str(
                _grafico(cinematica, resultado, pasta_saida / f"{base}_grafico.png"))
        except ImportError:
            pass
        try:
            resultado.arquivos_gerados.update(_quadros_chave(sequencia, resultado, pasta_saida, base))
            if anotar:
                resultado.arquivos_gerados["video anotado"] = str(
                    anotar_video(sequencia, resultado, pasta_saida / f"{base}_anotado.mp4"))
        except (ImportError, ValueError, OSError):
            pass

    if banco is not None:
        resultado.analise_id = banco.abrir_analise("video", prontuario=prontuario, origem=caminho.name)
        if central is not None:
            central.processar(anomalias, analise_id=resultado.analise_id)
        banco.concluir_analise(resultado.analise_id, resultado.como_dicionario())
    if pasta_saida is not None:
        (pasta_saida / f"{caminho.stem}.md").write_text(resultado.relatorio(), encoding="utf-8")
        (pasta_saida / f"{caminho.stem}.json").write_text(
            json.dumps(resultado.como_dicionario(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
    return resultado
