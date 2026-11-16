"""Subcomando ``video`` do CLI: analisar um video, analisar os exemplos, avaliar."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ..config import CAMINHOS
from ..dados.banco import preparar_banco
from .alertas import CentralAlertas


def _analisar(args: argparse.Namespace) -> int:
    from .video.analise import PASTA_RELATORIOS, analisar_video

    caminho = Path(args.arquivo)
    if not caminho.exists():
        print(f"Arquivo nao encontrado: {caminho}")
        return 1
    banco = preparar_banco()
    prontuario = None
    if args.prontuario:
        encontrados = banco.buscar_paciente(args.prontuario)
        prontuario = encontrados[0]["prontuario"] if encontrados else args.prontuario
    central = None if args.sem_alertas else CentralAlertas(banco)
    print(f"Analisando {caminho.name} (YOLOv8-pose + YOLOv8)...")
    resultado = analisar_video(caminho, tipo=args.tipo, prontuario=prontuario,
                               config_areas=args.areas, banco=banco, central=central,
                               pasta_saida=PASTA_RELATORIOS, anotar=args.anotar)
    print(resultado.relatorio())
    print(f"Relatorio em {PASTA_RELATORIOS / (caminho.stem + '.md')}")
    return 0


EXEMPLOS = (
    ("urfall_video/fall-01-cam0.mp4", "875431", "urfall_cam0.json"),
    ("urfall_video/adl-26-cam0.mp4", "875431", "urfall_cam0.json"),
    ("exercicios/agachamento_cdc.webm", "117256", None),
    ("exercicios/afundo_alternado.webm", "117256", None),
    ("exercicios/agachamento_unipodal.webm", "117256", None),
    ("exercicios/escada_de_dedos.ogv", "117256", None),
)


def _exemplos(args: argparse.Namespace) -> int:
    from .video.analise import PASTA_RELATORIOS, analisar_video

    banco = preparar_banco()
    central = CentralAlertas(banco)
    configuracoes = Path(__file__).parent / "video" / "configuracoes"
    for relativo, prontuario, areas in EXEMPLOS:
        caminho = CAMINHOS.multimodal / relativo
        if not caminho.exists():
            print(f"{relativo}: ausente (rode scripts/baixar_datasets.py urfall_video exercicios)")
            continue
        tipo = "exercicio" if relativo.startswith("exercicios/") else "quarto"
        r = analisar_video(caminho, tipo=tipo, prontuario=prontuario,
                           config_areas=configuracoes / areas if areas else None, banco=banco,
                           central=central, pasta_saida=PASTA_RELATORIOS, anotar=args.anotar)
        resumo = r.exercicio.resumo if r.exercicio else {}
        print(f"{caminho.name:28s} {r.tipo:9s} {r.duracao_s:5.1f}s  "
              + (f"{resumo.get('repeticoes')} rep, {resumo.get('articulacao')}, amplitude "
                 f"{resumo.get('amplitude_media_graus') or resumo.get('amplitude_maxima_graus')} graus"
                 if resumo else
                 ", ".join(e.tipo for e in r.eventos_queda) or "sem eventos"))
        for anomalia in r.anomalias:
            print(f"{'':30s}[{anomalia.severidade.upper():8s}] {anomalia.descricao}")
    print(f"\nRelatorios em {PASTA_RELATORIOS}")
    return 0


def _avaliar(args: argparse.Namespace) -> int:
    from .video import avaliacao

    sequencias = avaliacao.sequencias_em_cache()
    if not sequencias:
        print("Sem poses em cache: rode 'video extrair' antes.")
        return 1
    resultado = avaliacao.avaliar(sequencias=sequencias)
    resultado["gerado_em"] = datetime.now().isoformat(timespec="seconds")
    linhas = ["# Avaliacao da deteccao de queda por video (UR Fall, RGB)", "",
              f"Gerado em {resultado['gerado_em']}. Modelo: YOLOv8s-pose; limiares calibrados "
              "nas sequencias impares.", "",
              "| Particao | Quedas | Atividades | Sensibilidade | Especificidade | "
              "Atividades que deitam sem alarme | Latencia mediana | Erros |",
              "|---|---|---|---|---|---|---|---|"]
    for nome, d in resultado["particoes"].items():
        latencia = f"{d['latencia_mediana_s']:.2f} s" if d["latencia_mediana_s"] is not None else "-"
        linhas.append(f"| {nome} | {d['quedas']} | {d['atividades']} | {d['sensibilidade']:.0%} | "
                      f"{d['especificidade']:.0%} | {d['atividades_deitando_sem_alarme']} | "
                      f"{latencia} | {', '.join(d['erros']) or '-'} |")
    markdown = "\n".join(linhas) + "\n"
    CAMINHOS.avaliacao.mkdir(parents=True, exist_ok=True)
    (CAMINHOS.avaliacao / "video.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (CAMINHOS.avaliacao / "video.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def _extrair(args: argparse.Namespace) -> int:
    from .video.extracao import extrair

    pastas = [CAMINHOS.multimodal / "urfall_video", CAMINHOS.multimodal / "exercicios"]
    arquivos = [a for pasta in pastas for a in sorted(pasta.glob("*"))
                if a.suffix in {".mp4", ".webm", ".ogv"}]
    for indice, arquivo in enumerate(arquivos, start=1):
        sequencia = extrair(arquivo)
        print(f"{indice:3d}/{len(arquivos)} {arquivo.name:30s} {sequencia.quadros:5d} quadros")
    return 0


def registrar(subparsers) -> None:
    video = subparsers.add_parser("video", help="Analise de video: postura, queda, exercicio.")
    acoes = video.add_subparsers(dest="acao", required=True)

    analisar = acoes.add_parser("analisar", help="Analisa um video e gera o relatorio.")
    analisar.add_argument("arquivo")
    analisar.add_argument("--tipo", default="auto", choices=["auto", "quarto", "exercicio"])
    analisar.add_argument("--prontuario", default=None, help="Prontuario, leito ou nome.")
    analisar.add_argument("--areas", default=None, help="JSON com areas criticas da camera.")
    analisar.add_argument("--anotar", action="store_true", help="Gera tambem o video anotado.")
    analisar.add_argument("--sem-alertas", action="store_true")
    analisar.set_defaults(funcao=_analisar)

    exemplos = acoes.add_parser("exemplos", help="Analisa os videos de exemplo.")
    exemplos.add_argument("--anotar", action="store_true")
    exemplos.set_defaults(funcao=_exemplos)

    extrair = acoes.add_parser("extrair", help="Extrai e guarda em cache as poses dos videos.")
    extrair.set_defaults(funcao=_extrair)

    avaliar = acoes.add_parser("avaliar", help="Avalia a deteccao de queda no UR Fall.")
    avaliar.set_defaults(funcao=_avaliar)
