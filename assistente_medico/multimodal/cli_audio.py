"""Subcomando ``audio`` do CLI: gerar consultas, analisar um audio, avaliar."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ..config import AZURE
from ..dados.banco import preparar_banco
from .alertas import CentralAlertas


def _gerar(args: argparse.Namespace) -> int:
    from .audio import roteiros
    from .azure.fala import sintetizar

    if not AZURE.fala_configurada:
        print("Azure Speech nao configurado: defina AZURE_SPEECH_KEY e AZURE_SPEECH_REGION no .env")
        return 1
    banco = preparar_banco()
    nomes = {p["prontuario"]: p["nome"] for p in banco.listar_pacientes()}
    print("Sintetizando consultas dos pacientes (Azure Text to Speech)...")
    consultas = roteiros.gerar_consultas(nomes, sintetizar)
    print(f"  {len(consultas)} arquivos em {roteiros.PASTA_AUDIO / 'consultas'}")
    if not args.sem_avaliacao:
        print("Sintetizando o conjunto de avaliacao (6 vozes x 4 condicoes)...")
        avaliacao = roteiros.gerar_avaliacao(sintetizar)
        print(f"  {len(avaliacao)} arquivos em {roteiros.PASTA_AUDIO / 'avaliacao'}")
    return 0


def _analisar(args: argparse.Namespace) -> int:
    from .audio.consulta import analisar_consulta, salvar_relatorio
    from .audio.roteiros import PASTA_AUDIO

    caminho = Path(args.arquivo)
    if not caminho.exists():
        print(f"Arquivo nao encontrado: {caminho}")
        return 1
    banco = preparar_banco()
    central = CentralAlertas(banco) if not args.sem_alertas else None
    resultado = analisar_consulta(caminho, prontuario=args.prontuario, basal=args.basal,
                                  banco=banco, central=central, usar_azure=not args.sem_azure)
    print(resultado.relatorio())
    destino = salvar_relatorio(resultado, PASTA_AUDIO / "relatorios")
    print(f"Relatorio salvo em {destino}")
    if central is not None:
        abertos = [a for a in banco.listar_alertas(prontuario=resultado.prontuario, limite=20)
                   if a["categoria"] in {"anomalia_audio", "anomalia_texto"}]
        print(f"{len(abertos)} alerta(s) de audio/texto abertos para o prontuario "
              f"{resultado.prontuario}.")
    return 0


def _consultas(args: argparse.Namespace) -> int:
    from .audio.consulta import analisar_consulta, salvar_relatorio
    from .audio.roteiros import PASTA_AUDIO

    banco = preparar_banco()
    central = CentralAlertas(banco)
    leitos = {p["prontuario"]: p["leito"] for p in banco.listar_pacientes()}
    arquivos = sorted((PASTA_AUDIO / "consultas").glob("*_consulta.wav"))
    if not arquivos:
        print("Nenhuma consulta encontrada: rode 'audio gerar' antes.")
        return 1
    for arquivo in arquivos:
        r = analisar_consulta(arquivo, banco=banco, central=central, usar_azure=not args.sem_azure)
        salvar_relatorio(r, PASTA_AUDIO / "relatorios")
        voz = ", ".join(i.condicao for i in r.indicadores if i.suspeita) or "sem alteracao"
        termos = ", ".join(f"{m.termo.rotulo}{' (negado)' if m.negada else ''}"
                           for m in r.mencoes) or "-"
        print(f"{leitos.get(r.prontuario, r.prontuario):8s} voz: {voz:20s} termos: {termos}")
        for anomalia in r.anomalias:
            print(f"{'':8s}  [{anomalia.severidade.upper():8s}] {anomalia.descricao}")
    return 0


def _avaliar(args: argparse.Namespace) -> int:
    from .audio import avaliacao

    resultado: dict = {"gerado_em": datetime.now().isoformat(timespec="seconds")}
    if (avaliacao.CAMINHOS.multimodal / "voiced").exists():
        print("VOICED: jitter, shimmer e HNR em vozes reais...")
        resultado["voiced"] = avaliacao.avaliar_voiced()
    else:
        print("VOICED ausente: rode scripts/baixar_datasets.py voiced")
    print("Condicoes vocais simuladas (calibracao x teste)...")
    resultado["condicoes"] = avaliacao.avaliar_condicoes()
    print("Consultas dos pacientes (pipeline completo)...")
    resultado["consultas"] = avaliacao.avaliar_consultas(preparar_banco(),
                                                         usar_azure=not args.sem_azure)
    destino = avaliacao.salvar(resultado)
    print()
    print(avaliacao.markdown(resultado))
    print(f"Resultados em {destino}")
    return 0


def registrar(subparsers) -> None:
    audio = subparsers.add_parser("audio", help="Analise de audio de consultas (Fase 4).")
    acoes = audio.add_subparsers(dest="acao", required=True)

    gerar = acoes.add_parser("gerar", help="Sintetiza as consultas de exemplo com Azure TTS.")
    gerar.add_argument("--sem-avaliacao", action="store_true",
                       help="Gera so as consultas, sem o conjunto de avaliacao.")
    gerar.set_defaults(funcao=_gerar)

    analisar = acoes.add_parser("analisar", help="Analisa um arquivo de audio.")
    analisar.add_argument("arquivo")
    analisar.add_argument("--prontuario", default=None)
    analisar.add_argument("--basal", default=None, help="Gravacao de linha de base do paciente.")
    analisar.add_argument("--sem-azure", action="store_true", help="So analise local.")
    analisar.add_argument("--sem-alertas", action="store_true", help="Nao envia a central.")
    analisar.set_defaults(funcao=_analisar)

    consultas = acoes.add_parser("consultas", help="Analisa todas as consultas de exemplo.")
    consultas.add_argument("--sem-azure", action="store_true")
    consultas.set_defaults(funcao=_consultas)

    avaliar = acoes.add_parser("avaliar", help="Avalia a analise de audio e salva os resultados.")
    avaliar.add_argument("--sem-azure", action="store_true")
    avaliar.set_defaults(funcao=_avaliar)
