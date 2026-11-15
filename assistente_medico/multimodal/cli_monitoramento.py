"""Subcomando ``monitorar`` do CLI: gerar dados, avaliar detectores e simular o fluxo."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta

from ..config import CAMINHOS
from ..dados.banco import preparar_banco
from .alertas import CentralAlertas
from .anomalias import (
    DetectorMovimentacao,
    DetectorPrescricoes,
    DetectorSinaisVitais,
    ParametrosSinaisVitais,
    avaliacao,
    sintetico,
)
from .anomalias.fontes import ler_bidmc, ler_urfall, registros_bidmc, registros_urfall
from .tempo_real import Leitura, Monitor

CORES = {"maxima": "\033[1;31m", "alta": "\033[31m", "media": "\033[33m", "informativa": "\033[36m"}
NEUTRO = "\033[0m"


def _gerar(args: argparse.Namespace) -> int:
    banco = preparar_banco()
    dados = sintetico.preparar_monitoramento(banco, forcar=True)
    banco.limpar_monitoramento()
    print(f"Series geradas em {sintetico.PASTA_PADRAO}")
    for nome, total in dados.get("carregados", {}).items():
        print(f"  {nome:14s} {total:5d} registros")
    print(f"  rotulos        {len(dados['rotulos']):5d} eventos")
    return 0


def _avaliar(args: argparse.Namespace) -> int:
    banco = preparar_banco()
    dados = sintetico.preparar_monitoramento(banco)
    alergias = {p["prontuario"]: banco.alergias(p["prontuario"]) for p in banco.listar_pacientes()}
    resultado: dict = {"gerado_em": datetime.now().isoformat(timespec="seconds")}

    print("Internacao sintetica (sinais vitais horarios, prescricoes, sensor de leito)...")
    resultado["internacao"] = avaliacao.avaliar_internacao(dados, alergias)
    if registros_urfall():
        print("Quedas (UR Fall)...")
        resultado["quedas"] = avaliacao.avaliar_quedas()
    else:
        print("UR Fall ausente: rode scripts/baixar_datasets.py urfall")
    if registros_bidmc() and not args.rapido:
        print("Sinais vitais com anomalias injetadas (BIDMC; ~3 min)...")
        resultado["sinais_vitais_bidmc"] = avaliacao.avaliar_sinais_vitais(progresso=lambda n: print(f"  {n}"))
    elif not registros_bidmc():
        print("BIDMC ausente: rode scripts/baixar_datasets.py bidmc")

    CAMINHOS.avaliacao.mkdir(parents=True, exist_ok=True)
    destino = CAMINHOS.avaliacao / "anomalias.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(resultado)
    (CAMINHOS.avaliacao / "anomalias.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"Resultados em {destino}")
    return 0


def _markdown(resultado: dict) -> str:
    partes = ["# Avaliacao dos detectores de anomalia\n", f"Gerado em {resultado['gerado_em']}.\n"]
    if "sinais_vitais_bidmc" in resultado:
        bidmc = resultado["sinais_vitais_bidmc"]
        partes += [
            "## Sinais vitais: PhysioNet BIDMC com anomalias injetadas\n",
            f"{bidmc['registros']} registros reais, {bidmc['horas_monitoradas']} h a 1 Hz. "
            "Cada registro e reproduzido limpo e com cada evento injetado; entre parenteses, "
            "a latencia mediana de deteccao.\n",
            avaliacao.tabela_markdown(bidmc), "",
        ]
    if "quedas" in resultado:
        partes += ["## Quedas: UR Fall Detection (acelerometro)\n",
                   "| Particao | Quedas | Atividades | Sensibilidade | Especificidade | Erros |",
                   "|---|---|---|---|---|---|"]
        for nome, d in resultado["quedas"]["particoes"].items():
            partes.append(f"| {nome} | {d['quedas']} | {d['atividades']} | {d['sensibilidade']:.0%} "
                          f"| {d['especificidade']:.0%} | {', '.join(d['erros']) or '-'} |")
        partes.append("")
    internacao = resultado["internacao"]
    partes += ["## Internacao sintetica (eventos rotulados)\n",
               "| Modalidade | Eventos | Detectados | Anomalias fora de rotulo |", "|---|---|---|---|"]
    for modalidade, d in internacao["por_modalidade"].items():
        partes.append(f"| {modalidade} | {d['eventos_rotulados']} | {d['detectados']} "
                      f"| {d['anomalias_fora_de_rotulo']} |")
    partes += ["", "| Paciente | Evento | Detectado | Tipos |", "|---|---|---|---|"]
    for e in internacao["eventos"]:
        partes.append(f"| {e['prontuario']} | {e['evento']} | {'sim' if e['detectado'] else 'nao'} "
                      f"| {', '.join(e['tipos_detectados'])} |")
    return "\n".join(partes) + "\n"


def _leituras_queda(dados: dict, prontuario: str) -> list[Leitura]:
    """Encaixa uma queda real do UR Fall na saida noturna do paciente, para a demonstracao."""
    saidas = [r for r in dados["rotulos"]
              if r["prontuario"] == prontuario and r["evento"] == "saida_noturna"]
    arquivos = [c for c in registros_urfall() if c.name.startswith("fall-01")]
    if not saidas or not arquivos:
        return []
    momento = datetime.fromisoformat(saidas[0]["inicio"]) + timedelta(minutes=7)
    return ler_urfall(arquivos[0], prontuario=prontuario, inicio=momento)


def _imprimir(leitura: Leitura, resultados: list, leitos: dict[str, str]) -> None:
    for r in resultados:
        if not r.emitido:
            continue
        a = r.anomalia
        cor = CORES.get(a.prioridade, "")
        print(f"{cor}{leitura.instante:%d/%m %H:%M:%S}  ALERTA #{r.alerta_id:<4} "
              f"[{a.prioridade.upper():8s}] {leitos.get(a.prontuario, a.prontuario):8s} "
              f"{a.modalidade:13s} {a.descricao}{NEUTRO}")


def _simular(args: argparse.Namespace) -> int:
    banco = preparar_banco()
    dados = sintetico.preparar_monitoramento(banco)
    pacientes = banco.listar_pacientes()
    leitos = {p["prontuario"]: p["leito"] for p in pacientes}
    alvo = None
    if args.prontuario:
        encontrados = banco.buscar_paciente(args.prontuario)
        if not encontrados:
            print(f"Paciente nao encontrado: {args.prontuario}")
            return 1
        alvo = encontrados[0]["prontuario"]

    if not args.manter_historico:
        banco.limpar_monitoramento()
    central = CentralAlertas(banco)
    analise = banco.abrir_analise("monitoramento", prontuario=alvo,
                                  origem="bidmc" if args.bidmc else "internacao")

    if args.bidmc:
        caminhos = [c for c in registros_bidmc() if c.stem.startswith(f"bidmc_{args.bidmc:02d}")]
        if not caminhos:
            print("Registro BIDMC nao encontrado: rode scripts/baixar_datasets.py bidmc")
            return 1
        alvo = alvo or "559120"
        leituras = ler_bidmc(caminhos[0], prontuario=alvo, inicio=datetime.now().replace(microsecond=0))
        if args.injetar:
            leituras, injecao = avaliacao.injetar(leituras, args.injetar, random.Random(args.semente))
            print(f"Evento '{args.injetar}' injetado as {injecao.inicio:%H:%M:%S}.")
        detectores = [DetectorSinaisVitais(ParametrosSinaisVitais.alta_frequencia())]
        aquecimento: list[Leitura] = []
    else:
        leituras = avaliacao.leituras_internacao(dados)
        if args.com_queda:
            leituras = sorted(leituras + _leituras_queda(dados, "875431"), key=lambda item: item.instante)
        if alvo:
            leituras = [leitura for leitura in leituras if leitura.prontuario == alvo]
        corte = leituras[-1].instante - timedelta(hours=args.horas) if leituras else None
        aquecimento = [leitura for leitura in leituras if corte and leitura.instante < corte]
        leituras = [leitura for leitura in leituras if not corte or leitura.instante >= corte]
        detectores = [
            DetectorSinaisVitais(ParametrosSinaisVitais.horario()),
            DetectorPrescricoes.do_banco(banco),
            DetectorMovimentacao(risco_queda=dados["risco_queda"]),
        ]

    for leitura in aquecimento:
        for detector in detectores:
            if leitura.modalidade in detector.modalidades:
                detector.atualizar(leitura)

    print(f"Monitorando {len(leituras)} leituras"
          + (f" de {leitos.get(alvo, alvo)}" if alvo else f" de {len(pacientes)} pacientes")
          + (f", aceleracao {args.aceleracao:g}x" if args.aceleracao else "") + "\n")
    monitor = Monitor(central, detectores, analise_id=analise)
    resumo = monitor.reproduzir(leituras, aceleracao=args.aceleracao or None,
                                ao_processar=lambda leitura, r: _imprimir(leitura, r, leitos))
    banco.concluir_analise(analise, resumo.como_dicionario())
    print(f"\n{resumo.leituras} leituras | {resumo.anomalias} anomalias | "
          f"{len(resumo.alertas)} alertas emitidos | {resumo.suprimidos} repeticoes suprimidas | "
          f"{resumo.abaixo_do_limiar} abaixo do limiar")
    print("Reconheca um alerta com: python -m assistente_medico.cli alertas --reconhecer ID "
          "--profissional 'Nome'")
    return 0


def registrar(subparsers) -> None:
    monitorar = subparsers.add_parser("monitorar", help="Monitoramento multimodal (Fase 4).")
    acoes = monitorar.add_subparsers(dest="acao", required=True)

    gerar = acoes.add_parser("gerar", help="Gera as series de internacao e carrega no banco.")
    gerar.set_defaults(funcao=_gerar)

    avaliar = acoes.add_parser("avaliar", help="Avalia os detectores e salva os resultados.")
    avaliar.add_argument("--rapido", action="store_true", help="Pula a avaliacao no BIDMC.")
    avaliar.set_defaults(funcao=_avaliar)

    simular = acoes.add_parser("simular", help="Reproduz o fluxo de leituras e emite alertas.")
    simular.add_argument("--prontuario", default=None, help="Prontuario, leito ou nome.")
    simular.add_argument("--horas", type=float, default=24.0,
                         help="Quantas horas finais reproduzir (o resto aquece os detectores).")
    simular.add_argument("--aceleracao", type=float, default=0.0,
                         help="0 = sem espera; 3600 = uma hora por segundo.")
    simular.add_argument("--sem-queda", dest="com_queda", action="store_false",
                         help="Nao encaixa a queda do UR Fall na madrugada do ENF-221.")
    simular.add_argument("--bidmc", type=int, default=None, metavar="N",
                         help="Usa o registro BIDMC N (1 Hz) no lugar da internacao sintetica.")
    simular.add_argument("--injetar", default=None, choices=avaliacao.EVENTOS,
                         help="Com --bidmc: injeta um evento no registro.")
    simular.add_argument("--semente", type=int, default=7)
    simular.add_argument("--manter-historico", action="store_true",
                         help="Nao apaga anomalias e alertas de simulacoes anteriores.")
    simular.set_defaults(funcao=_simular)
