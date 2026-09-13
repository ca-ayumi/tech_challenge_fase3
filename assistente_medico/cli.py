"""Interface de linha de comando do assistente medico.

Subcomandos:

    preparar     Gera prontuarios, dataset e banco a partir dos dados brutos.
    perguntar    Faz uma pergunta ao assistente, pelo fluxo completo.
    paciente     Mostra o contexto e as pendencias de um paciente.
    alertas      Lista os alertas abertos no painel institucional.
    validacoes   Lista e decide os pedidos de validacao humana pendentes.
    auditoria    Mostra a trilha de uma interacao.
    politicas    Lista as politicas de seguranca vigentes.
    diagrama     Imprime o diagrama Mermaid do fluxo LangGraph.

Exemplo:
    python -m assistente_medico.cli perguntar "Ha pendencia no leito PS-07?" --perfil medico
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .auditoria import Auditoria, configurar_logging
from .config import CAMINHOS, MODELO, NOME_ASSISTENTE, VERSAO_ASSISTENTE
from .dados.banco import preparar_banco
from .dados.contexto_paciente import montar_contexto
from .seguranca.politicas import resumo_politicas
from .seguranca.regras_clinicas import avaliar


def _assistente(backend: str | None = None):
    from .grafo import AssistenteMedico, Dependencias
    from .llm import criar_modelo_chat

    return AssistenteMedico(Dependencias(
        banco=preparar_banco(), modelo=criar_modelo_chat(backend or MODELO.backend)
    ))


def comando_preparar(args: argparse.Namespace) -> int:
    from .dados import construir_dataset

    CAMINHOS.garantir()
    print("1/3 Gerando prontuarios sinteticos...")
    caminho_prontuarios = CAMINHOS.dados_brutos / "prontuarios.jsonl"
    if args.forcar or not caminho_prontuarios.exists():
        import subprocess

        subprocess.run(
            [sys.executable, str(CAMINHOS.raiz / "scripts" / "gerar_prontuarios_sinteticos.py")],
            check=True,
        )
    else:
        print(f"    ja existe: {caminho_prontuarios}")

    print("2/3 Populando o banco do hospital...")
    banco = preparar_banco(forcar=args.forcar)
    print(f"    {len(banco.listar_pacientes())} pacientes em {banco.caminho}")

    print("3/3 Construindo o dataset de fine-tuning...")
    sys.argv = ["construir_dataset"]
    construir_dataset.main()
    return 0


def comando_perguntar(args: argparse.Namespace) -> int:
    assistente = _assistente(args.backend)
    estado = assistente.responder(
        args.pergunta, perfil=args.perfil, usuario=args.usuario, prontuario=args.prontuario
    )

    print()
    print(estado["resposta"])
    print()
    print("-" * 72)
    print(f"interacao ....... {estado['interacao']}")
    print(f"categoria ....... {estado['categoria']} (intencao: {estado['intencao']})")
    if estado.get("prontuario"):
        print(f"paciente ........ prontuario {estado['prontuario']}")
    print(f"etapas .......... {' > '.join(estado['etapas'])}")
    if estado.get("violacoes"):
        print(f"politicas ....... {', '.join(v['codigo'] for v in estado['violacoes'])}")
    if estado.get("alertas"):
        print(f"alertas ......... {len(estado['alertas'])} emitidos "
              f"(ids {estado.get('alertas_registrados')})")
    if estado.get("validacao_id"):
        print(f"validacao ....... pedido {estado['validacao_id']} pendente de decisao humana")
    explicacao = estado.get("explicacao") or {}
    if explicacao.get("cobertura_das_citacoes") is not None:
        print(f"cobertura ....... {explicacao['cobertura_das_citacoes']:.0%} das citacoes "
              "fundamentadas")
    if explicacao.get("citacoes_nao_fundamentadas"):
        print(f"ATENCAO ......... citacoes nao fundamentadas: "
              f"{', '.join(explicacao['citacoes_nao_fundamentadas'])}")

    if args.explicar:
        from .explicabilidade import Explicacao, FonteUtilizada

        print()
        print(Explicacao(
            fontes=[FonteUtilizada(**f) for f in explicacao.get("fontes", [])],
            campos_prontuario=explicacao.get("campos_prontuario", []),
            regras_aplicadas=[r for r in explicacao.get("regras_aplicadas", []) if r],
            citadas=explicacao.get("referencias_citadas", []),
            recuperadas=explicacao.get("referencias_recuperadas", []),
            nao_fundamentadas=explicacao.get("citacoes_nao_fundamentadas", []),
        ).como_markdown())

    if args.trilha:
        print()
        print("Trilha de auditoria:")
        for evento in assistente.trilha(estado["interacao"]):
            print(f"  {evento['registrado_em']}  {evento['evento']}")
    return 0


def comando_paciente(args: argparse.Namespace) -> int:
    banco = preparar_banco()
    encontrados = banco.buscar_paciente(args.termo)
    if not encontrados:
        print(f"Nenhum paciente para '{args.termo}'.")
        return 1
    prontuario = encontrados[0]["prontuario"]
    contexto = montar_contexto(banco, prontuario)
    if contexto is None:
        return 1

    print(f"{encontrados[0]['nome']} — prontuario {prontuario}")
    print("-" * 72)
    print(contexto.como_texto())
    alertas = avaliar(contexto)
    print()
    if alertas:
        print(f"Pendencias ({len(alertas)}):")
        for alerta in alertas:
            print(f"  {alerta.como_linha()}")
    else:
        print("Nenhuma pendencia identificada pelas regras vigentes.")
    return 0


def comando_alertas(args: argparse.Namespace) -> int:
    banco = preparar_banco()
    alertas = banco.listar_alertas(prontuario=args.prontuario, limite=args.limite)
    if not alertas:
        print("Nenhum alerta aberto.")
        return 0
    for alerta in alertas:
        print(f"[{alerta['prioridade'].upper():9s}] #{alerta['id']:<4} "
              f"prontuario {alerta['prontuario'] or '-':8s} {alerta['mensagem'][:90]}")
        print(f"{'':13s}fonte: {alerta['fonte']} | {alerta['criado_em']}")
    return 0


def comando_validacoes(args: argparse.Namespace) -> int:
    banco = preparar_banco()
    if args.decidir:
        banco.decidir_validacao(
            validacao_id=args.decidir, status=args.status,
            profissional=args.profissional, justificativa=args.justificativa,
        )
        Auditoria(banco=banco).registrar(
            f"VALIDACAO-{args.decidir}", "decisao_humana",
            {"validacao_id": args.decidir, "status": args.status,
             "justificativa": args.justificativa},
            usuario=args.profissional,
        )
        print(f"Validacao {args.decidir} registrada como '{args.status}' "
              f"por {args.profissional}.")
        return 0

    pendentes = banco.listar_validacoes(status=args.status_filtro, limite=args.limite)
    if not pendentes:
        print(f"Nenhuma validacao com status '{args.status_filtro}'.")
        return 0
    for validacao in pendentes:
        print(f"#{validacao['id']:<4} {validacao['criado_em']} "
              f"prontuario {validacao['prontuario'] or '-'} "
              f"[{validacao['status']}]")
        print(f"      interacao: {validacao['interacao']}")
        print(f"      {validacao['sugestao'][:150].replace(chr(10), ' ')}...")
    print()
    print("Para decidir: python -m assistente_medico.cli validacoes --decidir <id> "
          "--status aceita --profissional 'Dra. Fulana (CRM-SP 00000)'")
    return 0


def comando_auditoria(args: argparse.Namespace) -> int:
    auditoria = Auditoria(banco=preparar_banco())
    if args.interacao:
        eventos = auditoria.trilha(args.interacao)
        if not eventos:
            print(f"Nenhum evento para a interacao {args.interacao}.")
            return 1
        for evento in eventos:
            print(f"{evento['registrado_em']}  {evento['evento']}")
            if args.detalhado:
                print(json.dumps(evento["conteudo"], ensure_ascii=False, indent=4)[:1500])
        return 0

    for interacao in auditoria.ultimas_interacoes(limite=args.limite):
        print(f"{interacao['registrado_em']}  {interacao['interacao']}  "
              f"perfil={interacao['perfil'] or '-':10s} "
              f"prontuario={interacao['prontuario'] or '-':8s} "
              f"eventos={interacao['eventos']}")
    return 0


def comando_politicas(_: argparse.Namespace) -> int:
    print(f"{'CODIGO':24s} {'MOMENTO':8s} {'ACAO':12s} {'SEVERIDADE':11s} FONTE")
    print("-" * 96)
    for politica in resumo_politicas():
        print(f"{politica['codigo']:24s} {politica['momento']:8s} {politica['acao']:12s} "
              f"{politica['severidade']:11s} {politica['fonte']}")
        print(f"{'':24s} {politica['descricao']}")
    return 0


def comando_diagrama(args: argparse.Namespace) -> int:

    assistente = _assistente("eco")
    diagrama = assistente.diagrama_mermaid()
    if args.saida:
        Path(args.saida).write_text(diagrama, encoding="utf-8")
        print(f"Diagrama gravado em {args.saida}")
    else:
        print(diagrama)
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assistente-medico",
        description=f"{NOME_ASSISTENTE} v{VERSAO_ASSISTENTE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="comando", required=True)

    preparar = subparsers.add_parser("preparar", help="Gera dados, banco e dataset.")
    preparar.add_argument("--forcar", action="store_true", help="Recria tudo do zero.")
    preparar.set_defaults(funcao=comando_preparar)

    perguntar = subparsers.add_parser("perguntar", help="Pergunta ao assistente.")
    perguntar.add_argument("pergunta")
    perguntar.add_argument("--perfil", default="medico")
    perguntar.add_argument("--usuario", default="nao identificado")
    perguntar.add_argument("--prontuario", default=None)
    perguntar.add_argument("--backend", default=None, choices=["mlx", "transformers", "eco"])
    perguntar.add_argument("--explicar", action="store_true", help="Mostra a base da resposta.")
    perguntar.add_argument("--trilha", action="store_true", help="Mostra a trilha de auditoria.")
    perguntar.set_defaults(funcao=comando_perguntar)

    paciente = subparsers.add_parser("paciente", help="Contexto e pendencias de um paciente.")
    paciente.add_argument("termo", help="Prontuario, leito ou parte do nome.")
    paciente.set_defaults(funcao=comando_paciente)

    alertas = subparsers.add_parser("alertas", help="Alertas abertos.")
    alertas.add_argument("--prontuario", default=None)
    alertas.add_argument("--limite", type=int, default=30)
    alertas.set_defaults(funcao=comando_alertas)

    validacoes = subparsers.add_parser("validacoes", help="Validacoes humanas.")
    validacoes.add_argument("--status-filtro", default="pendente")
    validacoes.add_argument("--limite", type=int, default=20)
    validacoes.add_argument("--decidir", type=int, default=None)
    validacoes.add_argument("--status", default="aceita",
                            choices=["aceita", "recusada", "modificada"])
    validacoes.add_argument("--profissional", default="nao identificado")
    validacoes.add_argument("--justificativa", default=None)
    validacoes.set_defaults(funcao=comando_validacoes)

    auditoria = subparsers.add_parser("auditoria", help="Trilha de auditoria.")
    auditoria.add_argument("--interacao", default=None)
    auditoria.add_argument("--limite", type=int, default=20)
    auditoria.add_argument("--detalhado", action="store_true")
    auditoria.set_defaults(funcao=comando_auditoria)

    politicas = subparsers.add_parser("politicas", help="Politicas de seguranca vigentes.")
    politicas.set_defaults(funcao=comando_politicas)

    diagrama = subparsers.add_parser("diagrama", help="Diagrama Mermaid do fluxo.")
    diagrama.add_argument("--saida", default=None)
    diagrama.set_defaults(funcao=comando_diagrama)

    return parser


def main(argumentos: list[str] | None = None) -> int:
    configurar_logging()
    parser = construir_parser()
    args = parser.parse_args(argumentos)
    return args.funcao(args)


if __name__ == "__main__":
    raise SystemExit(main())
