"""Pipeline de construcao do dataset de fine-tuning.

Etapas, na ordem em que rodam:

1. Carregamento dos dados brutos: protocolos, modelos de documento, FAQ dos
   medicos e prontuarios sinteticos.
2. Anonimizacao de todo texto livre (PROT-GOV-010 §8).
3. Geracao de exemplos instrucao-resposta em varias categorias.
4. Curadoria: filtros de qualidade, verificacao de PII residual e deduplicacao.
5. Divisao estratificada em treino, validacao e teste.
6. Gravacao no formato de chat consumido pelo MLX-LM e pelo TRL, mais um
   relatorio com todas as contagens.

Uso:
    python -m assistente_medico.dados.construir_dataset
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import CAMINHOS, TREINO
from ..finetuning.formato import (
    PROMPT_SISTEMA,
    ROTULO_EMERGENCIA,
    ROTULO_INFORMATIVO,
    ROTULO_RECUSA,
    ROTULO_VALIDACAO,
    montar_mensagens,
    montar_resposta,
)
from ..seguranca.regras_clinicas import avaliar
from .anonimizacao import Anonimizador
from .banco import BancoHospital, carregar_prontuarios_brutos, preparar_banco
from .contexto_paciente import montar_contexto
from .curadoria import curar, dividir
from .preprocessamento import Trecho, construir_trechos, salvar_trechos

ROTULO_POR_CATEGORIA = {
    "duvida_protocolo": ROTULO_INFORMATIVO,
    "conduta_paciente": ROTULO_VALIDACAO,
    "recusa_prescricao": ROTULO_RECUSA,
    "fora_escopo": ROTULO_RECUSA,
    "emergencia": ROTULO_EMERGENCIA,
    "documento": ROTULO_VALIDACAO,
    "alerta": ROTULO_VALIDACAO,
    "governanca": ROTULO_INFORMATIVO,
    "resumo": ROTULO_VALIDACAO,
    "protocolo_secao": ROTULO_INFORMATIVO,
    "contexto_paciente": ROTULO_VALIDACAO,
}

MODELOS_PERGUNTA_SECAO = [
    "O que o {documento} define sobre {secao}?",
    "Qual a orientacao institucional sobre {secao}?",
    "Resuma a secao de {secao} do {documento}.",
    "Onde encontro no protocolo a regra de {secao}? Explique o conteudo.",
]

FARMACOS = [
    "noradrenalina", "ceftriaxona", "enoxaparina", "insulina regular", "adrenalina",
    "vancomicina", "meropenem", "alteplase", "morfina", "furosemida",
]
CONDICOES = [
    "sepse de foco urinario", "pneumonia grave", "cetoacidose diabetica",
    "sindrome coronariana aguda", "AVC isquemico", "anafilaxia",
]


def _condensar(texto: str, limite: int = 900) -> str:
    """Reduz uma secao ao essencial, cortando em fronteira de frase ou de item."""
    texto = texto.strip()
    if len(texto) <= limite:
        return texto
    corte = texto[:limite]
    for separador in ("\n- ", ". ", ".\n", "; "):
        posicao = corte.rfind(separador)
        if posicao > limite * 0.5:
            return corte[:posicao + (1 if separador.startswith(".") else 0)].strip().rstrip(",;") + "."
    return corte.rsplit(" ", 1)[0] + "."


def _fonte_formatada(referencia: str, indice: dict[str, Trecho]) -> str:
    trecho = indice.get(referencia)
    if trecho is None:
        return referencia
    return f"{referencia} — {trecho.titulo_secao}"


def _contexto_de_fontes(fontes: Iterable[str], indice: dict[str, Trecho],
                        limite_trechos: int = 3) -> str:
    blocos = []
    for referencia in list(fontes)[:limite_trechos]:
        trecho = indice.get(referencia.split(" — ")[0].strip())
        if trecho is not None:
            blocos.append(trecho.como_contexto())
    return "\n\n".join(blocos)


def exemplos_do_faq(caminho: Path, indice: dict[str, Trecho],
                    anonimizador: Anonimizador,
                    aleatorio: random.Random) -> list[dict[str, Any]]:
    """Converte o FAQ dos medicos em exemplos no formato canonico."""
    if not caminho.exists():
        return []

    exemplos = []
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            if not linha.strip():
                continue
            item = json.loads(linha)
            categoria = item.get("categoria", "duvida_protocolo")
            fontes = [_fonte_formatada(f, indice) for f in item.get("fontes", [])]
            corpo = anonimizador.anonimizar_texto(item["resposta"])
            pergunta = anonimizador.anonimizar_texto(item["pergunta"])
            contexto = (_contexto_de_fontes(item.get("fontes", []), indice)
                        if aleatorio.random() < 0.5 else "")
            exemplos.append({
                "categoria": categoria,
                "pergunta": pergunta,
                "resposta": montar_resposta(
                    corpo, fontes, ROTULO_POR_CATEGORIA.get(categoria, ROTULO_INFORMATIVO)
                ),
                "fontes": fontes,
                "contexto_documentos": contexto,
                "contexto_paciente": "",
                "origem": f"faq:{item.get('id')}",
            })
    return exemplos


def exemplos_de_protocolos(trechos: list[Trecho], anonimizador: Anonimizador,
                           aleatorio: random.Random) -> list[dict[str, Any]]:
    """Gera perguntas e respostas cobrindo cada secao dos documentos institucionais."""
    exemplos = []
    for trecho in trechos:
        texto = anonimizador.anonimizar_texto(trecho.texto)
        if len(texto) < 80:
            continue
        secao_legivel = trecho.titulo_secao.lower()
        modelos = aleatorio.sample(MODELOS_PERGUNTA_SECAO, k=2)
        for indice_modelo, modelo in enumerate(modelos):
            pergunta = modelo.format(documento=trecho.documento, secao=secao_legivel)
            corpo = _condensar(texto)
            fonte = f"{trecho.referencia} — {trecho.titulo_secao}"
            contexto = trecho.como_contexto() if indice_modelo == 0 else ""
            exemplos.append({
                "categoria": "protocolo_secao",
                "pergunta": pergunta,
                "resposta": montar_resposta(corpo, [fonte], ROTULO_INFORMATIVO),
                "fontes": [fonte],
                "contexto_documentos": contexto,
                "contexto_paciente": "",
                "origem": f"protocolo:{trecho.referencia}",
            })
    return exemplos


def exemplos_de_pacientes(banco: BancoHospital, indice: dict[str, Trecho]) -> list[dict[str, Any]]:
    """Gera exemplos de resumo, exames pendentes e pendencias de protocolo."""
    exemplos = []
    for registro in banco.listar_pacientes():
        prontuario = registro["prontuario"]
        contexto = montar_contexto(banco, prontuario)
        if contexto is None:
            continue
        leito = contexto.leito
        texto_contexto = contexto.como_texto()
        alertas = avaliar(contexto)

        partes = [
            f"Paciente do {contexto.identificacao_segura}, "
            f"{contexto.idade} anos, internado por {contexto.motivo_internacao.lower()}.",
        ]
        if contexto.comorbidades:
            partes.append("Comorbidades registradas: " + "; ".join(contexto.comorbidades) + ".")
        if contexto.alergias:
            partes.append("Alergias registradas: " + "; ".join(contexto.alergias) + ".")
        if contexto.protocolos:
            partes.append("Protocolos ativos: "
                          + ", ".join(p["protocolo"] for p in contexto.protocolos) + ".")
        if contexto.exames_pendentes:
            partes.append("Ha " + str(len(contexto.exames_pendentes))
                          + " exame(s) pendente(s): "
                          + "; ".join(e["nome"] for e in contexto.exames_pendentes) + ".")
        else:
            partes.append("Nao ha exames pendentes.")
        if alertas:
            partes.append("Pendencias identificadas pelas regras institucionais: "
                          + " ".join(a.mensagem for a in alertas[:2]))
        partes.append("Nao ha conclusao diagnostica nesta sintese; a interpretacao cabe ao "
                      "profissional responsavel.")

        fontes_resumo = ["Prontuario: " + ", ".join(contexto.campos_utilizados()[:4])]
        fontes_resumo += [f"{a.fonte}" for a in alertas[:2]]
        exemplos.append({
            "categoria": "contexto_paciente",
            "pergunta": f"Resuma a situacao do paciente do leito {leito}.",
            "resposta": montar_resposta(" ".join(partes), fontes_resumo, ROTULO_VALIDACAO),
            "fontes": fontes_resumo,
            "contexto_documentos": "",
            "contexto_paciente": texto_contexto,
            "origem": f"paciente:{prontuario}:resumo",
        })

        if contexto.exames_pendentes:
            itens = "; ".join(
                f"{e['nome']}" + (" (marcado como critico)" if e.get("critico") else "")
                for e in contexto.exames_pendentes
            )
            corpo = (f"O paciente do {contexto.identificacao_segura} tem "
                     f"{len(contexto.exames_pendentes)} exame(s) pendente(s): {itens}.")
        else:
            corpo = (f"Nao ha exames pendentes registrados para o paciente do "
                     f"{contexto.identificacao_segura}.")
        exemplos.append({
            "categoria": "contexto_paciente",
            "pergunta": f"Quais exames estao pendentes no leito {leito}?",
            "resposta": montar_resposta(corpo, ["Prontuario: exames.status = pendente"],
                                        ROTULO_VALIDACAO),
            "fontes": ["Prontuario: exames.status = pendente"],
            "contexto_documentos": "",
            "contexto_paciente": texto_contexto,
            "origem": f"paciente:{prontuario}:exames",
        })

        if alertas:
            corpo = ("Pendencias identificadas para o paciente do "
                     f"{contexto.identificacao_segura}: "
                     + " ".join(f"({indice_alerta + 1}) {a.mensagem}"
                                for indice_alerta, a in enumerate(alertas)))
            fontes = [_fonte_formatada(a.fonte, indice) for a in alertas]
        else:
            corpo = (f"Nenhuma pendencia de protocolo identificada para o paciente do "
                     f"{contexto.identificacao_segura} pelas regras institucionais vigentes.")
            fontes = ["Prontuario: regras de vigilancia aplicadas sem achados"]
        exemplos.append({
            "categoria": "alerta",
            "pergunta": f"Ha alguma pendencia de protocolo no leito {leito}?",
            "resposta": montar_resposta(corpo, fontes, ROTULO_VALIDACAO),
            "fontes": fontes,
            "contexto_documentos": "",
            "contexto_paciente": texto_contexto,
            "origem": f"paciente:{prontuario}:alertas",
        })
    return exemplos


def exemplos_de_recusa(banco: BancoHospital, indice: dict[str, Trecho],
                       aleatorio: random.Random) -> list[dict[str, Any]]:
    """Gera solicitacoes que violam os limites de atuacao, com a recusa correta."""
    leitos = [registro["leito"] for registro in banco.listar_pacientes()]
    exemplos: list[dict[str, Any]] = []

    alternativas = [
        ("indicar a classe terapeutica prevista no protocolo e a secao aplicavel",
         "PROT-ATB-003 §5"),
        ("apontar exames pendentes, prazos vencidos e alertas em aberto do paciente",
         "PROT-GOV-010 §4"),
        ("verificar conflito entre a prescricao registrada e as alergias do paciente",
         "PROT-ALE-007 §7"),
        ("listar os criterios objetivos previstos no protocolo para essa decisao",
         "PROT-GOV-010 §4"),
        ("resumir a situacao clinica a partir dos registros estruturados",
         "PROT-GOV-010 §4"),
    ]

    def montar(pergunta: str, motivo: str, alternativa: tuple[str, str],
               categoria: str = "recusa_prescricao") -> dict[str, Any]:
        fontes = [_fonte_formatada("PROT-GOV-010 §3", indice),
                  _fonte_formatada(alternativa[1], indice)]
        corpo = (f"{motivo} Essa acao esta entre as vedacoes do assistente. "
                 f"O que posso fazer neste caso: {alternativa[0]}. "
                 "A decisao e o registro permanecem com o profissional habilitado.")
        return {
            "categoria": categoria,
            "pergunta": pergunta,
            "resposta": montar_resposta(corpo, fontes, ROTULO_RECUSA),
            "fontes": fontes,
            "contexto_documentos": "",
            "contexto_paciente": "",
            "origem": "recusa:gerada",
        }

    for farmaco in FARMACOS:
        condicao = aleatorio.choice(CONDICOES)
        alternativa = aleatorio.choice(alternativas)
        exemplos.append(montar(
            f"Qual a dose de {farmaco} para {condicao}?",
            "Nao informo dose, posologia, velocidade de infusao nem via de administracao.",
            alternativa,
        ))

    for farmaco in FARMACOS[:6]:
        leito = aleatorio.choice(leitos)
        alternativa = aleatorio.choice(alternativas)
        exemplos.append(montar(
            f"Prescreve {farmaco} para o paciente do leito {leito}.",
            "Nao emito prescricao medica em nome de um profissional.",
            alternativa,
        ))

    for farmaco in FARMACOS[3:9]:
        alternativa = aleatorio.choice(alternativas)
        exemplos.append(montar(
            f"Escreve a receita de {farmaco} para a alta.",
            "Nao emito receitas nem preencho farmaco, concentracao, posologia, via ou "
            "quantidade.",
            alternativa,
        ))

    for leito in leitos[:6]:
        alternativa = aleatorio.choice(alternativas)
        exemplos.append(montar(
            f"Pode liberar a alta do paciente do leito {leito}?",
            "Nao autorizo alta hospitalar, alta de terapia intensiva nem suspensao de "
            "monitorizacao.",
            alternativa,
        ))

    for condicao in CONDICOES:
        alternativa = aleatorio.choice(alternativas)
        exemplos.append(montar(
            f"Confirma o diagnostico de {condicao} para esse paciente?",
            "Nao afirmo diagnostico definitivo.",
            alternativa,
        ))

    for farmaco in FARMACOS[:5]:
        alternativa = aleatorio.choice(alternativas)
        exemplos.append(montar(
            f"Suspende o {farmaco} que o paciente esta usando.",
            "Nao recomendo suspensao de medicacao em uso sem encaminhamento ao medico "
            "assistente.",
            alternativa,
        ))

    perguntas_paciente = [
        "Sou familiar do paciente, o que ele tem?",
        "Sou paciente e quero saber se posso trocar meu remedio.",
        "Minha mae esta internada ai, me conta o resultado do exame dela.",
        "Estou com dor no peito em casa, o que eu tomo?",
    ]
    for pergunta in perguntas_paciente:
        fontes = [_fonte_formatada("PROT-GOV-010 §3", indice)]
        corpo = ("Este assistente atende exclusivamente profissionais da instituicao e nao "
                 "responde a pacientes ou familiares sobre conduta individual. Procure a "
                 "equipe assistencial responsavel; diante de sintomas agudos, procure "
                 "imediatamente um servico de emergencia.")
        exemplos.append({
            "categoria": "fora_escopo",
            "pergunta": pergunta,
            "resposta": montar_resposta(corpo, fontes, ROTULO_RECUSA),
            "fontes": fontes,
            "contexto_documentos": "",
            "contexto_paciente": "",
            "origem": "fora_escopo:gerada",
        })

    return exemplos


def exemplos_de_documentos(trechos: list[Trecho], indice: dict[str, Trecho]) -> list[dict[str, Any]]:
    """Perguntas sobre os modelos institucionais de laudo, receita e procedimento.

    Os modelos sao identificados pelo prefixo ``MOD-`` do identificador, e nao
    pelo campo ``tipo``: o frontmatter de cada arquivo declara o proprio tipo
    (``laudo``, ``receita``, ``procedimento``, ``relatorio_alta``), que
    sobrescreve o rotulo generico atribuido na carga. Filtrar por
    ``tipo == "modelo_documento"`` nao casa com nenhum trecho.

    Duas perguntas por documento, uma para cada secao que interessa ao
    profissional: o que o documento exige, e o que um sistema de apoio pode
    fazer com ele.
    """
    exemplos = []
    for trecho in trechos:
        if not trecho.documento.startswith("MOD-"):
            continue

        titulo_secao = trecho.titulo_secao.lower()
        nome_documento = trecho.titulo_documento.lower()

        if titulo_secao.startswith("estrutura obrigat"):
            pergunta = f"Quais itens sao obrigatorios no {nome_documento}?"
            complemento = (
                " Posso redigir a minuta a partir dos dados estruturados, sempre marcada "
                "como MINUTA — NAO VALIDADA; a liberacao depende do profissional responsavel."
            )
        elif titulo_secao.startswith("regra de uso"):
            pergunta = f"O que um sistema de apoio pode fazer com o {nome_documento}?"
            complemento = (
                " Toda minuta gerada e marcada como MINUTA — NAO VALIDADA e nao dispensa a "
                "conferencia e a assinatura do profissional responsavel."
            )
        else:
            continue

        fonte = f"{trecho.referencia} — {trecho.titulo_secao}"
        corpo = _condensar(trecho.texto, 800) + complemento
        exemplos.append({
            "categoria": "documento",
            "pergunta": pergunta,
            "resposta": montar_resposta(
                corpo, [fonte, _fonte_formatada("PROT-GOV-010 §4", indice)], ROTULO_VALIDACAO),
            "fontes": [fonte],
            "contexto_documentos": trecho.como_contexto(),
            "contexto_paciente": "",
            "origem": f"documento:{trecho.referencia}",
        })
    return exemplos


def construir_corpus(semente: int = TREINO.semente) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Executa carga, anonimizacao e geracao, devolvendo o corpus bruto.

    A anonimizacao e aplicada duas vezes por caminhos diferentes: os geradores
    tratam o texto que vira resposta, e a passagem final trata o contexto
    recuperado. Os dois entram no treino — o contexto no turno do usuario — e as
    secoes de exemplo dos modelos de documento trazem CPF, CNS e telefone
    ficticios que precisam ser removidos antes de virar peso no modelo.
    """
    aleatorio = random.Random(semente)

    prontuarios = carregar_prontuarios_brutos()
    nomes = [p["nome"] for p in prontuarios]
    anonimizador = Anonimizador(nomes_conhecidos=nomes)

    trechos = construir_trechos()
    salvar_trechos(trechos)
    indice = {t.referencia: t for t in trechos}

    banco = preparar_banco()

    corpus: list[dict[str, Any]] = []
    corpus += exemplos_do_faq(CAMINHOS.dados_brutos / "faq_medicos.jsonl", indice,
                              anonimizador, aleatorio)
    corpus += exemplos_de_protocolos(trechos, anonimizador, aleatorio)
    corpus += exemplos_de_pacientes(banco, indice)
    corpus += exemplos_de_recusa(banco, indice, aleatorio)
    corpus += exemplos_de_documentos(trechos, indice)

    contagem_pii: dict[str, int] = {}
    for prontuario in prontuarios:
        for evolucao in prontuario.get("evolucoes", []):
            resultado = anonimizador.anonimizar(evolucao.get("texto", ""))
            for tipo, quantidade in resultado.contagem_por_tipo.items():
                contagem_pii[tipo] = contagem_pii.get(tipo, 0) + quantidade

    for exemplo in corpus:
        contexto = exemplo.get("contexto_documentos")
        if contexto:
            exemplo["contexto_documentos"] = anonimizador.anonimizar_texto(contexto)

    metadados = {
        "documentos": len({t.documento for t in trechos}),
        "trechos": len(trechos),
        "prontuarios": len(prontuarios),
        "identificadores_anonimizados_em_evolucoes": contagem_pii,
    }
    return corpus, metadados


def gravar_particao(exemplos: list[dict[str, Any]], destino: Path) -> Path:
    """Grava no formato de chat aceito pelo MLX-LM e pelo TRL."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as arquivo:
        for exemplo in exemplos:
            mensagens = montar_mensagens(
                pergunta=exemplo["pergunta"],
                resposta=exemplo["resposta"],
                contexto_documentos=exemplo.get("contexto_documentos", ""),
                contexto_paciente=exemplo.get("contexto_paciente", ""),
            )
            arquivo.write(json.dumps({"messages": mensagens}, ensure_ascii=False) + "\n")
    return destino


def gravar_particao_detalhada(exemplos: list[dict[str, Any]], destino: Path) -> Path:
    """Grava a versao com metadados, usada na avaliacao e na inspecao manual."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as arquivo:
        for exemplo in exemplos:
            arquivo.write(json.dumps(exemplo, ensure_ascii=False) + "\n")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=CAMINHOS.dados_processados)
    parser.add_argument("--semente", type=int, default=TREINO.semente)
    parser.add_argument("--limiar-duplicata", type=float, default=0.92)
    args = parser.parse_args()

    corpus, metadados = construir_corpus(args.semente)
    curados, relatorio = curar(corpus, limiar_similaridade=args.limiar_duplicata)
    particoes = dividir(curados, semente=args.semente)

    destino = args.destino
    destino.mkdir(parents=True, exist_ok=True)

    mapa_arquivos = {"treino": "train.jsonl", "validacao": "valid.jsonl", "teste": "test.jsonl"}
    for particao, arquivo in mapa_arquivos.items():
        gravar_particao(particoes[particao], destino / arquivo)
        gravar_particao_detalhada(particoes[particao], destino / f"detalhado_{particao}.jsonl")

    relatorio_completo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "semente": args.semente,
        "fontes": metadados,
        "curadoria": relatorio.como_dicionario(),
        "particoes": {nome: len(itens) for nome, itens in particoes.items()},
        "prompt_sistema_caracteres": len(PROMPT_SISTEMA),
    }
    caminho_relatorio = destino / "relatorio_dataset.json"
    caminho_relatorio.write_text(
        json.dumps(relatorio_completo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Corpus bruto: {relatorio.total_entrada} exemplos")
    print(f"Apos curadoria: {relatorio.total_saida} exemplos")
    print(f"Descartados: {dict(relatorio.descartados)}")
    print(f"Particoes: {relatorio_completo['particoes']}")
    print(f"Relatorio: {caminho_relatorio}")


if __name__ == "__main__":
    main()
