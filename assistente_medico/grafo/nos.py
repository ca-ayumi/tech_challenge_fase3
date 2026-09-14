"""Nos do fluxo de decisao.

Cada no e uma funcao pura sobre o estado: recebe o estado, devolve os campos que
alterou. As dependencias (banco, recuperador, modelo, guardrails, auditoria)
entram por injecao em ``Dependencias``, o que permite testar cada no isolado com
dublês.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..auditoria import Auditoria
from ..dados.banco import BancoHospital, preparar_banco
from ..dados.contexto_paciente import montar_contexto
from ..explicabilidade import montar_explicacao
from ..finetuning.formato import (
    PROMPT_SISTEMA,
    montar_pergunta,
    montar_resposta,
)
from ..llm.langchain_llm import ChatAssistenteMedico, criar_modelo_chat
from ..rag.recuperador import RecuperadorProtocolos, TrechoRecuperado, recuperador_padrao
from ..seguranca.guardrails import Guardrails
from ..seguranca.regras_clinicas import avaliar
from .estado import EstadoAssistente

logger = logging.getLogger(__name__)

PADRAO_LEITO = re.compile(r"\b(PS|UTI|ENF|CIR)[\s-]?(\d{2,3})\b", re.IGNORECASE)
PADRAO_LEITO_SOLTO = re.compile(r"\bleito\s+([A-Za-z]{0,3}[\s-]?\d{2,3})\b", re.IGNORECASE)
PADRAO_PRONTUARIO = re.compile(r"\bprontu[aá]rio\s*(?:n[ºo°]?\.?\s*)?(\d{4,8})\b", re.IGNORECASE)

PALAVRAS_DOCUMENTO = (
    "laudo", "receita", "receituario", "receituário", "relatorio de alta",
    "relatório de alta", "descricao de procedimento", "descrição de procedimento",
    "minuta", "modelo de documento",
)


@dataclass
class Dependencias:
    """Objetos externos usados pelos nos."""

    banco: BancoHospital = field(default_factory=preparar_banco)
    recuperador: RecuperadorProtocolos = field(default_factory=recuperador_padrao)
    modelo: ChatAssistenteMedico | None = None
    guardrails: Guardrails = field(default_factory=Guardrails)
    auditoria: Auditoria | None = None

    def __post_init__(self) -> None:
        if self.modelo is None:
            self.modelo = criar_modelo_chat()
        if self.auditoria is None:
            self.auditoria = Auditoria(banco=self.banco)


def _identificar_paciente(dependencias: Dependencias, estado: EstadoAssistente) -> str | None:
    """Resolve a referencia ao paciente na pergunta ou no parametro explicito."""
    informado = estado.get("prontuario_informado")
    if informado:
        encontrados = dependencias.banco.buscar_paciente(str(informado))
        if encontrados:
            return encontrados[0]["prontuario"]

    pergunta = estado.get("pergunta", "")

    achado = PADRAO_PRONTUARIO.search(pergunta)
    if achado:
        encontrados = dependencias.banco.buscar_paciente(achado.group(1))
        if encontrados:
            return encontrados[0]["prontuario"]

    achado = PADRAO_LEITO.search(pergunta)
    if achado:
        leito = f"{achado.group(1).upper()}-{achado.group(2)}"
        encontrados = dependencias.banco.buscar_paciente(leito)
        if encontrados:
            return encontrados[0]["prontuario"]

    achado = PADRAO_LEITO_SOLTO.search(pergunta)
    if achado:
        bruto = achado.group(1).strip().upper().replace(" ", "-")
        encontrados = dependencias.banco.buscar_paciente(bruto)
        if encontrados:
            return encontrados[0]["prontuario"]
        numero = re.sub(r"\D", "", bruto)
        for paciente in dependencias.banco.listar_pacientes():
            if paciente["leito"].endswith(numero):
                return paciente["prontuario"]
    return None


def criar_nos(dependencias: Dependencias) -> dict[str, Any]:
    """Cria as funcoes de no ligadas as dependencias informadas."""
    auditoria = dependencias.auditoria
    assert auditoria is not None

    def triagem(estado: EstadoAssistente) -> dict[str, Any]:
        """Aplica o guardrail de entrada e classifica a intencao da pergunta."""
        pergunta = estado.get("pergunta", "")
        perfil = estado.get("perfil", "medico")

        resultado = dependencias.guardrails.avaliar_entrada(pergunta, perfil)
        prontuario = _identificar_paciente(dependencias, estado)

        texto = pergunta.lower()
        if resultado.categoria == "emergencia":
            intencao = "emergencia"
        elif any(palavra in texto for palavra in PALAVRAS_DOCUMENTO):
            intencao = "documento"
        elif prontuario:
            intencao = "paciente"
        else:
            intencao = "protocolo"

        auditoria.registrar(
            estado["interacao"], "triagem",
            {
                "pergunta": pergunta,
                "categoria": resultado.categoria,
                "intencao": intencao,
                "violacoes": [v.como_dicionario() for v in resultado.violacoes],
                "prontuario_resolvido": prontuario,
            },
            usuario=estado.get("usuario"), perfil=perfil, prontuario=prontuario,
        )

        return {
            "categoria": resultado.categoria,
            "permitido": resultado.permitido,
            "intencao": intencao,
            "prontuario": prontuario,
            "violacoes": [v.como_dicionario() for v in resultado.violacoes],
            "resposta_bruta": resultado.resposta_pronta or "",
            "etapas": ["triagem"],
        }

    def recusar(estado: EstadoAssistente) -> dict[str, Any]:
        """Entrega a recusa montada por regra, sem acionar o modelo."""
        resposta = estado.get("resposta_bruta") or montar_resposta(
            "Solicitacao fora dos limites de atuacao do assistente.",
            ["PROT-GOV-010 §3"],
        )
        auditoria.registrar(
            estado["interacao"], "recusa_por_politica",
            {"resposta": resposta, "violacoes": estado.get("violacoes", [])},
            usuario=estado.get("usuario"), perfil=estado.get("perfil"),
        )
        return {
            "resposta": resposta,
            "avisos": ["modelo nao foi acionado: solicitacao bloqueada na entrada"],
            "etapas": ["recusa"],
        }

    def carregar_contexto_paciente(estado: EstadoAssistente) -> dict[str, Any]:
        """Le o prontuario e monta o contexto clinico minimizado."""
        prontuario = estado.get("prontuario")
        if not prontuario:
            return {"etapas": ["contexto_paciente:sem_paciente"]}

        with auditoria.medir(estado["interacao"], "contexto_paciente",
                             prontuario=prontuario) as dados:
            contexto = montar_contexto(dependencias.banco, prontuario)
            if contexto is None:
                dados["encontrado"] = False
                return {"etapas": ["contexto_paciente:nao_encontrado"]}
            dados["encontrado"] = True
            dados["campos"] = contexto.campos_utilizados()

        return {
            "contexto_paciente": contexto.como_texto(),
            "campos_prontuario": contexto.campos_utilizados(),
            "etapas": ["contexto_paciente"],
        }

    def verificar_exames_pendentes(estado: EstadoAssistente) -> dict[str, Any]:
        """Separa os exames pendentes do paciente, destacando os criticos."""
        prontuario = estado.get("prontuario")
        if not prontuario:
            return {"etapas": ["exames:sem_paciente"]}

        pendentes = dependencias.banco.exames_pendentes(prontuario)
        criticos = [e for e in pendentes if e.get("critico")]
        auditoria.registrar(
            estado["interacao"], "exames_pendentes",
            {"total": len(pendentes), "criticos": len(criticos),
             "exames": [e["nome"] for e in pendentes]},
            prontuario=prontuario,
        )
        return {
            "exames_pendentes": [
                {"nome": e["nome"], "codigo": e["codigo"], "critico": bool(e.get("critico")),
                 "solicitado_em": e.get("solicitado_em")}
                for e in pendentes
            ],
            "etapas": ["exames_pendentes"],
        }

    def avaliar_regras(estado: EstadoAssistente) -> dict[str, Any]:
        """Aplica as regras deterministicas de vigilancia sobre o paciente."""
        prontuario = estado.get("prontuario")
        if not prontuario:
            return {"etapas": ["regras:sem_paciente"]}

        contexto = montar_contexto(dependencias.banco, prontuario)
        if contexto is None:
            return {"etapas": ["regras:sem_contexto"]}

        alertas = avaliar(contexto)
        auditoria.registrar(
            estado["interacao"], "regras_aplicadas",
            {"alertas": [a.como_dicionario() for a in alertas],
             "regras": [a.regra for a in alertas]},
            prontuario=prontuario,
        )
        return {
            "alertas": [a.como_dicionario() for a in alertas],
            "etapas": ["avaliar_regras"],
        }

    def recuperar_protocolos(estado: EstadoAssistente) -> dict[str, Any]:
        """Recupera as secoes de protocolo relevantes para a pergunta."""
        pergunta = estado.get("pergunta", "")
        prontuario = estado.get("prontuario")

        preferidos: list[str] = []
        if prontuario:
            preferidos = [p["protocolo"]
                          for p in dependencias.banco.protocolos_ativos(prontuario)]
        preferidos += [a["fonte"].split(" ")[0] for a in estado.get("alertas", [])
                       if a.get("fonte")]

        consulta = pergunta
        if estado.get("alertas"):
            consulta = pergunta + " " + " ".join(a["mensagem"] for a in estado["alertas"][:2])

        with auditoria.medir(estado["interacao"], "recuperacao_protocolos") as dados:
            recuperados: list[TrechoRecuperado] = dependencias.recuperador.recuperar(
                consulta, documentos_preferidos=preferidos
            )
            dados["consulta"] = consulta[:300]
            dados["documentos_preferidos"] = preferidos
            dados["recuperados"] = [r.como_dicionario() for r in recuperados]

        return {
            "contexto_documentos": dependencias.recuperador.formatar_contexto(recuperados),
            "recuperados": [r.como_dicionario() for r in recuperados],
            "etapas": ["recuperar_protocolos"],
        }

    def gerar_resposta(estado: EstadoAssistente) -> dict[str, Any]:
        """Chama a LLM customizada com o contexto montado."""
        instrucao = estado.get("pergunta", "")
        if estado.get("categoria") == "emergencia":
            instrucao = (
                f"{instrucao}\n\n[Orientacao do sistema: situacao de emergencia detectada. "
                "Oriente o acionamento imediato da equipe antes de qualquer outra "
                "informacao, e cite o protocolo aplicavel.]"
            )
        if estado.get("alertas"):
            resumo = "; ".join(a["mensagem"] for a in estado["alertas"][:3])
            instrucao = (f"{instrucao}\n\n[Pendencias detectadas pelas regras institucionais: "
                         f"{resumo}]")

        conteudo = montar_pergunta(
            instrucao,
            contexto_documentos=estado.get("contexto_documentos", ""),
            contexto_paciente=estado.get("contexto_paciente", ""),
        )
        mensagens = [SystemMessage(content=PROMPT_SISTEMA), HumanMessage(content=conteudo)]

        try:
            with auditoria.medir(estado["interacao"], "geracao_modelo") as dados:
                resposta = dependencias.modelo.invoke(mensagens)
                dados["metadados"] = resposta.response_metadata
                dados["caracteres"] = len(resposta.content)
            return {
                "resposta_bruta": resposta.content,
                "metadados_modelo": resposta.response_metadata,
                "etapas": ["gerar_resposta"],
            }
        except Exception as erro:  # pragma: no cover - falha de backend
            logger.exception("Falha na geracao")
            corpo = ("Nao foi possivel gerar a resposta com o modelo neste momento. "
                     "O contexto institucional recuperado esta listado abaixo para "
                     "consulta direta.\n\n"
                     + estado.get("contexto_documentos", "")[:1500])
            return {
                "resposta_bruta": montar_resposta(
                    corpo, [r["referencia"] for r in estado.get("recuperados", [])]
                ),
                "erro": f"{type(erro).__name__}: {erro}",
                "etapas": ["gerar_resposta:falha"],
            }

    def validar_saida(estado: EstadoAssistente) -> dict[str, Any]:
        """Revisa a resposta gerada antes da entrega."""
        fontes_esperadas = [r["referencia"] for r in estado.get("recuperados", [])][:3]
        resultado = dependencias.guardrails.avaliar_saida(
            estado.get("resposta_bruta", ""),
            categoria_entrada=estado.get("categoria", "permitido"),
            fontes_esperadas=fontes_esperadas,
        )
        auditoria.registrar(
            estado["interacao"], "guardrail_saida",
            {"aprovado": resultado.aprovado,
             "violacoes": [v.como_dicionario() for v in resultado.violacoes],
             "avisos": resultado.avisos},
            prontuario=estado.get("prontuario"),
        )
        return {
            "resposta": resultado.texto,
            "violacoes": [v.como_dicionario() for v in resultado.violacoes],
            "avisos": resultado.avisos,
            "etapas": ["validar_saida"],
        }

    def emitir_alertas(estado: EstadoAssistente) -> dict[str, Any]:
        """Registra no painel institucional os alertas disparados pelas regras."""
        alertas = estado.get("alertas", [])
        if not alertas:
            return {"etapas": ["emitir_alertas:nenhum"]}

        registrados = []
        for alerta in alertas:
            identificador = dependencias.banco.registrar_alerta(
                prioridade=alerta["prioridade"], categoria=alerta["categoria"],
                mensagem=alerta["mensagem"], prontuario=estado.get("prontuario"),
                fonte=alerta.get("fonte"), interacao=estado["interacao"],
            )
            registrados.append(identificador)

        auditoria.registrar(
            estado["interacao"], "alertas_emitidos",
            {"ids": registrados, "prioridades": [a["prioridade"] for a in alertas]},
            prontuario=estado.get("prontuario"),
        )
        return {"alertas_registrados": registrados, "etapas": ["emitir_alertas"]}

    def explicar(estado: EstadoAssistente) -> dict[str, Any]:
        """Monta o rastro de fontes e detecta citacoes nao fundamentadas."""
        recuperados = [
            TrechoRecuperado(
                trecho=dependencias.recuperador.por_referencia(item["referencia"]),
                score=item.get("score", 0.0), motivo=item.get("motivo", ""),
            )
            for item in estado.get("recuperados", [])
            if dependencias.recuperador.por_referencia(item["referencia"]) is not None
        ]
        explicacao = montar_explicacao(
            estado.get("resposta", ""),
            recuperados=recuperados,
            campos_prontuario=estado.get("campos_prontuario", []),
            regras_aplicadas=[a.get("regra", "") for a in estado.get("alertas", [])],
        )
        auditoria.registrar(
            estado["interacao"], "explicabilidade", explicacao.como_dicionario(),
            prontuario=estado.get("prontuario"),
        )
        return {"explicacao": explicacao.como_dicionario(), "etapas": ["explicar"]}

    def abrir_validacao(estado: EstadoAssistente) -> dict[str, Any]:
        """Abre pedido de validacao humana quando a resposta sugere conduta."""
        if estado.get("categoria") == "recusa":
            return {"etapas": ["validacao:dispensada"]}
        if not (estado.get("prontuario") or estado.get("alertas")):
            return {"etapas": ["validacao:dispensada"]}

        identificador = dependencias.banco.registrar_validacao(
            interacao=estado["interacao"],
            sugestao=estado.get("resposta", ""),
            prontuario=estado.get("prontuario"),
            fontes=[r["referencia"] for r in estado.get("recuperados", [])],
        )
        auditoria.registrar(
            estado["interacao"], "validacao_aberta",
            {"validacao_id": identificador, "status": "pendente"},
            prontuario=estado.get("prontuario"),
        )
        return {"validacao_id": identificador, "etapas": ["abrir_validacao"]}

    def concluir(estado: EstadoAssistente) -> dict[str, Any]:
        """Fecha a interacao registrando o desfecho."""
        auditoria.registrar(
            estado["interacao"], "interacao_concluida",
            {
                "categoria": estado.get("categoria"),
                "intencao": estado.get("intencao"),
                "etapas": estado.get("etapas", []),
                "violacoes": [v.get("codigo") for v in estado.get("violacoes", [])],
                "alertas_registrados": estado.get("alertas_registrados", []),
                "validacao_id": estado.get("validacao_id"),
                "cobertura_citacoes": estado.get("explicacao", {}).get(
                    "cobertura_das_citacoes"),
                "erro": estado.get("erro"),
            },
            usuario=estado.get("usuario"), perfil=estado.get("perfil"),
            prontuario=estado.get("prontuario"),
        )
        return {"etapas": ["concluir"]}

    return {
        "triagem": triagem,
        "recusar": recusar,
        "carregar_contexto_paciente": carregar_contexto_paciente,
        "verificar_exames_pendentes": verificar_exames_pendentes,
        "avaliar_regras": avaliar_regras,
        "recuperar_protocolos": recuperar_protocolos,
        "gerar_resposta": gerar_resposta,
        "validar_saida": validar_saida,
        "emitir_alertas": emitir_alertas,
        "explicar": explicar,
        "abrir_validacao": abrir_validacao,
        "concluir": concluir,
    }
