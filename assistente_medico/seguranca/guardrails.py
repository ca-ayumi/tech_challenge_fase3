"""Guardrails de entrada e de saida.

Entrada: classifica a solicitacao antes de qualquer chamada ao modelo. Se ela
viola um limite de atuacao, a resposta e montada por regra e o modelo nao chega
a ser chamado — o que elimina a chance de o modelo "escorregar" e tambem economiza
tempo de inferencia.

Saida: revisa o texto gerado antes de entregar. Verifica dose, linguagem
prescritiva, afirmacao diagnostica, presenca de fonte, rotulo de validacao e
vazamento de dados pessoais. Uma violacao critica substitui a resposta; as
demais anexam aviso.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import SEGURANCA
from ..dados.anonimizacao import Anonimizador
from ..finetuning.formato import (
    ROTULO_EMERGENCIA,
    ROTULO_RECUSA,
    ROTULO_VALIDACAO,
    montar_resposta,
    segue_formato,
    separar_secoes,
)
from .politicas import POLITICAS_ENTRADA, POLITICAS_SAIDA, Politica


@dataclass
class Violacao:
    """Registro de uma politica que disparou."""

    codigo: str
    descricao: str
    severidade: str
    acao: str
    fonte: str
    trecho: str = ""

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "codigo": self.codigo,
            "descricao": self.descricao,
            "severidade": self.severidade,
            "acao": self.acao,
            "fonte": self.fonte,
            "trecho": self.trecho,
        }


@dataclass
class ResultadoEntrada:
    """Decisao do guardrail de entrada."""

    permitido: bool
    categoria: str
    violacoes: list[Violacao] = field(default_factory=list)
    resposta_pronta: str | None = None

    @property
    def codigos(self) -> list[str]:
        return [v.codigo for v in self.violacoes]


@dataclass
class ResultadoSaida:
    """Decisao do guardrail de saida."""

    texto: str
    aprovado: bool
    violacoes: list[Violacao] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    @property
    def codigos(self) -> list[str]:
        return [v.codigo for v in self.violacoes]


ALTERNATIVAS_POR_CODIGO = {
    "ENT-01-prescricao": ("indicar a classe terapeutica prevista no protocolo, a duracao padrao "
                          "e a secao aplicavel", "PROT-ATB-003 §5"),
    "ENT-02-dose": ("apontar em qual secao do protocolo a conduta esta descrita e quais "
                    "checagens sao exigidas antes da prescricao", "PROT-GOV-010 §4"),
    "ENT-03-alta": ("listar os criterios objetivos de estabilidade previstos no protocolo e "
                    "reunir os dados do prontuario que apoiam a sua decisao", "PROT-GOV-010 §4"),
    "ENT-04-diagnostico": ("apresentar os achados registrados, exames liberados e pendentes, e "
                           "o protocolo aplicavel a investigacao", "PROT-GOV-010 §4"),
    "ENT-05-suspensao": ("sinalizar conflitos com alergias registradas e pendencias de "
                         "reavaliacao previstas em protocolo", "PROT-ALE-007 §7"),
    "ENT-06-paciente": ("orientar a procurar a equipe assistencial responsavel", "PROT-GOV-010 §3"),
    "ENT-08-sobreposicao": ("responder a pergunta clinica dentro dos limites vigentes, citando "
                            "a secao de protocolo aplicavel", "PROT-GOV-010 §4"),
}


class Guardrails:
    """Aplica as politicas de entrada e de saida."""

    def __init__(self, anonimizador: Anonimizador | None = None,
                 exigir_validacao: bool | None = None) -> None:
        self.anonimizador = anonimizador or Anonimizador(
            estrategias={"prontuario": "manter"}
        )
        self.exigir_validacao = (
            SEGURANCA.exigir_validacao_humana if exigir_validacao is None else exigir_validacao
        )

    def avaliar_entrada(self, pergunta: str, perfil: str = "medico") -> ResultadoEntrada:
        violacoes: list[Violacao] = []
        emergencia: Politica | None = None
        bloqueios: list[Politica] = []

        if perfil not in SEGURANCA.perfis_autorizados:
            politica = next(p for p in POLITICAS_ENTRADA if p.codigo == "ENT-06-paciente")
            violacoes.append(Violacao(
                codigo=politica.codigo,
                descricao=f"Perfil '{perfil}' nao autorizado a receber conduta individual.",
                severidade=politica.severidade, acao=politica.acao, fonte=politica.fonte,
                trecho=f"perfil={perfil}",
            ))
            bloqueios.append(politica)

        for politica in POLITICAS_ENTRADA:
            trecho = politica.casa(pergunta)
            if not trecho:
                continue
            violacoes.append(Violacao(
                codigo=politica.codigo, descricao=politica.descricao,
                severidade=politica.severidade, acao=politica.acao,
                fonte=politica.fonte, trecho=trecho,
            ))
            if politica.acao == "encaminhar":
                emergencia = politica
            elif politica.acao == "bloquear":
                bloqueios.append(politica)

        if emergencia is not None:
            return ResultadoEntrada(
                permitido=True, categoria="emergencia", violacoes=violacoes,
                resposta_pronta=None,
            )

        if bloqueios:
            unicos: list[Politica] = []
            for politica in bloqueios:
                if politica.codigo not in {p.codigo for p in unicos}:
                    unicos.append(politica)
            bloqueios = unicos
            violacoes = [v for i, v in enumerate(violacoes)
                         if v.codigo not in {x.codigo for x in violacoes[:i]}]
            return ResultadoEntrada(
                permitido=False, categoria="recusa", violacoes=violacoes,
                resposta_pronta=self._montar_recusa(bloqueios),
            )

        return ResultadoEntrada(permitido=True, categoria="permitido", violacoes=[])

    def _montar_recusa(self, politicas: list[Politica]) -> str:
        motivos = []
        fontes = []
        alternativas = []
        for politica in politicas:
            if politica.mensagem:
                motivos.append(politica.mensagem)
            fontes.append(politica.fonte)
            alternativa = ALTERNATIVAS_POR_CODIGO.get(politica.codigo)
            if alternativa and alternativa[0] not in alternativas:
                alternativas.append(alternativa[0])
                fontes.append(alternativa[1])

        corpo = " ".join(motivos)
        if alternativas:
            corpo += (" O que posso fazer neste caso: " + "; ".join(alternativas) + ".")
        corpo += (" A decisao e o registro permanecem com o profissional habilitado "
                  "(PROT-GOV-010 §5).")

        vistas: list[str] = []
        for fonte in fontes:
            if fonte not in vistas:
                vistas.append(fonte)
        return montar_resposta(corpo, vistas, ROTULO_RECUSA)

    def avaliar_saida(self, texto: str, categoria_entrada: str = "permitido",
                      fontes_esperadas: list[str] | None = None) -> ResultadoSaida:
        violacoes: list[Violacao] = []
        avisos: list[str] = []

        for politica in POLITICAS_SAIDA:
            trecho = politica.casa(texto)
            if trecho:
                violacoes.append(Violacao(
                    codigo=politica.codigo, descricao=politica.descricao,
                    severidade=politica.severidade, acao=politica.acao,
                    fonte=politica.fonte, trecho=trecho,
                ))

        criticas = [v for v in violacoes if v.acao == "bloquear"]
        if criticas:
            corpo = (
                "A resposta gerada foi bloqueada pela politica de seguranca antes da entrega: "
                + " ".join(v.descricao for v in criticas)
                + " Reformule a pergunta pedindo o criterio previsto em protocolo, a secao "
                  "aplicavel ou o contexto do paciente — a definicao terapeutica e do "
                  "profissional habilitado."
            )
            fontes = []
            for violacao in criticas:
                if violacao.fonte not in fontes:
                    fontes.append(violacao.fonte)
            return ResultadoSaida(
                texto=montar_resposta(corpo, fontes, ROTULO_RECUSA),
                aprovado=False, violacoes=violacoes,
                avisos=["resposta substituida por violacao critica de politica"],
            )

        texto_final = texto

        resultado_pii = self.anonimizador.anonimizar(texto_final)
        efetivas = [o for o in resultado_pii.ocorrencias if o.estrategia != "manter"]
        if efetivas and SEGURANCA.mascarar_pii_nos_logs:
            tipos = ", ".join(sorted({o.tipo for o in efetivas}))
            violacoes.append(Violacao(
                codigo="SAI-05-pii",
                descricao=f"Dados pessoais na resposta ({tipos}); mascarados antes da entrega.",
                severidade="alta", acao="mascarar", fonte="PROT-GOV-010 §8",
                trecho=tipos,
            ))
            texto_final = resultado_pii.texto
            avisos.append("dados pessoais mascarados na resposta")

        if not segue_formato(texto_final):
            estruturada = separar_secoes(texto_final)
            corpo = estruturada.corpo or texto_final.strip()
            rotulo = ROTULO_EMERGENCIA if categoria_entrada == "emergencia" else ROTULO_VALIDACAO
            texto_final = montar_resposta(corpo, estruturada.fontes or (fontes_esperadas or []),
                                          estruturada.validacao or rotulo)
            avisos.append("formato normalizado para a estrutura institucional")
            violacoes.append(Violacao(
                codigo="SAI-06-formato",
                descricao="Resposta fora da estrutura de tres blocos; normalizada.",
                severidade="baixa", acao="registrar", fonte="PROT-GOV-010 §7",
            ))

        estruturada = separar_secoes(texto_final)

        if not estruturada.fontes:
            if fontes_esperadas:
                texto_final = montar_resposta(estruturada.corpo, fontes_esperadas,
                                              estruturada.validacao)
                avisos.append("fontes recuperadas anexadas a resposta")
            else:
                texto_final = montar_resposta(
                    estruturada.corpo
                    + "\n\nObservacao: nao foi possivel fundamentar esta resposta em documento "
                      "institucional recuperado.",
                    [], estruturada.validacao,
                )
                violacoes.append(Violacao(
                    codigo="SAI-07-sem-fonte",
                    descricao="Resposta sem fonte institucional rastreavel.",
                    severidade="media", acao="anexar_aviso", fonte="PROT-GOV-010 §7",
                ))
                avisos.append("resposta marcada como nao fundamentada")

        estruturada = separar_secoes(texto_final)
        if self.exigir_validacao and not estruturada.validacao:
            rotulo = ROTULO_EMERGENCIA if categoria_entrada == "emergencia" else ROTULO_VALIDACAO
            texto_final = montar_resposta(estruturada.corpo, estruturada.fontes, rotulo)
            avisos.append("rotulo de validacao humana anexado")

        avisos_diagnostico = [v for v in violacoes if v.acao == "anexar_aviso"]
        if avisos_diagnostico:
            estruturada = separar_secoes(texto_final)
            corpo = (estruturada.corpo
                     + "\n\nAviso: esta resposta nao estabelece diagnostico; a conclusao "
                       "diagnostica cabe ao profissional responsavel (PROT-GOV-010 §3).")
            texto_final = montar_resposta(corpo, estruturada.fontes, estruturada.validacao)

        return ResultadoSaida(
            texto=texto_final,
            aprovado=not any(v.severidade == "critica" for v in violacoes),
            violacoes=violacoes,
            avisos=avisos,
        )
