"""Montagem do contexto clinico entregue ao modelo.

Aplica minimizacao de dados: o modelo de linguagem recebe apenas o que e
clinicamente necessario (idade, unidade, leito, sinais vitais, exames,
alergias, protocolos). Identificadores diretos — nome, CPF, CNS, RG, telefone,
e-mail, endereco — nao entram no prompt. O profissional continua vendo o nome
na interface; o modelo trabalha com o leito e o numero de prontuario.

O mesmo construtor e usado na geracao do dataset de treino e na inferencia, o
que evita divergencia entre o que o modelo viu no treino e o que recebe em
producao.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .banco import BancoHospital

SEXO_EXTENSO = {"F": "feminino", "M": "masculino"}


def _agora() -> datetime:
    return datetime.now()


def _parse(valor: str | None) -> datetime | None:
    if not valor:
        return None
    try:
        return datetime.fromisoformat(valor)
    except ValueError:
        return None


def minutos_desde(valor: str | None) -> int | None:
    """Minutos decorridos desde um instante ISO, ou None se nao houver data."""
    instante = _parse(valor)
    if instante is None:
        return None
    return max(int((_agora() - instante).total_seconds() // 60), 0)


def formatar_intervalo(minutos: int | None) -> str:
    if minutos is None:
        return "sem registro de horario"
    if minutos < 60:
        return f"ha {minutos} min"
    horas, resto = divmod(minutos, 60)
    if horas < 24:
        return f"ha {horas}h{resto:02d}min"
    dias, horas_restantes = divmod(horas, 24)
    return f"ha {dias}d{horas_restantes}h"


def calcular_idade(data_nascimento: str | None, referencia: datetime | None = None) -> int | None:
    nascimento = _parse(data_nascimento)
    if nascimento is None:
        return None
    referencia = referencia or _agora()
    idade = referencia.year - nascimento.year
    if (referencia.month, referencia.day) < (nascimento.month, nascimento.day):
        idade -= 1
    return idade


def _numero(valor: Any) -> str:
    if valor is None:
        return "-"
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    return str(valor).replace(".", ",")


@dataclass
class ContextoPaciente:
    """Visao consolidada do paciente, em forma estruturada e em texto."""

    prontuario: str
    leito: str
    unidade: str
    idade: int | None
    sexo: str
    motivo_internacao: str
    minutos_internado: int | None
    comorbidades: list[str] = field(default_factory=list)
    alergias: list[str] = field(default_factory=list)
    medicacoes: list[str] = field(default_factory=list)
    protocolos: list[dict[str, Any]] = field(default_factory=list)
    sinais: list[dict[str, Any]] = field(default_factory=list)
    exames_liberados: list[dict[str, Any]] = field(default_factory=list)
    exames_pendentes: list[dict[str, Any]] = field(default_factory=list)
    eventos: list[dict[str, Any]] = field(default_factory=list)
    nome_exibicao: str = ""

    @property
    def identificacao_segura(self) -> str:
        return f"leito {self.leito} (prontuario {self.prontuario})"

    def ultimo_sinal(self) -> dict[str, Any] | None:
        return self.sinais[0] if self.sinais else None

    def evento(self, tipo: str) -> dict[str, Any] | None:
        for evento in self.eventos:
            if evento["tipo"] == tipo:
                return evento
        return None

    def tem_protocolo(self, identificador: str) -> bool:
        return any(p["protocolo"] == identificador for p in self.protocolos)

    def como_texto(self) -> str:
        """Serializa o contexto para o prompt, sem identificadores diretos."""
        idade = f"{self.idade} anos" if self.idade is not None else "idade nao informada"
        sexo = SEXO_EXTENSO.get(self.sexo, "nao informado")
        linhas = [
            f"Paciente do {self.identificacao_segura}, {idade}, sexo {sexo}, "
            f"unidade {self.unidade}.",
            f"Internado {formatar_intervalo(self.minutos_internado)}. "
            f"Motivo da internacao: {self.motivo_internacao}.",
            f"Comorbidades: {'; '.join(self.comorbidades) if self.comorbidades else 'nenhuma registrada'}.",
            f"Alergias registradas: {'; '.join(self.alergias) if self.alergias else 'nenhuma registrada'}.",
            f"Medicacoes em uso: {'; '.join(self.medicacoes) if self.medicacoes else 'nenhuma registrada'}.",
        ]

        if self.protocolos:
            ativos = "; ".join(
                f"{p['protocolo']} (aberto {formatar_intervalo(minutos_desde(p.get('aberto_em')))})"
                for p in self.protocolos
            )
            linhas.append(f"Protocolos ativos: {ativos}.")
        else:
            linhas.append("Protocolos ativos: nenhum.")

        sinal = self.ultimo_sinal()
        if sinal:
            linhas.append(
                "Sinais vitais mais recentes ("
                f"{formatar_intervalo(minutos_desde(sinal.get('aferido_em')))}): "
                f"PA {_numero(sinal.get('pas'))}x{_numero(sinal.get('pad'))} mmHg, "
                f"FC {_numero(sinal.get('fc'))} bpm, "
                f"FR {_numero(sinal.get('fr'))} irpm, "
                f"Tax {_numero(sinal.get('temperatura'))} C, "
                f"SatO2 {_numero(sinal.get('saturacao'))}%, "
                f"Glasgow {_numero(sinal.get('glasgow'))}."
            )

        if self.exames_liberados:
            itens = "; ".join(
                f"{e['nome']} = {e.get('resultado') or 'sem resultado'}"
                + (f" {e['unidade']}"
                   if e.get("unidade") and e["unidade"] not in (e.get("resultado") or "")
                   else "")
                + (f" (referencia {e['referencia']})" if e.get("referencia") else "")
                for e in self.exames_liberados
            )
            linhas.append(f"Exames liberados: {itens}.")

        if self.exames_pendentes:
            itens = "; ".join(
                f"{e['nome']}"
                + (" [critico]" if e.get("critico") else "")
                + f" solicitado {formatar_intervalo(minutos_desde(e.get('solicitado_em')))}"
                for e in self.exames_pendentes
            )
            linhas.append(f"Exames pendentes: {itens}.")
        else:
            linhas.append("Exames pendentes: nenhum.")

        if self.eventos:
            itens = "; ".join(
                f"{e['tipo']}: {'registrado ' + formatar_intervalo(minutos_desde(e.get('registrado_em'))) if e.get('ocorreu') else 'NAO registrado'}"
                for e in self.eventos
            )
            linhas.append(f"Marcos assistenciais: {itens}.")

        return "\n".join(linhas)

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "prontuario": self.prontuario,
            "leito": self.leito,
            "unidade": self.unidade,
            "idade": self.idade,
            "sexo": self.sexo,
            "motivo_internacao": self.motivo_internacao,
            "comorbidades": self.comorbidades,
            "alergias": self.alergias,
            "medicacoes": self.medicacoes,
            "protocolos": [p["protocolo"] for p in self.protocolos],
            "exames_pendentes": [e["nome"] for e in self.exames_pendentes],
            "exames_liberados": [
                {"nome": e["nome"], "resultado": e.get("resultado")} for e in self.exames_liberados
            ],
        }

    def campos_utilizados(self) -> list[str]:
        """Campos do prontuario usados na resposta — alimenta a explicabilidade."""
        campos = ["pacientes.leito", "pacientes.prontuario", "pacientes.data_nascimento",
                  "pacientes.unidade", "pacientes.motivo_internacao"]
        if self.comorbidades:
            campos.append("comorbidades.descricao")
        if self.alergias:
            campos.append("alergias.agente")
        if self.medicacoes:
            campos.append("medicacoes.descricao")
        if self.protocolos:
            campos.append("protocolos_ativos.protocolo")
        if self.sinais:
            campos.append("sinais_vitais")
        if self.exames_liberados or self.exames_pendentes:
            campos.append("exames.status")
        return campos


def montar_contexto(banco: BancoHospital, prontuario: str) -> ContextoPaciente | None:
    """Le o prontuario no banco e monta o contexto minimizado."""
    paciente = banco.obter_paciente(prontuario)
    if paciente is None:
        return None

    exames = banco.exames(prontuario)
    return ContextoPaciente(
        prontuario=paciente["prontuario"],
        leito=paciente.get("leito") or "sem leito",
        unidade=paciente.get("unidade") or "unidade nao informada",
        idade=calcular_idade(paciente.get("data_nascimento")),
        sexo=paciente.get("sexo") or "",
        motivo_internacao=paciente.get("motivo_internacao") or "nao informado",
        minutos_internado=minutos_desde(paciente.get("data_admissao")),
        comorbidades=banco.comorbidades(prontuario),
        alergias=banco.alergias(prontuario),
        medicacoes=banco.medicacoes(prontuario),
        protocolos=banco.protocolos_ativos(prontuario),
        sinais=banco.sinais_vitais(prontuario),
        exames_liberados=[e for e in exames if e["status"] == "liberado"],
        exames_pendentes=[e for e in exames if e["status"] == "pendente"],
        eventos=banco.eventos(prontuario),
        nome_exibicao=paciente.get("nome") or "",
    )


def resolver_prontuario(banco: BancoHospital, termo: str) -> str | None:
    """Resolve um termo livre (leito, nome ou prontuario) em um numero de prontuario."""
    encontrados = banco.buscar_paciente(termo)
    if len(encontrados) == 1:
        return encontrados[0]["prontuario"]
    return None
