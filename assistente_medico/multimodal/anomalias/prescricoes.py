"""Alteracoes inesperadas na evolucao das prescricoes.

Cada nova prescricao (ou suspensao) chega como leitura e e comparada com o
historico do proprio paciente:

- conflito com alergia registrada, inclusive por grupo (alergia a penicilina x
  amoxicilina), o que a regra de texto da Fase 3 nao enxergava;
- salto de dose diaria em relacao a prescricao anterior do mesmo medicamento:
  razao proxima de 10 sugere erro de digitacao (virgula ou zero a mais);
- dose diaria acima do limite da tabela institucional de referencia;
- duplicidade terapeutica (dois anticoagulantes ativos ao mesmo tempo);
- suspensao de medicamento critico sem outro da mesma classe ativo.

As mensagens descrevem o que mudou no historico; nao recomendam conduta nem
dose (PROT-GOV-010).
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ...config import CAMINHOS
from ..tempo_real import Leitura
from ..tipos import Anomalia


def normalizar(texto: str | None) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto or "")
    return " ".join("".join(c for c in sem_acento if not unicodedata.combining(c)).lower().split())


def carregar_referencia(caminho: Path | None = None) -> dict[str, dict[str, Any]]:
    caminho = Path(caminho or CAMINHOS.dados_brutos / "monitoramento" / "referencia_medicamentos.json")
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return {normalizar(nome): ficha for nome, ficha in dados["medicamentos"].items()}


@dataclass(frozen=True)
class ParametrosPrescricao:
    razao_brusca: float = 2.0
    razao_erro_digitacao: float = 5.0
    janela_brusca: timedelta = timedelta(hours=24)


@dataclass
class _Prescricao:
    medicamento: str
    dose_diaria: float | None
    dose: float | None
    unidade: str | None
    instante: datetime


@dataclass
class _Estado:
    ativas: dict[str, _Prescricao] = field(default_factory=dict)
    ultimas: dict[str, _Prescricao] = field(default_factory=dict)


def dose_diaria(dose: float | None, intervalo_horas: float | None) -> float | None:
    if dose is None:
        return None
    if not intervalo_horas:
        return float(dose)
    return float(dose) * 24.0 / float(intervalo_horas)


def _fmt(valor: float | None) -> str:
    if valor is None:
        return "?"
    return f"{valor:g}"


class DetectorPrescricoes:
    nome = "prescricao"
    modalidades = ("prescricao",)

    def __init__(self, alergias: Mapping[str, Sequence[str]] | None = None,
                 referencia: dict[str, dict[str, Any]] | None = None,
                 parametros: ParametrosPrescricao | None = None) -> None:
        self.alergias = {p: [normalizar(a) for a in lista] for p, lista in (alergias or {}).items()}
        self.referencia = referencia if referencia is not None else carregar_referencia()
        self.parametros = parametros or ParametrosPrescricao()
        self._estados: dict[str, _Estado] = {}

    @classmethod
    def do_banco(cls, banco, **kwargs: Any) -> DetectorPrescricoes:
        alergias = {p["prontuario"]: banco.alergias(p["prontuario"])
                    for p in banco.listar_pacientes()}
        return cls(alergias=alergias, **kwargs)

    def _ficha(self, medicamento: str) -> dict[str, Any]:
        return self.referencia.get(medicamento, {})

    def _anomalia(self, leitura: Leitura, tipo: str, severidade: str, descricao: str,
                  evidencias: dict[str, Any], escore: float = 1.0) -> Anomalia:
        instante = leitura.instante.isoformat(timespec="seconds")
        return Anomalia(
            modalidade="prescricao", tipo=tipo, severidade=severidade, descricao=descricao,
            prontuario=leitura.prontuario, inicio=instante, fim=instante,
            escore=round(escore, 3), detector="historico_prescricao", evidencias=evidencias,
        )

    def atualizar(self, leitura: Leitura) -> list[Anomalia]:
        valores = leitura.valores
        estado = self._estados.setdefault(leitura.prontuario, _Estado())
        medicamento = normalizar(valores.get("medicamento"))
        if not medicamento:
            return []
        nome = valores.get("medicamento")
        if valores.get("situacao", "ativa") != "ativa":
            return self._suspensao(leitura, estado, medicamento, nome)

        atual = _Prescricao(
            medicamento=medicamento,
            dose_diaria=dose_diaria(valores.get("dose"), valores.get("intervalo_horas")),
            dose=valores.get("dose"), unidade=valores.get("unidade"), instante=leitura.instante,
        )
        achados = []
        achados.extend(self._alergia(leitura, medicamento, nome))
        achados.extend(self._salto(leitura, estado.ultimas.get(medicamento), atual, nome))
        achados.extend(self._maximo(leitura, atual, nome))
        achados.extend(self._duplicidade(leitura, estado, medicamento, nome))
        estado.ativas[medicamento] = atual
        estado.ultimas[medicamento] = atual
        return achados

    def _alergia(self, leitura: Leitura, medicamento: str, nome: str) -> list[Anomalia]:
        grupos = {medicamento, *(normalizar(g) for g in self._ficha(medicamento).get("grupos_alergenicos", []))}
        for alergia in self.alergias.get(leitura.prontuario, []):
            if not alergia:
                continue
            if any(alergia in grupo or grupo in alergia for grupo in grupos if grupo):
                return [self._anomalia(
                    leitura, "conflito_alergia", "critica",
                    f"Prescricao de {nome} para paciente com alergia registrada a '{alergia}'",
                    {"medicamento": nome, "alergia": alergia, "grupos": sorted(grupos)},
                )]
        return []

    def _salto(self, leitura: Leitura, anterior: _Prescricao | None, atual: _Prescricao,
               nome: str) -> list[Anomalia]:
        p = self.parametros
        if (anterior is None or not anterior.dose_diaria or not atual.dose_diaria
                or anterior.unidade != atual.unidade):
            return []
        razao = atual.dose_diaria / anterior.dose_diaria
        maior = max(razao, 1 / razao)
        evidencias = {"medicamento": nome, "dose_diaria_anterior": anterior.dose_diaria,
                      "dose_diaria_nova": atual.dose_diaria, "unidade": atual.unidade,
                      "razao": round(razao, 2), "prescricao_anterior": anterior.instante.isoformat()}
        sentido = "aumento" if razao > 1 else "reducao"
        if maior >= p.razao_erro_digitacao:
            suspeita = " (possivel erro de digitacao: zero ou virgula)" if 8 <= maior <= 12 else ""
            return [self._anomalia(
                leitura, "salto_dose", "critica",
                f"{nome}: {sentido} de {maior:.0f}x na dose diaria em relacao a prescricao "
                f"anterior ({_fmt(anterior.dose_diaria)} -> {_fmt(atual.dose_diaria)} "
                f"{atual.unidade or ''}/dia){suspeita}",
                evidencias,
            )]
        titulavel = self._ficha(atual.medicamento).get("titulavel", False)
        recente = atual.instante - anterior.instante <= p.janela_brusca
        if maior >= p.razao_brusca and recente and not titulavel:
            return [self._anomalia(
                leitura, "mudanca_brusca_dose", "alta",
                f"{nome}: {sentido} de {maior:.1f}x na dose diaria em menos de "
                f"{int(p.janela_brusca.total_seconds() // 3600)} h",
                evidencias, escore=min(maior / p.razao_erro_digitacao, 1.0),
            )]
        return []

    def _maximo(self, leitura: Leitura, atual: _Prescricao, nome: str) -> list[Anomalia]:
        ficha = self._ficha(atual.medicamento)
        limite = ficha.get("dose_maxima_diaria")
        if not limite or atual.dose_diaria is None or atual.unidade != ficha.get("unidade"):
            return []
        if atual.dose_diaria <= limite:
            return []
        return [self._anomalia(
            leitura, "acima_referencia", "alta",
            f"{nome}: dose diaria prescrita ({_fmt(atual.dose_diaria)} {atual.unidade}/dia) acima "
            "do limite da tabela institucional de referencia",
            {"medicamento": nome, "dose_diaria": atual.dose_diaria, "limite": limite,
             "unidade": atual.unidade},
        )]

    def _duplicidade(self, leitura: Leitura, estado: _Estado, medicamento: str,
                     nome: str) -> list[Anomalia]:
        ficha = self._ficha(medicamento)
        if not ficha.get("sem_duplicidade"):
            return []
        conflitantes = [
            outra for chave, outra in estado.ativas.items()
            if chave != medicamento and self._ficha(chave).get("classe") == ficha.get("classe")
        ]
        if not conflitantes:
            return []
        outros = ", ".join(sorted(c.medicamento for c in conflitantes))
        return [self._anomalia(
            leitura, "duplicidade_terapeutica", "alta",
            f"Prescricao de {nome} com outro {ficha['classe']} ainda ativo ({outros})",
            {"medicamento": nome, "classe": ficha["classe"], "ativos": outros},
        )]

    def _suspensao(self, leitura: Leitura, estado: _Estado, medicamento: str,
                   nome: str) -> list[Anomalia]:
        removida = estado.ativas.pop(medicamento, None)
        ficha = self._ficha(medicamento)
        if removida is None or not ficha.get("suspensao_critica"):
            return []
        mesma_classe = [c for c in estado.ativas if self._ficha(c).get("classe") == ficha.get("classe")]
        if mesma_classe:
            return []
        duracao = leitura.instante - removida.instante
        return [self._anomalia(
            leitura, "suspensao_sem_substituto", "moderada",
            f"Suspensao de {nome} ({ficha.get('classe')}) sem outro medicamento da mesma classe "
            "ativo; confirmar se foi intencional",
            {"medicamento": nome, "classe": ficha.get("classe"),
             "horas_em_uso": round(duracao.total_seconds() / 3600, 1)},
        )]
