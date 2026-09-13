"""Fixtures compartilhadas pelos testes.

Os testes nunca tocam o banco de producao: cada sessao recebe um SQLite
temporario populado com os mesmos prontuarios sinteticos do repositorio.
"""

from __future__ import annotations

import pytest

from assistente_medico.dados.anonimizacao import Anonimizador
from assistente_medico.dados.banco import BancoHospital, carregar_prontuarios_brutos
from assistente_medico.dados.preprocessamento import construir_trechos
from assistente_medico.rag.recuperador import RecuperadorProtocolos


@pytest.fixture(scope="session")
def prontuarios() -> list[dict]:
    return carregar_prontuarios_brutos()


@pytest.fixture(scope="session")
def banco(tmp_path_factory, prontuarios) -> BancoHospital:
    caminho = tmp_path_factory.mktemp("banco") / "hospital_teste.db"
    instancia = BancoHospital(caminho)
    instancia.popular(prontuarios)
    return instancia


@pytest.fixture(scope="session")
def trechos():
    return construir_trechos()


@pytest.fixture(scope="session")
def recuperador(trechos) -> RecuperadorProtocolos:
    return RecuperadorProtocolos(trechos)


@pytest.fixture
def anonimizador(prontuarios) -> Anonimizador:
    return Anonimizador(sal="sal-de-teste",
                        nomes_conhecidos=[p["nome"] for p in prontuarios])


@pytest.fixture
def assistente(banco, recuperador, tmp_path):
    """Assistente completo com backend deterministico, sem carregar modelo neural."""
    from assistente_medico.auditoria import Auditoria
    from assistente_medico.grafo import AssistenteMedico, Dependencias
    from assistente_medico.llm import criar_modelo_chat

    dependencias = Dependencias(
        banco=banco,
        recuperador=recuperador,
        modelo=criar_modelo_chat("eco"),
        auditoria=Auditoria(banco=banco, caminho_jsonl=tmp_path / "auditoria.jsonl"),
    )
    return AssistenteMedico(dependencias)
