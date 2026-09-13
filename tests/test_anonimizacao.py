"""Testes do pipeline de anonimizacao."""

from __future__ import annotations

import pytest

from assistente_medico.dados.anonimizacao import Anonimizador, mascarar_para_log


@pytest.mark.parametrize(
    "texto, rotulo",
    [
        ("CPF 321.654.987-00", "[CPF]"),
        ("CNS 700 5012 3344 8877", "[CNS]"),
        ("telefone (11) 97744-2210", "[TELEFONE]"),
        ("e-mail fulano.teste@provedorexemplo.com.br", "[EMAIL]"),
        ("nascida em 12/03/1958", "[DATA]"),
        ("mora na Rua das Acacias, 421", "[ENDERECO]"),
        ("RG 28.445.120-7", "[RG]"),
    ],
)
def test_remove_identificadores_diretos(anonimizador, texto, rotulo):
    assert rotulo in anonimizador.anonimizar_texto(texto)


def test_pseudonimo_e_estavel_e_nao_reversivel(anonimizador):
    primeiro = anonimizador.pseudonimo("nome_paciente", "Maria Aparecida da Silva")
    segundo = anonimizador.pseudonimo("nome_paciente", "maria aparecida da silva")
    assert primeiro == segundo, "o mesmo nome deve gerar sempre o mesmo codigo"
    assert "Maria" not in primeiro
    assert primeiro.startswith("[PACIENTE_")


def test_sal_diferente_gera_pseudonimo_diferente():
    um = Anonimizador(sal="sal-a").pseudonimo("nome_paciente", "Joao Batista")
    outro = Anonimizador(sal="sal-b").pseudonimo("nome_paciente", "Joao Batista")
    assert um != outro


def test_nome_conhecido_e_pseudonimizado(anonimizador):
    resultado = anonimizador.anonimizar("Paciente Maria Aparecida da Silva admitida hoje.")
    assert "Maria" not in resultado.texto
    assert resultado.contagem_por_tipo.get("nome_paciente") == 1


def test_nao_confunde_texto_clinico_com_nome_proprio(anonimizador):
    texto = ("Posicionar o paciente em decubito dorsal com membros inferiores elevados. "
             "Aplica-se a pacientes com 18 anos ou mais.")
    assert anonimizador.anonimizar_texto(texto) == texto


def test_profissional_com_titulo_e_pseudonimizado(anonimizador):
    resultado = anonimizador.anonimizar("Responsavel: Dr. Ricardo Almeida Sobral (CRM-SP 118.442)")
    assert "[PROFISSIONAL_" in resultado.texto
    assert "[CRM_" in resultado.texto
    assert "Ricardo" not in resultado.texto


def test_anonimiza_estrutura_aninhada(anonimizador, prontuarios):
    tratado = anonimizador.anonimizar_estrutura(prontuarios[0])
    assert tratado["nome"].startswith("[PACIENTE_")
    assert tratado["cpf"] == "[CPF]"
    # Dados clinicos nao podem ser perdidos no processo.
    assert tratado["comorbidades"] == prontuarios[0]["comorbidades"]
    assert tratado["exames"][0]["nome"] == prontuarios[0]["exames"][0]["nome"]


def test_evolucoes_perdem_toda_pii(anonimizador, prontuarios):
    import re

    for prontuario in prontuarios:
        for evolucao in prontuario.get("evolucoes", []):
            tratado = anonimizador.anonimizar_texto(evolucao["texto"])
            assert not re.search(r"\d{3}\.\d{3}\.\d{3}-\d{2}", tratado)
            assert not re.search(r"\(\d{2}\)\s?9?\d{4}-\d{4}", tratado)
            assert prontuario["nome"] not in tratado


def test_mascarar_para_log_usa_padrao():
    assert "[CPF]" in mascarar_para_log("CPF 123.456.789-00")
