"""Testes das ferramentas LangChain sobre a base estruturada.

As ferramentas sao a unica porta pela qual o assistente toca dados do hospital,
e por isso concentram tanto o contrato de leitura (o que cada uma devolve) quanto
as barreiras de escrita e de consulta livre.
"""

from __future__ import annotations

import json
import re

import pytest

from assistente_medico.ferramentas import criar_ferramentas, ferramentas_de_leitura

NOMES_ESPERADOS = {
    "buscar_paciente",
    "consultar_prontuario",
    "listar_exames_pendentes",
    "verificar_pendencias_de_protocolo",
    "consultar_protocolo",
    "consultar_base_estruturada",
    "registrar_alerta_equipe",
    "solicitar_validacao_humana",
}


@pytest.fixture
def ferramentas(banco, recuperador):
    return criar_ferramentas(banco, recuperador, interacao="INT-TESTE-0001",
                             usuario="dra.helena")


@pytest.fixture
def por_nome(ferramentas):
    return {ferramenta.name: ferramenta for ferramenta in ferramentas}


# --------------------------------------------------------------- catalogo
def test_expoe_exatamente_as_oito_ferramentas(por_nome):
    assert set(por_nome) == NOMES_ESPERADOS


def test_toda_ferramenta_tem_descricao_para_o_modelo(ferramentas):
    for ferramenta in ferramentas:
        assert ferramenta.description, f"{ferramenta.name} sem descricao"
        assert len(ferramenta.description) > 50


def test_ferramentas_de_escrita_estao_marcadas(por_nome):
    escrita = {"registrar_alerta_equipe", "solicitar_validacao_humana"}
    for nome, ferramenta in por_nome.items():
        tipo = (ferramenta.metadata or {}).get("tipo_acesso")
        assert tipo == ("escrita" if nome in escrita else "leitura")


def test_filtro_de_leitura_exclui_as_de_efeito_colateral(ferramentas):
    somente_leitura = ferramentas_de_leitura(ferramentas)
    assert len(somente_leitura) == 6
    assert "registrar_alerta_equipe" not in {f.name for f in somente_leitura}


# ---------------------------------------------------------------- leitura
@pytest.mark.parametrize("termo", ["PS-07", "884521", "Maria"])
def test_busca_paciente_por_leito_prontuario_e_nome(por_nome, termo):
    resultado = por_nome["buscar_paciente"].invoke({"termo": termo})
    assert "884521" in resultado


def test_busca_sem_resultado_explica_em_vez_de_quebrar(por_nome):
    resultado = por_nome["buscar_paciente"].invoke({"termo": "ZZ-99"})
    assert "Nenhum paciente encontrado" in resultado


def test_prontuario_consultado_nao_expoe_identificadores_diretos(por_nome, prontuarios):
    resultado = por_nome["consultar_prontuario"].invoke({"prontuario": "884521"})
    paciente = next(p for p in prontuarios if p["prontuario"] == "884521")
    assert "leito PS-07" in resultado
    for campo in ("nome", "cpf", "cns", "telefone", "endereco"):
        valor = paciente.get(campo)
        if valor:
            assert valor not in resultado, f"{campo} vazou no contexto enviado ao modelo"


def test_prontuario_inexistente_devolve_mensagem(por_nome):
    assert "nao encontrado" in por_nome["consultar_prontuario"].invoke(
        {"prontuario": "000000"})


def test_exames_pendentes_marcam_os_criticos(por_nome):
    resultado = por_nome["listar_exames_pendentes"].invoke({"prontuario": "884521"})
    exames = json.loads(resultado)
    assert exames
    assert any(exame["critico"] for exame in exames)
    assert all({"exame", "codigo", "critico", "solicitado_em"} <= set(e) for e in exames)


def test_pendencias_de_protocolo_citam_a_fonte(por_nome):
    resultado = por_nome["verificar_pendencias_de_protocolo"].invoke(
        {"prontuario": "884521"})
    alertas = json.loads(resultado)
    assert alertas
    for alerta in alertas:
        assert alerta["fonte"].startswith("PROT-")
        assert alerta["prioridade"] in {"maxima", "alta", "media", "informativa"}


def test_consulta_de_protocolo_devolve_referencia_citavel(por_nome):
    resultado = por_nome["consultar_protocolo"].invoke(
        {"consulta": "pacote da primeira hora na sepse"})
    assert "PROT-SEP-001" in resultado
    assert "§" in resultado


# ------------------------------------------------- consulta livre (SQL)
def test_sql_de_leitura_e_aceito(por_nome):
    resultado = por_nome["consultar_base_estruturada"].invoke(
        {"sql": "SELECT leito, unidade FROM pacientes"})
    linhas = json.loads(resultado)
    assert linhas and "leito" in linhas[0]


@pytest.mark.parametrize("sql", [
    "DELETE FROM pacientes",
    "UPDATE pacientes SET nome = 'x'",
    "INSERT INTO alergias (prontuario, agente) VALUES ('1', 'x')",
    "DROP TABLE exames",
    "SELECT * FROM pacientes; DROP TABLE exames",
])
def test_sql_de_escrita_e_recusado(por_nome, sql):
    resultado = por_nome["consultar_base_estruturada"].invoke({"sql": sql})
    assert resultado.startswith("Consulta recusada:")


@pytest.mark.parametrize("tabela", ["auditoria", "validacoes"])
def test_sql_nao_alcanca_as_tabelas_de_governanca(por_nome, tabela):
    resultado = por_nome["consultar_base_estruturada"].invoke(
        {"sql": f"SELECT * FROM {tabela}"})
    assert resultado.startswith("Consulta recusada:")


def test_consulta_sem_resultado_e_informada(por_nome):
    resultado = por_nome["consultar_base_estruturada"].invoke(
        {"sql": "SELECT leito FROM pacientes WHERE unidade = 'inexistente'"})
    assert "nao retornou linhas" in resultado


# ---------------------------------------------------------------- escrita
def test_alerta_registrado_chega_ao_painel(por_nome, banco):
    antes = len(banco.listar_alertas(prontuario="884521"))
    resultado = por_nome["registrar_alerta_equipe"].invoke({
        "prontuario": "884521", "prioridade": "alta",
        "mensagem": "Lactato pendente acima do prazo.", "fonte": "PROT-SEP-001 §4",
    })
    assert "registrado" in resultado

    identificador = int(re.search(r"\d+", resultado).group())
    depois = banco.listar_alertas(prontuario="884521")
    assert len(depois) == antes + 1

    registrado = next(a for a in depois if a["id"] == identificador)
    assert registrado["interacao"] == "INT-TESTE-0001"
    assert registrado["prioridade"] == "alta"
    assert registrado["fonte"] == "PROT-SEP-001 §4"


def test_prioridade_invalida_e_recusada_sem_gravar(por_nome, banco):
    antes = len(banco.listar_alertas(prontuario="884521"))
    resultado = por_nome["registrar_alerta_equipe"].invoke({
        "prontuario": "884521", "prioridade": "urgentissima", "mensagem": "x",
    })
    assert "Prioridade invalida" in resultado
    assert len(banco.listar_alertas(prontuario="884521")) == antes


def test_validacao_humana_abre_pendente(por_nome, banco):
    resultado = por_nome["solicitar_validacao_humana"].invoke({
        "sugestao": "Reavaliar antimicrobiano conforme protocolo.",
        "prontuario": "884521", "fontes": "PROT-ATB-003 §3; PROT-GOV-010 §5",
    })
    assert "pendente" in resultado

    identificador = int(re.search(r"\d+", resultado).group())
    aberta = next(v for v in banco.listar_validacoes(status="pendente")
                  if v["id"] == identificador)
    assert aberta["interacao"] == "INT-TESTE-0001"
    assert aberta["prontuario"] == "884521"
    assert json.loads(aberta["fontes"]) == ["PROT-ATB-003 §3", "PROT-GOV-010 §5"]


def test_nenhuma_ferramenta_prescreve_ou_calcula_dose(por_nome):
    proibidos = ("prescrev", "dose", "posologia", "receita", "autoriza_alta")
    for nome in por_nome:
        assert not any(termo in nome for termo in proibidos)
