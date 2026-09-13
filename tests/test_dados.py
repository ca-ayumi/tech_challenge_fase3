"""Testes do pipeline de dados: preprocessamento, curadoria, banco e contexto."""

from __future__ import annotations

import pytest

from assistente_medico.dados.contexto_paciente import (
    calcular_idade,
    formatar_intervalo,
    montar_contexto,
)
from assistente_medico.dados.curadoria import curar, detectar_pii_residual, dividir, jaccard
from assistente_medico.dados.preprocessamento import carregar_documentos, dividir_em_trechos
from assistente_medico.finetuning.formato import montar_resposta


# ------------------------------------------------------------ preprocessamento
def test_documentos_sao_carregados_com_metadados():
    documentos = carregar_documentos()
    assert len(documentos) >= 10
    protocolos = [d for d in documentos if d.tipo == "protocolo"]
    assert len(protocolos) >= 10
    for documento in documentos:
        assert documento.identificador.startswith(("PROT-", "MOD-", "POP-"))
        assert documento.titulo


def test_trechos_tem_referencia_citavel(trechos):
    assert len(trechos) > 50
    for trecho in trechos:
        assert "§" in trecho.referencia
        assert trecho.referencia.startswith(trecho.documento)
        assert len(trecho.texto) >= 40


def test_referencias_sao_unicas(trechos):
    referencias = [t.referencia for t in trechos]
    assert len(referencias) == len(set(referencias))


def test_divisao_respeita_secoes_numeradas():
    documento = next(d for d in carregar_documentos() if d.identificador == "PROT-SEP-001")
    trechos = dividir_em_trechos(documento)
    secoes = [t.secao for t in trechos]
    assert "3" in secoes and "4" in secoes
    quarta = next(t for t in trechos if t.secao == "4")
    assert "primeira hora" in quarta.titulo_secao.lower()
    assert "60 minutos" in quarta.texto


# -------------------------------------------------------------------- curadoria
def test_jaccard_de_textos_identicos_e_um():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_curadoria_remove_duplicata_exata():
    exemplo = {
        "categoria": "duvida_protocolo",
        "pergunta": "Quais os criterios de abertura do protocolo de sepse?",
        "resposta": montar_resposta("Foco infeccioso e dois sinais de alerta presentes no "
                                    "paciente adulto.", ["PROT-SEP-001 §3"]),
        "fontes": ["PROT-SEP-001 §3"],
    }
    aprovados, relatorio = curar([exemplo, dict(exemplo)])
    assert len(aprovados) == 1
    assert relatorio.duplicatas_exatas == 1


def test_curadoria_descarta_exemplo_com_pii_residual():
    exemplo = {
        "categoria": "duvida_protocolo",
        "pergunta": "Como conduzir esse paciente internado na enfermaria?",
        "resposta": montar_resposta("Contato do responsavel: (11) 98123-4455 para aviso.",
                                    ["PROT-SEP-001 §3"]),
        "fontes": ["PROT-SEP-001 §3"],
    }
    aprovados, relatorio = curar([exemplo])
    assert aprovados == []
    assert relatorio.descartados["pii_residual"] == 1
    assert relatorio.pii_residual["telefone"] == 1


def test_curadoria_descarta_formato_invalido():
    exemplo = {
        "categoria": "duvida_protocolo",
        "pergunta": "Qual o criterio de abertura do protocolo de sepse aqui?",
        "resposta": "Resposta sem a estrutura institucional de tres blocos exigida pelo projeto.",
        "fontes": ["PROT-SEP-001 §3"],
    }
    aprovados, relatorio = curar([exemplo])
    assert aprovados == []
    assert relatorio.descartados["formato_invalido"] == 1


def test_detecta_pii_residual():
    assert detectar_pii_residual("CPF 123.456.789-00") == ["cpf"]
    assert detectar_pii_residual("texto limpo sem identificador") == []


def test_divisao_e_estratificada_e_deterministica():
    exemplos = [
        {"categoria": "a", "pergunta": f"p{i}", "resposta": f"r{i}"} for i in range(20)
    ] + [
        {"categoria": "b", "pergunta": f"q{i}", "resposta": f"s{i}"} for i in range(20)
    ]
    primeira = dividir(exemplos, semente=42)
    segunda = dividir(exemplos, semente=42)
    assert [e["pergunta"] for e in primeira["treino"]] == [e["pergunta"] for e in segunda["treino"]]

    total = sum(len(p) for p in primeira.values())
    assert total == len(exemplos), "nenhum exemplo pode se perder na divisao"
    assert primeira["validacao"] and primeira["teste"]
    # As duas categorias precisam aparecer em treino.
    assert {e["categoria"] for e in primeira["treino"]} == {"a", "b"}


# ------------------------------------------------------------------- contexto
def test_idade_calculada_em_anos_completos():
    from datetime import datetime

    assert calcular_idade("1958-03-12", datetime(2026, 3, 11)) == 67
    assert calcular_idade("1958-03-12", datetime(2026, 3, 12)) == 68
    assert calcular_idade(None) is None


@pytest.mark.parametrize(
    "minutos, esperado",
    [(0, "ha 0 min"), (45, "ha 45 min"), (90, "ha 1h30min"), (60 * 26, "ha 1d2h"), (None, "sem registro de horario")],
)
def test_formatacao_de_intervalo(minutos, esperado):
    assert formatar_intervalo(minutos) == esperado


def test_contexto_nao_expoe_identificadores_diretos(banco, prontuarios):
    paciente = prontuarios[0]
    contexto = montar_contexto(banco, paciente["prontuario"])
    texto = contexto.como_texto()

    assert paciente["nome"] not in texto
    assert paciente["cpf"] not in texto
    assert paciente["telefone"] not in texto
    assert paciente["endereco"] not in texto
    # Mas os dados clinicos precisam estar la.
    assert contexto.leito in texto
    assert "Exames pendentes" in texto
    assert str(contexto.idade) in texto


def test_contexto_lista_campos_usados(banco):
    contexto = montar_contexto(banco, "884521")
    campos = contexto.campos_utilizados()
    assert "pacientes.leito" in campos
    assert "exames.status" in campos


def test_prontuario_inexistente_retorna_none(banco):
    assert montar_contexto(banco, "000000") is None


# ----------------------------------------------------------------------- banco
def test_banco_populado(banco):
    assert len(banco.listar_pacientes()) == 12


def test_busca_por_leito_nome_e_prontuario(banco):
    assert banco.buscar_paciente("PS-07")[0]["prontuario"] == "884521"
    assert banco.buscar_paciente("884521")[0]["leito"] == "PS-07"
    assert banco.buscar_paciente("Maria Aparecida")[0]["prontuario"] == "884521"
    assert banco.buscar_paciente("inexistente") == []


def test_consulta_somente_leitura_aceita_select(banco):
    linhas = banco.consultar_somente_leitura("SELECT leito FROM pacientes WHERE leito = 'PS-07'")
    assert linhas == [{"leito": "PS-07"}]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM pacientes",
        "UPDATE pacientes SET nome = 'x'",
        "SELECT * FROM auditoria",
        "SELECT * FROM validacoes",
        "DROP TABLE pacientes",
    ],
)
def test_consulta_somente_leitura_bloqueia_escrita_e_governanca(banco, sql):
    with pytest.raises(ValueError):
        banco.consultar_somente_leitura(sql)


def test_ciclo_de_validacao_humana(banco):
    identificador = banco.registrar_validacao(
        interacao="INT-TESTE", sugestao="Sugestao de conduta.", prontuario="884521",
        fontes=["PROT-SEP-001 §4"],
    )
    pendentes = [v for v in banco.listar_validacoes("pendente") if v["id"] == identificador]
    assert pendentes and pendentes[0]["status"] == "pendente"

    banco.decidir_validacao(identificador, "aceita", "Dra. Teste (CRM-SP 00.000)")
    aceitas = [v for v in banco.listar_validacoes("aceita") if v["id"] == identificador]
    assert aceitas and aceitas[0]["profissional"].startswith("Dra. Teste")

    with pytest.raises(ValueError):
        banco.decidir_validacao(identificador, "status_invalido", "x")
