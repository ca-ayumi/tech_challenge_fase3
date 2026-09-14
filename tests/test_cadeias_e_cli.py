"""Testes das cadeias LCEL e da interface de linha de comando.

As cadeias sao a composicao declarativa do pipeline em LangChain — o caminho
alternativo ao grafo. A CLI e a porta pela qual o pipeline inteiro e operado.
Ambas rodam com o backend deterministico, sem carregar modelo neural.
"""

from __future__ import annotations

import pytest

from assistente_medico import cli
from assistente_medico.cadeias import (
    criar_cadeia_consulta,
    criar_cadeia_recuperacao,
    criar_cadeia_simples,
)
from assistente_medico.llm import criar_modelo_chat
from assistente_medico.seguranca.guardrails import Guardrails


@pytest.fixture(scope="module")
def modelo():
    return criar_modelo_chat("eco")


# ------------------------------------------------------------- cadeias
def test_cadeia_de_recuperacao_monta_os_dois_contextos(modelo, banco, recuperador):
    cadeia = criar_cadeia_recuperacao(recuperador, banco)
    saida = cadeia.invoke({
        "pergunta": "Qual o prazo do pacote da primeira hora na sepse?",
        "prontuario": "884521",
    })
    assert "PROT-SEP-001" in saida["contexto_documentos"]
    assert "leito PS-07" in saida["contexto_paciente"]
    assert saida["recuperados"]
    assert saida["prompt_sistema"].startswith("Voce e o Assistente")


def test_cadeia_sem_paciente_informa_ausencia(modelo, banco, recuperador):
    cadeia = criar_cadeia_recuperacao(recuperador, banco)
    saida = cadeia.invoke({"pergunta": "Quais criterios abrem o protocolo de sepse?"})
    assert "Nenhum paciente informado" in saida["contexto_paciente"]


def test_cadeia_de_consulta_devolve_resposta_fontes_e_explicacao(modelo, banco, recuperador):
    cadeia = criar_cadeia_consulta(modelo, recuperador, banco, Guardrails())
    saida = cadeia.invoke({"pergunta": "Qual o prazo do pacote da primeira hora na sepse?"})

    assert {"resposta", "fontes", "violacoes", "avisos", "aprovado", "explicacao"} <= set(saida)
    assert saida["fontes"]
    assert saida["explicacao"]["cobertura_das_citacoes"] is not None


def test_guardrail_de_saida_atua_na_cadeia(modelo, banco, recuperador):
    cadeia = criar_cadeia_consulta(modelo, recuperador, banco, Guardrails())
    saida = cadeia.invoke({"pergunta": "Quais criterios abrem o protocolo de sepse?"})
    assert "**Resposta**" in saida["resposta"]
    assert "**Validacao**" in saida["resposta"] or "**Validação**" in saida["resposta"]


def test_cadeia_simples_e_a_linha_de_base(modelo):
    texto = criar_cadeia_simples(modelo).invoke(
        {"pergunta": "Quais criterios abrem o protocolo de sepse?"})
    assert isinstance(texto, str) and texto.strip()


# ----------------------------------------------------------------- CLI
@pytest.fixture
def cli_com_banco_de_teste(monkeypatch, banco, tmp_path):
    """Aponta a CLI para o banco temporario, sem tocar o de desenvolvimento."""
    monkeypatch.setattr(cli, "preparar_banco", lambda *a, **k: banco)
    monkeypatch.setenv("BACKEND_LLM", "eco")
    return banco


def test_politicas_lista_as_doze_vigentes(cli_com_banco_de_teste, capsys):
    assert cli.main(["politicas"]) == 0
    saida = capsys.readouterr().out
    assert "ENT-01-prescricao" in saida
    assert "SAI-01-dose" in saida
    assert saida.count("PROT-GOV-010") >= 12


def test_paciente_mostra_contexto_e_pendencias(cli_com_banco_de_teste, capsys):
    assert cli.main(["paciente", "PS-07"]) == 0
    saida = capsys.readouterr().out
    assert "leito PS-07" in saida
    assert "Pendencias" in saida
    assert "PROT-SEP-001" in saida


def test_paciente_inexistente_retorna_codigo_de_erro(cli_com_banco_de_teste, capsys):
    assert cli.main(["paciente", "ZZ-99"]) == 1
    assert "Nenhum paciente" in capsys.readouterr().out


def test_perguntar_percorre_o_fluxo_e_mostra_as_etapas(cli_com_banco_de_teste, capsys):
    assert cli.main(["perguntar", "Ha pendencia no leito PS-07?", "--backend", "eco"]) == 0
    saida = capsys.readouterr().out
    assert "**Resposta**" in saida
    assert "interacao ....... INT-" in saida
    assert "triagem" in saida and "concluir" in saida


def test_perguntar_com_pedido_de_dose_nao_aciona_o_modelo(cli_com_banco_de_teste, capsys):
    assert cli.main(["perguntar", "Qual a dose de noradrenalina?", "--backend", "eco"]) == 0
    saida = capsys.readouterr().out
    assert "categoria ....... recusa" in saida
    assert "ENT-02-dose" in saida
    assert "gerar_resposta" not in saida


def test_perguntar_com_explicar_mostra_a_base_da_resposta(cli_com_banco_de_teste, capsys):
    assert cli.main(
        ["perguntar", "Quais criterios abrem o protocolo de sepse?",
         "--backend", "eco", "--explicar"]) == 0
    saida = capsys.readouterr().out
    assert "#### Base da resposta" in saida
    assert "Cobertura das citacoes" in saida or "Cobertura das citações" in saida


def test_alertas_lista_o_painel(cli_com_banco_de_teste, capsys):
    cli.main(["perguntar", "Ha pendencia no leito PS-07?", "--backend", "eco"])
    capsys.readouterr()
    assert cli.main(["alertas"]) == 0
    assert "PROT-" in capsys.readouterr().out


def test_validacoes_lista_e_decide(cli_com_banco_de_teste, banco, capsys):
    identificador = banco.registrar_validacao(
        interacao="INT-TESTE-CLI", sugestao="Reavaliar antimicrobiano.",
        prontuario="884521", fontes=["PROT-ATB-003 §3"])
    capsys.readouterr()

    assert cli.main(["validacoes"]) == 0
    assert f"#{identificador}" in capsys.readouterr().out

    assert cli.main(["validacoes", "--decidir", str(identificador), "--status", "aceita",
                     "--profissional", "Dra. Helena (CRM-SP 00000)"]) == 0
    assert "aceita" in capsys.readouterr().out
    decidida = next(v for v in banco.listar_validacoes(status="aceita")
                    if v["id"] == identificador)
    assert decidida["profissional"].startswith("Dra. Helena")


def test_auditoria_lista_interacoes_e_a_trilha(cli_com_banco_de_teste, banco, capsys):
    cli.main(["perguntar", "Ha pendencia no leito PS-07?", "--backend", "eco"])
    capsys.readouterr()

    assert cli.main(["auditoria"]) == 0
    listagem = capsys.readouterr().out
    assert "INT-" in listagem

    interacao = listagem.split("INT-")[1].split()[0]
    assert cli.main(["auditoria", "--interacao", f"INT-{interacao}", "--detalhado"]) == 0
    assert "triagem" in capsys.readouterr().out


def test_auditoria_de_interacao_inexistente(cli_com_banco_de_teste, capsys):
    assert cli.main(["auditoria", "--interacao", "INT-00000000-XXXXXXXX"]) == 1
    assert "Nenhum evento" in capsys.readouterr().out


def test_diagrama_mermaid_e_gerado(cli_com_banco_de_teste, tmp_path, capsys):
    destino = tmp_path / "fluxo.mmd"
    assert cli.main(["diagrama", "--saida", str(destino)]) == 0
    conteudo = destino.read_text(encoding="utf-8")
    assert "triagem" in conteudo
    assert "graph" in conteudo.lower() or "flowchart" in conteudo.lower()


def test_parser_exige_subcomando():
    with pytest.raises(SystemExit):
        cli.main([])
