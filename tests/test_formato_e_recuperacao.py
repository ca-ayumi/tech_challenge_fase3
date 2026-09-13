"""Testes do formato canonico, da recuperacao e da explicabilidade."""

from __future__ import annotations

import pytest

from assistente_medico.explicabilidade import montar_explicacao
from assistente_medico.finetuning.formato import (
    PROMPT_SISTEMA,
    ROTULO_RECUSA,
    montar_mensagens,
    montar_pergunta,
    montar_resposta,
    referencias_citadas,
    segue_formato,
    separar_secoes,
)


# ------------------------------------------------------------------- formato
def test_resposta_montada_tem_tres_blocos():
    texto = montar_resposta("Corpo da resposta.", ["PROT-SEP-001 §3"])
    assert segue_formato(texto)
    assert texto.index("**Resposta**") < texto.index("**Fontes**") < texto.index("**Validacao**")


def test_ida_e_volta_do_formato():
    original = montar_resposta("Corpo.", ["PROT-SEP-001 §3", "PROT-UTI-002 §2"], ROTULO_RECUSA)
    estruturada = separar_secoes(original)
    assert estruturada.corpo == "Corpo."
    assert estruturada.fontes == ["PROT-SEP-001 §3", "PROT-UTI-002 §2"]
    assert estruturada.validacao == ROTULO_RECUSA
    assert estruturada.texto() == original


def test_resposta_sem_fonte_marca_ausencia():
    texto = montar_resposta("Corpo sem fonte.", [])
    assert separar_secoes(texto).fontes == []
    assert "Nenhuma fonte" in texto


def test_aceita_validacao_com_acento():
    texto = montar_resposta("Corpo.", ["PROT-SEP-001 §3"]).replace("**Validacao**",
                                                                   "**Validação**")
    assert segue_formato(texto)
    assert separar_secoes(texto).validacao


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("Ver PROT-SEP-001 §4 e PROT-GOV-010 §3.", ["PROT-SEP-001 §4", "PROT-GOV-010 §3"]),
        ("Conforme PROT-SEP-001 apenas.", ["PROT-SEP-001"]),
        ("Ver MOD-REC-002 e POP-ENF-041.", ["MOD-REC-002", "POP-ENF-041"]),
        ("Sem referencia nenhuma aqui.", []),
    ],
)
def test_extracao_de_referencias(texto, esperado):
    assert referencias_citadas(texto) == esperado


def test_pergunta_inclui_os_contextos():
    texto = montar_pergunta("Pergunta?", contexto_documentos="DOC", contexto_paciente="PAC")
    assert "### Contexto do paciente" in texto
    assert "### Contexto institucional recuperado" in texto
    assert texto.index("Contexto do paciente") < texto.index("Contexto institucional")


def test_mensagens_seguem_ordem_de_chat():
    mensagens = montar_mensagens("P", "R")
    assert [m["role"] for m in mensagens] == ["system", "user", "assistant"]
    assert mensagens[0]["content"] == PROMPT_SISTEMA


def test_prompt_de_sistema_declara_as_vedacoes():
    for termo in ["prescricao", "dose", "diagnostico definitivo", "validacao humana"]:
        assert termo in PROMPT_SISTEMA.lower()


# ---------------------------------------------------------------- recuperacao
@pytest.mark.parametrize(
    "consulta, esperado",
    [
        ("criterios para abrir o protocolo de sepse", "PROT-SEP-001 §3"),
        ("janela de trombolise no AVC", "PROT-NEU-003 §5"),
        ("meta de tempo do eletrocardiograma na dor toracica", "PROT-CAR-002 §2"),
        ("reposicao de potassio na cetoacidose", "PROT-END-005 §4"),
        ("o que o assistente nao pode fazer", "PROT-GOV-010 §3"),
    ],
)
def test_recupera_a_secao_correta(recuperador, consulta, esperado):
    referencias = [r.referencia for r in recuperador.recuperar(consulta, top_k=4)]
    assert esperado in referencias, f"esperava {esperado} entre {referencias}"


def test_citacao_explicita_tem_prioridade(recuperador):
    resultados = recuperador.recuperar("o que diz o PROT-GOV-010 §3")
    assert resultados[0].referencia == "PROT-GOV-010 §3"
    assert "secao citada" in resultados[0].motivo


def test_protocolo_ativo_do_paciente_recebe_reforco(recuperador):
    sem = recuperador.recuperar("quais sao os proximos passos")
    com = recuperador.recuperar("quais sao os proximos passos",
                                documentos_preferidos=["PROT-END-005"])
    documentos_com = {r.trecho.documento for r in com}
    assert "PROT-END-005" in documentos_com or sem != com


def test_consulta_vazia_nao_quebra(recuperador):
    assert recuperador.recuperar("") == []
    assert recuperador.recuperar("de a o") == []


def test_contexto_formatado_respeita_limite(recuperador):
    recuperados = recuperador.recuperar("sepse", top_k=4)
    contexto = recuperador.formatar_contexto(recuperados, limite_caracteres=300)
    assert len(contexto) <= 300 + 200  # um bloco inteiro por vez


# ----------------------------------------------------------- explicabilidade
def test_detecta_citacao_nao_fundamentada(recuperador):
    recuperados = recuperador.recuperar("criterios de abertura do protocolo de sepse", top_k=2)
    resposta = montar_resposta(
        "Conforme PROT-SEP-001 §3 e tambem PROT-FALSO-999 §1.",
        ["PROT-SEP-001 §3"],
    )
    explicacao = montar_explicacao(resposta, recuperados)
    assert "PROT-FALSO-999 §1" in explicacao.nao_fundamentadas
    assert explicacao.cobertura < 1.0
    assert explicacao.tem_alerta_de_fonte


def test_citacao_fundamentada_tem_cobertura_total(recuperador):
    recuperados = recuperador.recuperar("criterios de abertura do protocolo de sepse", top_k=2)
    resposta = montar_resposta("Conforme PROT-SEP-001 §3.", ["PROT-SEP-001 §3"])
    explicacao = montar_explicacao(resposta, recuperados)
    assert explicacao.nao_fundamentadas == []
    assert explicacao.cobertura == 1.0


def test_explicacao_lista_fontes_consultadas_mesmo_sem_citacao(recuperador):
    recuperados = recuperador.recuperar("sepse", top_k=3)
    explicacao = montar_explicacao(montar_resposta("Sem citar nada.", []), recuperados)
    assert len(explicacao.fontes) >= 3, "o auditor precisa ver o que foi consultado"
    assert "Base da resposta" in explicacao.como_markdown()
