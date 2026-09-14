"""Testes do pipeline de construcao do dataset de fine-tuning.

Cobre os cinco geradores, a anonimizacao aplicada ao que vai para o treino e a
gravacao no formato de chat consumido pelo MLX-LM e pelo TRL.
"""

from __future__ import annotations

import json
import random

import pytest

from assistente_medico.config import CAMINHOS
from assistente_medico.dados.anonimizacao import Anonimizador
from assistente_medico.dados.construir_dataset import (
    _condensar,
    construir_corpus,
    exemplos_de_documentos,
    exemplos_de_pacientes,
    exemplos_de_protocolos,
    exemplos_de_recusa,
    exemplos_do_faq,
    gravar_particao,
)
from assistente_medico.dados.curadoria import detectar_pii_residual, texto_treinado
from assistente_medico.finetuning.formato import (
    ROTULO_RECUSA,
    referencias_citadas,
    segue_formato,
    separar_secoes,
)

CAMPOS_OBRIGATORIOS = {
    "categoria", "pergunta", "resposta", "fontes",
    "contexto_documentos", "contexto_paciente", "origem",
}


@pytest.fixture
def aleatorio():
    return random.Random(42)


@pytest.fixture
def indice(trechos):
    return {trecho.referencia: trecho for trecho in trechos}


@pytest.fixture(scope="session")
def corpus():
    corpus, metadados = construir_corpus(semente=42)
    return corpus, metadados


# ------------------------------------------------------------- _condensar
def test_condensar_respeita_o_limite_e_corta_em_fronteira():
    texto = "Primeira frase completa. Segunda frase completa. Terceira frase completa."
    resultado = _condensar(texto, limite=40)
    assert len(resultado) <= len(texto)
    assert resultado.endswith(".")
    assert not resultado.endswith(" .")


def test_condensar_devolve_texto_curto_intacto():
    texto = "Coletar lactato arterial."
    assert _condensar(texto, limite=900) == texto


# ---------------------------------------------------------------- geradores
def test_todo_exemplo_tem_os_campos_do_contrato(corpus):
    exemplos, _ = corpus
    for exemplo in exemplos:
        assert CAMPOS_OBRIGATORIOS <= set(exemplo), exemplo.get("origem")


def test_toda_resposta_segue_os_tres_blocos(corpus):
    exemplos, _ = corpus
    for exemplo in exemplos:
        assert segue_formato(exemplo["resposta"]), exemplo["origem"]


def test_protocolos_geram_uma_variante_com_rag_e_uma_sem(trechos, aleatorio):
    anonimizador = Anonimizador(sal="sal-de-teste")
    exemplos = exemplos_de_protocolos(trechos, anonimizador, aleatorio)
    assert exemplos
    com_contexto = [e for e in exemplos if e["contexto_documentos"]]
    sem_contexto = [e for e in exemplos if not e["contexto_documentos"]]
    assert com_contexto and sem_contexto
    assert len(com_contexto) == len(sem_contexto)


def test_faq_vira_exemplo_com_fonte(indice, aleatorio):
    anonimizador = Anonimizador(sal="sal-de-teste")
    exemplos = exemplos_do_faq(
        CAMINHOS.dados_brutos / "faq_medicos.jsonl", indice, anonimizador, aleatorio)
    assert len(exemplos) >= 50
    assert all(e["origem"].startswith("faq:") for e in exemplos)


# --------------------------------------------------- modelos de documento
def test_documentos_cobrem_os_quatro_modelos(trechos, indice):
    exemplos = exemplos_de_documentos(trechos, indice)
    documentos = {e["origem"].split(":")[1].split(" ")[0] for e in exemplos}
    assert documentos == {"MOD-ALT-004", "MOD-LAU-001", "MOD-PRO-003", "MOD-REC-002"}


def test_documentos_geram_duas_perguntas_distintas_por_modelo(trechos, indice):
    exemplos = exemplos_de_documentos(trechos, indice)
    assert len(exemplos) == 8
    assert len({e["pergunta"] for e in exemplos}) == 8


def test_documentos_nao_dependem_do_campo_tipo(trechos, indice):
    """Regressao: o filtro antigo usava ``tipo == "modelo_documento"``.

    O frontmatter de cada arquivo declara o proprio tipo (``laudo``, ``receita``),
    que sobrescreve o rotulo generico da carga, e o filtro nunca casava — a
    funcao devolvia lista vazia e os modelos entravam no dataset apenas de
    carona no gerador de protocolos.
    """
    assert all(trecho.tipo != "modelo_documento" for trecho in trechos)
    assert exemplos_de_documentos(trechos, indice)


def test_documento_exige_validacao_humana(trechos, indice):
    for exemplo in exemplos_de_documentos(trechos, indice):
        assert "MINUTA" in exemplo["resposta"]
        assert separar_secoes(exemplo["resposta"]).validacao


# ------------------------------------------------------------------ recusas
def test_recusa_cita_a_vedacao_e_oferece_alternativa(banco, indice, aleatorio):
    exemplos = exemplos_de_recusa(banco, indice, aleatorio)
    assert exemplos
    for exemplo in exemplos:
        estruturada = separar_secoes(exemplo["resposta"])
        assert estruturada.validacao.startswith(ROTULO_RECUSA[:40])
        assert "PROT-GOV-010" in " ".join(referencias_citadas(exemplo["resposta"]))


def test_recusa_nao_contem_dose_na_resposta(banco, indice, aleatorio):
    from assistente_medico.seguranca.politicas import POLITICAS_SAIDA

    politica_dose = next(p for p in POLITICAS_SAIDA if p.codigo == "SAI-01-dose")
    for exemplo in exemplos_de_recusa(banco, indice, aleatorio):
        assert politica_dose.casa(exemplo["resposta"]) is None, exemplo["pergunta"]


# ---------------------------------------------------------------- pacientes
def test_contexto_do_paciente_nao_carrega_identificadores(banco, indice, prontuarios):
    exemplos = exemplos_de_pacientes(banco, indice)
    assert exemplos
    nomes = [p["nome"] for p in prontuarios]
    cpfs = [p["cpf"] for p in prontuarios if p.get("cpf")]
    for exemplo in exemplos:
        contexto = exemplo["contexto_paciente"]
        assert contexto
        assert not any(nome in contexto for nome in nomes)
        assert not any(cpf in contexto for cpf in cpfs)


# -------------------------------------------------- anonimizacao do corpus
def test_nenhum_identificador_direto_sobrevive_no_corpus(corpus):
    """Regressao: a anonimizacao cobria a resposta, mas nao o contexto.

    As secoes de exemplo dos modelos de documento trazem CPF, CNS e telefone
    ficticios. Como o contexto recuperado entra no turno do usuario, ele chegava
    ao treino sem tratamento.
    """
    exemplos, _ = corpus
    vazamentos = [
        (e["origem"], detectar_pii_residual(texto_treinado(e)))
        for e in exemplos
        if detectar_pii_residual(texto_treinado(e))
    ]
    assert not vazamentos, f"PII no corpus: {vazamentos[:3]}"


def test_contexto_recuperado_passa_pelo_anonimizador(corpus):
    exemplos, _ = corpus
    contextos = "\n".join(e["contexto_documentos"] for e in exemplos if e["contexto_documentos"])
    assert contextos
    assert "[CPF]" in contextos or "[CNS]" in contextos or "[TELEFONE]" in contextos


def test_metadados_contabilizam_as_fontes(corpus):
    _, metadados = corpus
    assert metadados["documentos"] == 14
    assert metadados["trechos"] == 96
    assert metadados["prontuarios"] == 12
    assert metadados["identificadores_anonimizados_em_evolucoes"]


# ------------------------------------------------------------------ gravacao
def test_particao_gravada_no_formato_de_chat(tmp_path, corpus):
    exemplos, _ = corpus
    destino = gravar_particao(exemplos[:5], tmp_path / "train.jsonl")
    linhas = destino.read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 5
    for linha in linhas:
        mensagens = json.loads(linha)["messages"]
        assert [m["role"] for m in mensagens] == ["system", "user", "assistant"]
        assert all(m["content"].strip() for m in mensagens)


def test_contextos_entram_no_turno_do_usuario(tmp_path, corpus):
    exemplos, _ = corpus
    com_contexto = next(e for e in exemplos if e["contexto_documentos"])
    destino = gravar_particao([com_contexto], tmp_path / "um.jsonl")
    mensagens = json.loads(destino.read_text(encoding="utf-8"))["messages"]
    turno_usuario = mensagens[1]["content"]
    assert "### Contexto institucional recuperado" in turno_usuario
    assert com_contexto["pergunta"] in turno_usuario
