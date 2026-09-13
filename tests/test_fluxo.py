"""Testes de integracao do fluxo LangGraph.

Usam o backend deterministico ('eco'), de modo que o fluxo inteiro — triagem,
contexto, regras, recuperacao, guardrails, explicabilidade, validacao e
auditoria — e exercitado sem depender de um modelo baixado. O que se verifica
aqui e a governanca do fluxo, nao a qualidade do texto gerado.
"""

from __future__ import annotations

import json

from assistente_medico.grafo.fluxo import rota_apos_triagem


def test_rota_de_recusa():
    assert rota_apos_triagem({"permitido": False}) == "recusar"


def test_rota_com_paciente():
    assert rota_apos_triagem({"permitido": True, "prontuario": "884521"}) == \
        "carregar_contexto_paciente"


def test_rota_sem_paciente():
    assert rota_apos_triagem({"permitido": True, "prontuario": None}) == "recuperar_protocolos"


def test_pergunta_de_protocolo_percorre_o_tronco(assistente):
    estado = assistente.responder("Quais os criterios de abertura do protocolo de sepse?")
    assert estado["categoria"] == "permitido"
    assert estado["intencao"] == "protocolo"
    assert estado["prontuario"] is None
    for etapa in ("triagem", "recuperar_protocolos", "gerar_resposta", "validar_saida",
                  "explicar", "concluir"):
        assert etapa in estado["etapas"]
    assert estado["recuperados"], "a recuperacao precisa trazer trechos"
    assert "**Resposta**" in estado["resposta"]


def test_pergunta_sobre_paciente_aciona_prontuario_exames_e_regras(assistente):
    estado = assistente.responder("Ha alguma pendencia no leito PS-07?")
    assert estado["prontuario"] == "884521"
    assert estado["intencao"] == "paciente"
    assert "contexto_paciente" in estado["etapas"]
    assert "exames_pendentes" in estado["etapas"]
    assert "avaliar_regras" in estado["etapas"]
    assert estado["exames_pendentes"], "PS-07 tem exames pendentes no cenario"
    assert estado["alertas"], "PS-07 tem pendencia de protocolo de sepse"
    assert estado["alertas_registrados"], "os alertas precisam ir para o painel"
    assert estado["validacao_id"], "resposta sobre paciente exige validacao humana"


def test_solicitacao_de_dose_nao_chega_ao_modelo(assistente):
    estado = assistente.responder("Qual a dose de noradrenalina para o leito UTI-02?")
    assert estado["categoria"] == "recusa"
    assert estado["etapas"] == ["triagem", "recusa", "concluir"]
    assert "gerar_resposta" not in estado["etapas"]
    assert "ENT-02-dose" in [v["codigo"] for v in estado["violacoes"]]
    assert estado["validacao_id"] is None
    assert "PROT-GOV-010" in estado["resposta"]


def test_emergencia_e_encaminhada_sem_bloquear(assistente):
    estado = assistente.responder(
        "Paciente com dor toracica intensa agora, pressao 80 por 50."
    )
    assert estado["categoria"] == "emergencia"
    assert estado["permitido"] is True
    assert "gerar_resposta" in estado["etapas"]


def test_perfil_paciente_e_bloqueado(assistente):
    estado = assistente.responder("Quais exames estao pendentes?", perfil="paciente")
    assert estado["categoria"] == "recusa"
    assert "ENT-06-paciente" in [v["codigo"] for v in estado["violacoes"]]


def test_resposta_sempre_tem_os_tres_blocos(assistente):
    from assistente_medico.finetuning.formato import segue_formato

    perguntas = [
        "Quais os criterios de abertura do protocolo de sepse?",
        "Ha pendencia no leito UTI-04?",
        "Qual a dose de heparina?",
        "Quais itens sao obrigatorios no laudo de imagem?",
    ]
    for pergunta in perguntas:
        estado = assistente.responder(pergunta)
        assert segue_formato(estado["resposta"]), f"formato quebrado em: {pergunta}"


def test_interacao_gera_trilha_de_auditoria_completa(assistente):
    estado = assistente.responder("Ha alguma pendencia no leito PS-07?", usuario="dra.teste")
    eventos = [e["evento"] for e in assistente.trilha(estado["interacao"])]
    for esperado in ("triagem", "contexto_paciente", "exames_pendentes", "regras_aplicadas",
                     "recuperacao_protocolos", "geracao_modelo", "guardrail_saida",
                     "alertas_emitidos", "explicabilidade", "validacao_aberta",
                     "interacao_concluida"):
        assert esperado in eventos, f"evento ausente na trilha: {esperado}"


def test_auditoria_mascara_dados_pessoais(assistente, banco):
    estado = assistente.responder(
        "Paciente Maria Aparecida da Silva, CPF 321.654.987-00, tem exames pendentes?"
    )
    for evento in assistente.trilha(estado["interacao"]):
        bruto = json.dumps(evento["conteudo"], ensure_ascii=False)
        assert "321.654.987-00" not in bruto
        assert "Maria Aparecida da Silva" not in bruto


def test_recusa_tambem_e_auditada(assistente):
    estado = assistente.responder("Prescreve ceftriaxona para o leito PS-07.")
    eventos = [e["evento"] for e in assistente.trilha(estado["interacao"])]
    assert "recusa_por_politica" in eventos
    assert "interacao_concluida" in eventos


def test_alerta_de_conflito_de_alergia_chega_ao_painel(assistente, banco):
    estado = assistente.responder("Resuma a situacao do leito PS-05.")
    assert estado["prontuario"] == "990117"
    mensagens = " ".join(a["mensagem"] for a in estado["alertas"])
    assert "Conflito de alergia" in mensagens
    registrados = banco.listar_alertas(prontuario="990117")
    assert any("Conflito de alergia" in a["mensagem"] for a in registrados)


def test_explicacao_acompanha_a_resposta(assistente):
    estado = assistente.responder("Qual a janela de trombolise no AVC?")
    explicacao = estado["explicacao"]
    assert explicacao["fontes"]
    assert explicacao["cobertura_das_citacoes"] == 1.0
    assert explicacao["citacoes_nao_fundamentadas"] == []


def test_diagrama_mermaid_e_gerado(assistente):
    diagrama = assistente.diagrama_mermaid()
    assert "triagem" in diagrama
    assert "validar_saida" in diagrama


def test_interacoes_tem_identificadores_distintos(assistente):
    primeira = assistente.responder("Quais os criterios de abertura do protocolo de sepse?")
    segunda = assistente.responder("Quais os criterios de abertura do protocolo de sepse?")
    assert primeira["interacao"] != segunda["interacao"]
