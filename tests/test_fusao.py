"""Fusao multimodal, relogio de cenario e demonstracao ponta a ponta (modo local)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import pytest

from assistente_medico.auditoria import Auditoria
from assistente_medico.dados.banco import BancoHospital
from assistente_medico.multimodal import Anomalia, CentralAlertas, fusao, relogio

AGORA = datetime(2026, 9, 14, 22, 0)


def _a(modalidade: str, tipo: str, severidade: str = "alta", horas_atras: float = 0.0,
       prontuario: str = "P1", identificador: int = 0) -> dict:
    momento = (AGORA - timedelta(hours=horas_atras)).isoformat(timespec="seconds")
    return {"id": identificador, "modalidade": modalidade, "tipo": tipo, "severidade": severidade,
            "descricao": f"{tipo} ({severidade})", "prontuario": prontuario, "inicio": momento,
            "fim": momento, "detectada_em": momento}


def test_escore_decai_com_o_tempo_e_tem_retorno_decrescente():
    recente, _ = fusao.escore([_a("sinais_vitais", "desvio_basal_fc", "alta")], AGORA)
    antigo, _ = fusao.escore([_a("sinais_vitais", "desvio_basal_fc", "alta", horas_atras=6)], AGORA)
    assert recente == pytest.approx(25) and antigo == pytest.approx(12.5)
    repetido, contribuicoes = fusao.escore(
        [_a("sinais_vitais", t, "alta") for t in ("desvio_basal_fc", "tendencia_fc", "news2")], AGORA)
    assert repetido == pytest.approx(25 + 0.3 * 50)
    assert set(contribuicoes) == {"sinais_vitais"}


def test_concordancia_entre_modalidades_aumenta_o_escore():
    sozinho, _ = fusao.escore([_a("sinais_vitais", "x", "alta"), _a("sinais_vitais", "y", "alta")], AGORA)
    juntos, _ = fusao.escore([_a("sinais_vitais", "x", "alta"), _a("audio", "voz_dispneia", "alta")], AGORA)
    assert juntos == pytest.approx(50 * 1.15) and juntos > sozinho


def test_achado_critico_recente_garante_nivel_alto():
    pontos, _ = fusao.escore([_a("texto", "termo_critico_dor_toracica", "critica")], AGORA)
    assert fusao.nivel(pontos) == "alto"
    antigo, _ = fusao.escore([_a("texto", "termo_critico_dor_toracica", "critica", horas_atras=5)], AGORA)
    assert fusao.nivel(antigo) == "moderado"
    assert [fusao.nivel(v) for v in (80, 50, 25, 5)] == ["critico", "alto", "moderado", "baixo"]


def test_queda_confirmada_exige_as_duas_fontes_na_janela():
    proximas = [_a("movimentacao", "queda", "critica", 0.01), _a("video", "video_queda", "critica", 0)]
    [correlacao] = fusao.correlacionar(proximas)
    assert correlacao.regra.chave == "queda_confirmada"
    distantes = [_a("movimentacao", "queda", "critica", 0.5), _a("video", "video_queda", "critica", 0)]
    assert not fusao.correlacionar(distantes)


def test_regra_ordenada_respeita_a_sequencia():
    antes = [_a("prescricao", "suspensao_sem_substituto", "moderada", 10),
             _a("sinais_vitais", "desvio_basal_temperatura", "alta", 2)]
    assert [c.regra.chave for c in fusao.correlacionar(antes)] == ["febre_apos_suspensao"]
    invertida = [_a("prescricao", "suspensao_sem_substituto", "moderada", 2),
                 _a("sinais_vitais", "desvio_basal_temperatura", "alta", 10)]
    assert not fusao.correlacionar(invertida)


def test_avc_combina_voz_e_conteudo_e_cita_protocolo():
    anomalias = [_a("audio", "voz_disartria", "alta", 0, identificador=1),
                 _a("texto", "termo_critico_deficit_neurologico", "critica", 0, identificador=2)]
    avaliacao = fusao.avaliar_paciente("P1", anomalias, AGORA)
    [consolidada] = fusao.anomalias_consolidadas(avaliacao)
    assert consolidada.modalidade == "multimodal" and consolidada.severidade == "critica"
    assert consolidada.tipo == "fusao_suspeita_avc"
    assert "PROT-NEU-003" in consolidada.descricao
    assert consolidada.evidencias["anomalias"] == [1, 2]


def test_risco_elevado_sem_correlacao_gera_alerta_proprio():
    anomalias = [_a("sinais_vitais", "news2_elevado", "critica"),
                 _a("prescricao", "salto_dose", "critica"), _a("video", "exercicio_assimetria", "alta")]
    avaliacao = fusao.avaliar_paciente("P1", anomalias, AGORA)
    assert avaliacao.escore >= fusao.RISCO_ELEVADO and not avaliacao.correlacoes
    [consolidada] = fusao.anomalias_consolidadas(avaliacao)
    assert consolidada.tipo == "fusao_risco_elevado"


def test_janela_de_24_horas_descarta_anomalias_antigas():
    avaliacao = fusao.avaliar_paciente("P1", [_a("audio", "voz_fadiga", "alta", horas_atras=30)], AGORA)
    assert avaliacao.escore == 0 and not avaliacao.anomalias


@pytest.fixture
def banco_fusao(tmp_path, prontuarios) -> BancoHospital:
    banco = BancoHospital(tmp_path / "h.db")
    banco.popular(prontuarios)
    return banco


def _central(banco, tmp_path) -> CentralAlertas:
    return CentralAlertas(banco, notificadores=[],
                          auditoria=Auditoria(banco=banco, caminho_jsonl=tmp_path / "a.jsonl"))


def test_consolidacao_pelo_banco_emite_uma_vez(banco_fusao, tmp_path):
    central = _central(banco_fusao, tmp_path)
    referencia = relogio.agora(banco_fusao)
    instante = referencia.isoformat(timespec="seconds")
    central.processar([
        Anomalia("movimentacao", "queda", "critica", "queda pelo acelerometro", "875431",
                 instante, instante),
        Anomalia("video", "video_queda", "critica", "queda pela camera", "875431", instante, instante),
    ])
    resultados = fusao.consolidar(banco_fusao, central)
    primeiro = next(r for a, r in resultados if a.prontuario == "875431")
    assert [r.anomalia.tipo for r in primeiro] == ["fusao_queda_confirmada"]
    assert primeiro[0].emitido
    de_novo = next(r for a, r in fusao.consolidar(banco_fusao, central) if a.prontuario == "875431")
    assert not de_novo[0].emitido


def test_relogio_de_cenario_e_real(banco_fusao, monkeypatch):
    ultimo = max(s["aferido_em"] for s in banco_fusao.consultar_somente_leitura(
        "SELECT aferido_em FROM sinais_vitais", limite=500))
    assert relogio.agora(banco_fusao) == datetime.fromisoformat(ultimo)
    real = dataclasses.replace(relogio.MONITORAMENTO, relogio="real")
    monkeypatch.setattr(relogio, "MONITORAMENTO", real)
    assert abs((relogio.agora(banco_fusao) - datetime.now()).total_seconds()) < 5


def test_demonstracao_local_ponta_a_ponta(tmp_path, banco_fusao, monkeypatch):
    from assistente_medico import auditoria
    from assistente_medico.multimodal import alertas, demonstracao
    from assistente_medico.multimodal.anomalias import sintetico
    from assistente_medico.multimodal.audio import roteiros

    caminhos = dataclasses.replace(auditoria.CAMINHOS, logs=tmp_path / "logs")
    monkeypatch.setattr(auditoria, "CAMINHOS", caminhos)
    monkeypatch.setattr(alertas, "CAMINHOS", caminhos)
    monkeypatch.setattr(sintetico, "PASTA_PADRAO", tmp_path / "monitoramento")
    monkeypatch.setattr(roteiros, "PASTA_AUDIO", tmp_path / "audio")
    mensagens: list[str] = []

    resumo = demonstracao.executar(banco_fusao, usar_azure=False, usar_video=False,
                                   progresso=mensagens.append)

    assert mensagens[0].startswith("1/4") and mensagens[-1].startswith("4/4")
    assert resumo["fluxo"]["alertas_emitidos"] > 0
    consolidados = [a for a in banco_fusao.listar_alertas(limite=200)
                    if a["categoria"] == "anomalia_multimodal"]
    regras = {a["fonte"] for a in consolidados}
    assert "monitoramento/multimodal/fusao_tardia" in regras
    assert any("Febre apos suspensao" in a["mensagem"] for a in consolidados)
    assert any(item["nivel"] in {"critico", "alto"} for item in resumo["fusao"])
