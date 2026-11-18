"""Detectores de anomalia: sinais vitais, prescricoes, movimentacao e dados sinteticos."""

from __future__ import annotations

import dataclasses
import random
from datetime import datetime, timedelta

import pytest

from assistente_medico.dados.banco import BancoHospital
from assistente_medico.multimodal import Leitura
from assistente_medico.multimodal.anomalias import (
    DetectorMovimentacao,
    DetectorPrescricoes,
    DetectorSinaisVitais,
    GerenciadorEpisodios,
    ParametrosSinaisVitais,
    news2,
    sintetico,
)
from assistente_medico.multimodal.anomalias.avaliacao import avaliar_internacao, injetar
from assistente_medico.multimodal.anomalias.fontes import ler_bidmc, ler_urfall

T0 = datetime(2026, 9, 15, 8, 0, 0)


def _tipos(achados) -> set[str]:
    return {a.tipo for a in achados}


def _rodar(detector, leituras) -> list:
    achados = []
    for leitura in leituras:
        achados.extend(detector.atualizar(leitura))
    return achados


def test_news2_subescores_e_faixas():
    assert news2.subescore("saturacao", 91) == 3
    assert news2.subescore("fc", 131) == 3
    assert news2.subescore("temperatura", 39.1) == 2
    normal = news2.calcular({"fr": 16, "saturacao": 97, "pas": 120, "fc": 80,
                             "temperatura": 37.0, "glasgow": 15})
    assert normal["total"] == 0 and normal["faixa"] == "baixo" and not normal["parcial"]
    grave = news2.calcular({"fr": 26, "saturacao": 90, "fc": 125})
    assert grave["total"] == 8 and grave["faixa"] == "alto" and grave["parcial"]
    assert news2.calcular({"fr": 26})["faixa"] == "baixo_medio"


def test_episodio_exige_persistencia_emite_no_agravamento_e_fecha():
    episodios = GerenciadorEpisodios(tolerancia_normal=2)
    chave = ("desvio", "fc")
    assert episodios.positivo(chave, T0, "moderada", timedelta(seconds=20)) is None
    assert episodios.positivo(chave, T0 + timedelta(seconds=25), "moderada",
                              timedelta(seconds=20)) == T0
    assert episodios.positivo(chave, T0 + timedelta(seconds=26), "moderada") is None
    assert episodios.positivo(chave, T0 + timedelta(seconds=27), "alta") == T0
    episodios.negativo(chave)
    assert episodios.ativo(chave)
    episodios.negativo(chave)
    assert not episodios.ativo(chave)


def _serie_1hz(segundos: int, valores, semente: int = 1) -> list[Leitura]:
    gerador = random.Random(semente)
    leituras = []
    for s in range(segundos):
        base = valores(s)
        leituras.append(Leitura("P1", "sinais_vitais", T0 + timedelta(seconds=s), {
            "fc": None if base["fc"] is None else base["fc"] + gerador.choice([-1, 0, 1]),
            "fr": base.get("fr", 16), "saturacao": base.get("saturacao", 97),
        }))
    return leituras


def test_desvio_da_linha_de_base_dentro_da_faixa_normal_do_news2():
    leituras = _serie_1hz(420, lambda s: {"fc": 68 if s < 300 else 98})
    achados = _rodar(DetectorSinaisVitais(ParametrosSinaisVitais.alta_frequencia(
        camadas=frozenset({"limiar", "basal"}))), leituras)
    desvios = [a for a in achados if a.tipo == "desvio_basal_fc"]
    assert desvios and desvios[0].evidencias["z"] > 4
    assert not any(a.tipo.startswith("news2") for a in achados)
    assert datetime.fromisoformat(desvios[0].inicio) >= T0 + timedelta(seconds=300)


def test_artefato_curto_nao_vira_alarme_e_desconexao_vira_falha_de_sensor():
    curto = _serie_1hz(400, lambda s: {"fc": 0 if 300 <= s < 308 else 80})
    assert not _rodar(DetectorSinaisVitais(), curto)

    desconectado = _serie_1hz(400, lambda s: {"fc": None if s >= 300 else 80})
    achados = _rodar(DetectorSinaisVitais(), desconectado)
    assert _tipos(achados) == {"falha_sensor_fc"}
    assert achados[0].severidade == "baixa"


def test_tendencia_horaria_de_queda_pressorica():
    detector = DetectorSinaisVitais(ParametrosSinaisVitais.horario(
        camadas=frozenset({"tendencia"})))
    leituras = [
        Leitura("P1", "sinais_vitais", T0 + timedelta(hours=h),
                {"pas": 130 - (0 if h < 12 else 5 * (h - 12)), "fc": 80})
        for h in range(20)
    ]
    achados = _rodar(detector, leituras)
    assert "tendencia_pas" in _tipos(achados)
    assert all(a.severidade == "moderada" for a in achados)


def test_injecao_de_dessaturacao_preserva_o_inicio():
    leituras = _serie_1hz(400, lambda s: {"fc": 80})
    injetadas, injecao = injetar(leituras, "dessaturacao", random.Random(3))
    assert injetadas[0] is leituras[0]
    fim = injetadas[-1].valores["saturacao"]
    assert fim <= 97 - 6 and injecao.inicio > T0


@pytest.fixture
def detector_prescricoes() -> DetectorPrescricoes:
    return DetectorPrescricoes(alergias={"P1": ["Penicilina"]})


def _rx(medicamento, dose=None, horas=0.0, intervalo=None, unidade="mg", situacao="ativa"):
    return Leitura("P1", "prescricao", T0 + timedelta(hours=horas), {
        "medicamento": medicamento, "dose": dose, "intervalo_horas": intervalo,
        "unidade": unidade, "situacao": situacao})


def test_alergia_por_grupo_farmacologico(detector_prescricoes):
    [achado] = detector_prescricoes.atualizar(_rx("Amoxicilina com clavulanato", 875, 0, 12))
    assert achado.tipo == "conflito_alergia" and achado.severidade == "critica"
    assert not detector_prescricoes.atualizar(_rx("Ceftriaxona", 1000, 1, 12))


def test_salto_de_dose_de_dez_vezes_sugere_erro_de_digitacao(detector_prescricoes):
    detector_prescricoes.atualizar(_rx("Cefazolina", 2000, 0, 8))
    achados = detector_prescricoes.atualizar(_rx("Cefazolina", 20000, 30, 8))
    assert {"salto_dose", "acima_referencia"} <= _tipos(achados)
    salto = next(a for a in achados if a.tipo == "salto_dose")
    assert "erro de digitacao" in salto.descricao and salto.evidencias["razao"] == 10


def test_mudanca_brusca_respeita_janela_e_medicamento_titulavel(detector_prescricoes):
    detector_prescricoes.atualizar(_rx("Furosemida", 40, 0, 12))
    assert "mudanca_brusca_dose" in _tipos(detector_prescricoes.atualizar(_rx("Furosemida", 80, 6, 12)))
    assert not detector_prescricoes.atualizar(_rx("Furosemida", 160, 40, 12))
    detector_prescricoes.atualizar(_rx("Noradrenalina", 0.1, 0, None, "mcg/kg/min"))
    assert not detector_prescricoes.atualizar(_rx("Noradrenalina", 0.3, 2, None, "mcg/kg/min"))


def test_duplicidade_e_suspensao_sem_substituto(detector_prescricoes):
    detector_prescricoes.atualizar(_rx("Enoxaparina", 40, 0, 24))
    [duplicidade] = detector_prescricoes.atualizar(_rx("Heparina", 5000, 2, 8, "UI"))
    assert duplicidade.tipo == "duplicidade_terapeutica"
    assert not detector_prescricoes.atualizar(_rx("Heparina", None, 3, situacao="suspensa"))
    [suspensao] = detector_prescricoes.atualizar(_rx("Enoxaparina", None, 4, situacao="suspensa"))
    assert suspensao.tipo == "suspensao_sem_substituto"


def _leito(horas: float, posicao: str, atividade: float, prontuario: str = "P1") -> Leitura:
    return Leitura(prontuario, "movimentacao", T0 + timedelta(hours=horas),
                   {"posicao": posicao, "atividade": atividade})


def test_imobilidade_prolongada_escala_com_o_tempo():
    detector = DetectorMovimentacao()
    achados = _rodar(detector, [_leito(h / 4, "decubito_dorsal", 0.02) for h in range(24)])
    assert [a.severidade for a in achados if a.tipo == "imobilidade_prolongada"] == [
        "moderada", "alta"]
    assert not detector.atualizar(_leito(6.25, "decubito_lateral_direito", 0.1))


def test_agitacao_e_saida_noturna_em_paciente_com_risco_de_queda():
    detector = DetectorMovimentacao(risco_queda={"P1"})
    noite = T0.replace(hour=22) - T0
    horas_noite = noite.total_seconds() / 3600
    calmas = [_leito(horas_noite + q / 4, "decubito_dorsal", 0.04) for q in range(12)]
    agitadas = [_leito(horas_noite + 3 + q / 4, "decubito_dorsal", 0.7) for q in range(4)]
    saida = [_leito(horas_noite + 4.25, "fora_do_leito", 0.6)]
    achados = _rodar(detector, calmas + agitadas + saida)
    assert "agitacao_noturna" in _tipos(achados)
    [saida_noturna] = [a for a in achados if a.tipo == "saida_leito_noturna"]
    assert saida_noturna.severidade == "alta"


def _acelerometro(pico: float, vertical_depois: float) -> list[Leitura]:
    leituras = []
    for ms in range(0, 3000, 20):
        if ms < 1000:
            total, ay = 1.0, 1.0
        elif ms == 1000:
            total, ay = pico, 0.5
        else:
            total, ay = 1.0, vertical_depois
        leituras.append(Leitura("P1", "movimentacao", T0 + timedelta(milliseconds=ms),
                                {"aceleracao_g": total, "ay": ay}))
    return leituras


def test_queda_por_pico_de_aceleracao_com_postura_como_evidencia():
    [queda] = _rodar(DetectorMovimentacao(), _acelerometro(6.5, 0.1))
    assert queda.tipo == "queda" and queda.severidade == "critica"
    assert queda.evidencias["postura_pos_impacto"] == "horizontal"
    assert not _rodar(DetectorMovimentacao(), _acelerometro(2.2, 0.1))


def test_fontes_leem_bidmc_e_urfall(tmp_path):
    bidmc = tmp_path / "bidmc_07_Numerics.csv"
    bidmc.write_text("Time [s], HR, PULSE, RESP, SpO2\n0,94,93,25,97\n1,95,93,,0\n")
    [primeira, segunda] = ler_bidmc(bidmc)
    assert primeira.prontuario == "BIDMC_07" and primeira.valores["fc"] == 94
    assert segunda.valores["fr"] is None and segunda.valores["saturacao"] == 0

    urfall = tmp_path / "fall-03-acc.csv"
    urfall.write_text("30,1.0,0.1,0.99,0.2\n-2,1.02,0.1,0.98,0.2\n")
    leituras = ler_urfall(urfall, prontuario="P1")
    assert [round(leitura.valores["aceleracao_g"], 2) for leitura in leituras] == [1.02, 1.0]


def test_sintetico_e_deterministico_e_todos_os_eventos_sao_detectados(prontuarios):
    dados = sintetico.gerar(prontuarios)
    assert dados == sintetico.gerar(prontuarios)
    alergias = {p["prontuario"]: p["alergias"] for p in prontuarios}
    resultado = avaliar_internacao(dados, alergias)
    for modalidade, numeros in resultado["por_modalidade"].items():
        assert numeros["detectados"] == numeros["eventos_rotulados"], modalidade
    assert sum(n["eventos_rotulados"] for n in resultado["por_modalidade"].values()) >= 12
    assert not resultado["anomalias_fora_de_rotulo"]


def test_preparar_monitoramento_nao_duplica(tmp_path, prontuarios):
    banco = BancoHospital(tmp_path / "h.db")
    banco.popular(prontuarios)
    pasta = tmp_path / "monitoramento"
    primeira = sintetico.preparar_monitoramento(banco, pasta=pasta)
    sintetico.preparar_monitoramento(banco, pasta=pasta)
    total = banco.consultar_somente_leitura("SELECT count(*) AS n FROM prescricoes")[0]["n"]
    assert total == primeira["carregados"]["prescricoes"] > 0
    sintetico.preparar_monitoramento(banco, forcar=True, pasta=pasta)
    assert banco.consultar_somente_leitura("SELECT count(*) AS n FROM prescricoes")[0]["n"] == total


def test_cli_simular_emite_alertas(tmp_path, prontuarios, monkeypatch, capsys):
    from assistente_medico import auditoria
    from assistente_medico.cli import main
    from assistente_medico.multimodal import alertas, cli_monitoramento

    banco = BancoHospital(tmp_path / "h.db")
    banco.popular(prontuarios)
    caminhos = dataclasses.replace(auditoria.CAMINHOS, logs=tmp_path / "logs")
    monkeypatch.setattr(auditoria, "CAMINHOS", caminhos)
    monkeypatch.setattr(alertas, "CAMINHOS", caminhos)
    monkeypatch.setattr(cli_monitoramento, "preparar_banco", lambda: banco)
    monkeypatch.setattr(sintetico, "PASTA_PADRAO", tmp_path / "monitoramento")

    assert main(["monitorar", "simular", "--horas", "72", "--sem-queda"]) == 0
    saida = capsys.readouterr().out
    assert "ALERTA" in saida and "alertas emitidos" in saida
    assert banco.listar_alertas(limite=200)
    assert (tmp_path / "logs" / "alertas.jsonl").exists()
