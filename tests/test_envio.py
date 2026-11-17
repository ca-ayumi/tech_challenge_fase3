"""Envio de gravacoes pela equipe (upload na aba Monitoramento), sem YOLO nem Azure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from assistente_medico.auditoria import Auditoria
from assistente_medico.dados.banco import BancoHospital
from assistente_medico.multimodal import Anomalia, CentralAlertas, envio, relogio


@pytest.fixture
def banco_envio(tmp_path, prontuarios) -> BancoHospital:
    banco = BancoHospital(tmp_path / "h.db")
    banco.popular(prontuarios)
    return banco


@pytest.fixture
def central(banco_envio, tmp_path) -> CentralAlertas:
    return CentralAlertas(banco_envio, notificadores=[],
                          auditoria=Auditoria(banco=banco_envio, caminho_jsonl=tmp_path / "a.jsonl"))


def test_salvar_envio_valida_formato_e_reaproveita_conteudo(tmp_path):
    with pytest.raises(envio.EnvioInvalido):
        envio.salvar_envio("laudo.pdf", b"x", "video", pasta=tmp_path)
    with pytest.raises(envio.EnvioInvalido):
        envio.salvar_envio("consulta.wav", b"", "audio", pasta=tmp_path)
    with pytest.raises(envio.EnvioInvalido):
        envio.salvar_envio("consulta.mp4", b"x", "audio", pasta=tmp_path)
    primeiro = envio.salvar_envio("Sessão fisio (1).MP4", b"conteudo", "video", pasta=tmp_path)
    assert primeiro.parent == tmp_path / "video" and primeiro.suffix == ".mp4"
    assert " " not in primeiro.name and "(" not in primeiro.name
    assert envio.salvar_envio("outro_nome.mp4", b"conteudo", "video", pasta=tmp_path).name[:10] == \
        primeiro.name[:10]
    assert envio.salvar_envio("x.mp4", b"outro", "video", pasta=tmp_path) != primeiro


def test_video_enviado_vai_para_central_e_mostra_progresso(banco_envio, central, tmp_path, monkeypatch):
    from assistente_medico.multimodal.video import analise, extracao

    instante = relogio.agora(banco_envio).isoformat(timespec="seconds")
    recebido = {}

    def extrair_falso(caminho, progresso=None, **_):
        for indice in (0, 5, 10):
            progresso(indice)
        return "sequencia"

    def analisar_falso(caminho, tipo, prontuario, config_areas, banco, central, pasta_saida, anotar,
                       sequencia):
        recebido.update(tipo=tipo, areas=config_areas, sequencia=sequencia, anotar=anotar)
        anomalias = [Anomalia("video", "video_queda", "critica", "queda pela camera", prontuario,
                              instante, instante)]
        central.processar(anomalias)
        return SimpleNamespace(anomalias=anomalias, arquivos_gerados={})

    monkeypatch.setattr(extracao, "extrair", extrair_falso)
    monkeypatch.setattr(analise, "analisar_video", analisar_falso)
    monkeypatch.setattr(envio, "_quadros", lambda caminho: 10)
    etapas: list[float] = []
    caminho = envio.salvar_envio("queda.mp4", b"video", "video", pasta=tmp_path)

    resultado = envio.analisar_video_enviado(banco_envio, caminho, "875431", tipo="quarto",
                                             central=central, pasta_saida=tmp_path / "rel",
                                             progresso=lambda f, _texto: etapas.append(f))

    assert recebido == {"tipo": "quarto", "areas": None, "sequencia": "sequencia", "anotar": True}
    assert etapas == sorted(etapas) and etapas[-1] == 1.0
    assert [a["categoria"] for a in resultado.alertas_novos] == ["anomalia_video"]
    assert resultado.relatorio == tmp_path / "rel" / f"{caminho.stem}.md"


def test_audio_enviado_dispara_fusao_e_nao_repete(banco_envio, central, tmp_path, monkeypatch):
    from assistente_medico.multimodal.audio import consulta

    instante = relogio.agora(banco_envio).isoformat(timespec="seconds")
    bases: list = []

    def analisar_falso(caminho, prontuario, basal, banco, central, usar_azure):
        bases.append((basal, usar_azure))
        anomalias = [
            Anomalia("audio", "voz_disartria", "alta", "fala arrastada", prontuario, instante, instante),
            Anomalia("texto", "termo_critico_deficit_neurologico", "critica", "fala enrolada",
                     prontuario, instante, instante),
        ]
        central.processar(anomalias)
        return SimpleNamespace(anomalias=anomalias, fontes={})

    monkeypatch.setattr(consulta, "analisar_consulta", analisar_falso)
    monkeypatch.setattr(consulta, "salvar_relatorio", lambda r, pasta: pasta / "rel.md")
    caminho = envio.salvar_envio("consulta.wav", b"audio", "audio", pasta=tmp_path)

    primeiro = envio.analisar_audio_enviado(banco_envio, caminho, "553119", usar_azure=False,
                                            central=central, pasta_relatorios=tmp_path)
    assert bases == [(None, False)]
    assert len(primeiro.alertas_novos) == 2
    assert ["Suspeita de AVC" in a["mensagem"] for a in primeiro.consolidados_novos] == [True]

    de_novo = envio.analisar_audio_enviado(banco_envio, caminho, "553119", usar_azure=False,
                                           central=central, pasta_relatorios=tmp_path)
    assert not de_novo.alertas_novos and not de_novo.consolidados_novos


def test_basal_do_paciente(tmp_path, monkeypatch):
    from assistente_medico.multimodal.audio import roteiros

    monkeypatch.setattr(roteiros, "PASTA_AUDIO", tmp_path)
    assert envio.basal_do_paciente("553119") is None
    (tmp_path / "consultas").mkdir()
    (tmp_path / "consultas" / "553119_basal.wav").write_bytes(b"x")
    assert envio.basal_do_paciente("553119") == tmp_path / "consultas" / "553119_basal.wav"


def _pagina_envio(caminho_banco: str) -> None:
    from assistente_medico.app.envio import secao_envio
    from assistente_medico.dados.banco import BancoHospital

    secao_envio(BancoHospital(caminho_banco))


def test_tela_de_envio_tem_um_campo_para_video_e_audio(banco_envio):
    from streamlit.testing.v1 import AppTest

    tela = AppTest.from_function(_pagina_envio, args=(str(banco_envio.caminho),)).run()
    assert not tela.exception
    assert tela.selectbox(key="envio_paciente").options
    assert not tela.button and not tela.radio


def test_midia_reconhecida_pela_extensao():
    assert envio.midia_do_arquivo("553119_consulta.WAV") == "audio"
    assert envio.midia_do_arquivo("sessao.webm") == "video"
    assert envio.midia_do_arquivo("laudo.pdf") is None


def test_bidmc_ausente_e_listagem(tmp_path, monkeypatch):
    from assistente_medico.multimodal import demonstracao

    monkeypatch.setattr(demonstracao, "registros_bidmc", lambda: [tmp_path / "bidmc_07_Numerics.csv"])
    assert demonstracao.numeros_bidmc() == [7]
    with pytest.raises(FileNotFoundError):
        demonstracao.leituras_bidmc(3)


def _pagina_simulacao(caminho_banco: str) -> None:
    from assistente_medico.app.simulacao import secao_azure, secao_simulacao
    from assistente_medico.dados.banco import BancoHospital

    secao_azure()
    secao_simulacao(BancoHospital(caminho_banco))


def test_tela_de_simulacao_renderiza(banco_envio, monkeypatch):
    from streamlit.testing.v1 import AppTest

    tela = AppTest.from_function(_pagina_simulacao, args=(str(banco_envio.caminho),)).run()
    assert not tela.exception
    assert "Testar conexão com a Azure" in [b.label for b in tela.button]
    tela.radio(key="simulacao_cenario").set_value("internacao").run()
    assert not tela.exception
    assert "▶ Iniciar simulação" in [b.label for b in tela.button]
