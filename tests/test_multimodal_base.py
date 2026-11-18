"""Base do monitoramento multimodal: tipos, central de alertas, monitor e Azure."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timedelta

import pytest

from assistente_medico.auditoria import Auditoria
from assistente_medico.config import ConfigAzure
from assistente_medico.dados.banco import BancoHospital
from assistente_medico.multimodal import (
    Anomalia,
    CentralAlertas,
    Leitura,
    Monitor,
    NotificadorArquivo,
    NotificadorWebhook,
)
from assistente_medico.multimodal.azure import cliente as cliente_azure
from assistente_medico.multimodal.azure import requisitar, verificar_azure
from assistente_medico.multimodal.azure.cliente import ErroAzure


@pytest.fixture
def banco_limpo(tmp_path, prontuarios) -> BancoHospital:
    banco = BancoHospital(tmp_path / "hospital.db")
    banco.popular(prontuarios)
    return banco


class NotificadorMemoria:
    nome = "memoria"

    def __init__(self) -> None:
        self.recebidos: list[dict] = []

    def notificar(self, alerta: dict) -> None:
        self.recebidos.append(alerta)


class NotificadorQuebrado:
    nome = "quebrado"

    def notificar(self, alerta: dict) -> None:
        raise ConnectionError("canal fora do ar")


@pytest.fixture
def canal() -> NotificadorMemoria:
    return NotificadorMemoria()


@pytest.fixture
def central(banco_limpo, canal, tmp_path) -> CentralAlertas:
    return CentralAlertas(
        banco_limpo,
        auditoria=Auditoria(banco=banco_limpo, caminho_jsonl=tmp_path / "auditoria.jsonl"),
        notificadores=[canal],
        severidade_minima="moderada",
        janela_supressao=timedelta(minutes=30),
    )


def _anomalia(instante: str, severidade: str = "alta", tipo: str = "taquicardia",
              prontuario: str = "884521") -> Anomalia:
    return Anomalia(
        modalidade="sinais_vitais", tipo=tipo, severidade=severidade,
        descricao=f"FC elevada ({severidade})", prontuario=prontuario,
        inicio=instante, fim=instante, escore=0.9, detector="teste",
        evidencias={"fc": 131},
    )


def test_anomalia_valida_modalidade_e_severidade():
    with pytest.raises(ValueError):
        Anomalia(modalidade="olfato", tipo="x", severidade="alta", descricao="x")
    with pytest.raises(ValueError):
        Anomalia(modalidade="audio", tipo="x", severidade="gravissima", descricao="x")
    assert _anomalia("2026-09-14T22:00:00", "critica").prioridade == "maxima"


def test_central_emite_alerta_notifica_e_grava_anomalia(central, canal, banco_limpo):
    [resultado] = central.processar([_anomalia("2026-09-14T22:00:00")])

    assert resultado.emitido and resultado.alerta_id
    assert resultado.canais == ("memoria",)
    assert canal.recebidos[0]["prioridade"] == "alta"
    alerta = banco_limpo.listar_alertas(prontuario="884521")[0]
    assert alerta["categoria"] == "anomalia_sinais_vitais"
    assert alerta["fonte"] == "monitoramento/sinais_vitais/teste"
    [anomalia] = banco_limpo.listar_anomalias(prontuario="884521")
    assert anomalia["alerta_id"] == resultado.alerta_id
    assert anomalia["evidencias"] == {"fc": 131}


def test_central_suprime_repeticao_dentro_da_janela(central, canal):
    primeiro = central.processar([_anomalia("2026-09-14T22:00:00")])[0]
    repetido = central.processar([_anomalia("2026-09-14T22:10:00")])[0]
    depois_da_janela = central.processar([_anomalia("2026-09-14T22:45:00")])[0]

    assert primeiro.emitido
    assert repetido.decisao == "suprimido_repeticao" and repetido.alerta_id is None
    assert depois_da_janela.emitido
    assert len(canal.recebidos) == 2


def test_central_nao_suprime_agravamento(central):
    central.processar([_anomalia("2026-09-14T22:00:00", "moderada")])
    agravado = central.processar([_anomalia("2026-09-14T22:05:00", "critica")])[0]
    assert agravado.emitido


def test_central_supressao_e_por_paciente_e_tipo(central):
    central.processar([_anomalia("2026-09-14T22:00:00")])
    outro_tipo = central.processar([_anomalia("2026-09-14T22:01:00", tipo="dessaturacao")])[0]
    outro_paciente = central.processar([_anomalia("2026-09-14T22:01:00", prontuario="999")])[0]
    assert outro_tipo.emitido and outro_paciente.emitido


def test_central_registra_mas_nao_alerta_abaixo_do_limiar(central, canal, banco_limpo):
    [resultado] = central.processar([_anomalia("2026-09-14T22:00:00", "baixa")])
    assert resultado.decisao == "abaixo_do_limiar"
    assert not canal.recebidos
    assert banco_limpo.listar_anomalias(prontuario="884521")[0]["alerta_id"] is None


def test_falha_de_canal_nao_derruba_a_central(banco_limpo, canal, tmp_path):
    central = CentralAlertas(
        banco_limpo,
        auditoria=Auditoria(banco=banco_limpo, caminho_jsonl=tmp_path / "a.jsonl"),
        notificadores=[NotificadorQuebrado(), canal],
    )
    [resultado] = central.processar([_anomalia("2026-09-14T22:00:00")])
    assert resultado.emitido and resultado.canais == ("memoria",)


def test_reconhecimento_fecha_o_alerta_e_audita(central, banco_limpo):
    [resultado] = central.processar([_anomalia("2026-09-14T22:00:00")])
    assert central.reconhecer(resultado.alerta_id, profissional="enf. Ana")
    assert not central.reconhecer(resultado.alerta_id, profissional="enf. Ana")
    assert not banco_limpo.listar_alertas(prontuario="884521")
    eventos = banco_limpo.trilha_auditoria(f"ALERTA-{resultado.alerta_id}")
    assert eventos[0]["evento"] == "alerta_reconhecido"
    assert eventos[0]["usuario"] == "enf. Ana"


def test_notificador_arquivo_grava_jsonl(tmp_path):
    destino = tmp_path / "alertas.jsonl"
    NotificadorArquivo(destino).notificar({"prioridade": "alta", "mensagem": "teste"})
    assert json.loads(destino.read_text())["mensagem"] == "teste"


def test_notificador_webhook_envia_texto(monkeypatch):
    enviados = []

    class Resposta(io.BytesIO):
        status = 200

    def falso_urlopen(requisicao, timeout):
        enviados.append(json.loads(requisicao.data))
        return Resposta(b"ok")

    monkeypatch.setattr("urllib.request.urlopen", falso_urlopen)
    NotificadorWebhook("https://exemplo.invalido/webhook").notificar(
        {"prioridade": "maxima", "prontuario": "884521", "mensagem": "SpO2 82%"})
    assert enviados[0]["text"] == "[MAXIMA] prontuario 884521: SpO2 82%"


class DetectorLimiar:
    nome = "limiar_fc"
    modalidades = ("sinais_vitais",)

    def atualizar(self, leitura: Leitura) -> list[Anomalia]:
        if leitura.valores["fc"] <= 120:
            return []
        instante = leitura.instante.isoformat(timespec="seconds")
        return [_anomalia(instante, prontuario=leitura.prontuario)]


def test_monitor_reproduz_fluxo_com_aceleracao(central):
    inicio = datetime(2026, 9, 14, 22, 0)
    leituras = [
        Leitura("884521", "sinais_vitais", inicio + timedelta(minutes=5 * i), {"fc": fc})
        for i, fc in enumerate([98, 104, 125, 131, 128, 101])
    ] + [Leitura("884521", "movimentacao", inicio, {"atividade": 0.0})]
    esperas: list[float] = []
    vistos = []

    resumo = Monitor(central, [DetectorLimiar()]).reproduzir(
        leituras, aceleracao=60, dormir=esperas.append,
        ao_processar=lambda leitura, resultados: vistos.append(len(resultados)),
    )

    assert resumo.leituras == 7
    assert resumo.anomalias == 3
    assert len(resumo.alertas) == 1
    assert resumo.suprimidos == 2
    assert esperas[0] == pytest.approx(5.0)
    assert vistos[2] == 1


def test_diagnostico_sem_credenciais_nao_acessa_rede(monkeypatch):
    def proibido(*args, **kwargs):
        raise AssertionError("nao deveria chamar a rede")

    monkeypatch.setattr(cliente_azure.urllib.request, "urlopen", proibido)
    status = verificar_azure(ConfigAzure(chave_fala="", regiao_fala="", chave_linguagem="",
                                         endpoint_linguagem=""))
    assert [s.configurado for s in status] == [False, False]
    assert not any(s.disponivel for s in status)


class _RespostaFalsa(io.BytesIO):
    def __init__(self, corpo: bytes, status: int = 200) -> None:
        super().__init__(corpo)
        self.status = status
        self.headers = {}


def test_diagnostico_com_credenciais(monkeypatch):
    urls = []

    def falso_urlopen(requisicao, timeout):
        urls.append(requisicao.full_url)
        assert requisicao.headers["Ocp-apim-subscription-key"] in {"k1", "k2"}
        if "issueToken" in requisicao.full_url:
            return _RespostaFalsa(b"token")
        corpo = {"results": {"documents": [{"detectedLanguage": {"iso6391Name": "pt"}}]}}
        return _RespostaFalsa(json.dumps(corpo).encode())

    monkeypatch.setattr(cliente_azure.urllib.request, "urlopen", falso_urlopen)
    fala, linguagem = verificar_azure(ConfigAzure(
        chave_fala="k1", regiao_fala="canadacentral", chave_linguagem="k2",
        endpoint_linguagem="https://recurso.cognitiveservices.azure.com"))

    assert fala.disponivel and linguagem.disponivel
    assert urls[0].startswith("https://canadacentral.api.cognitive.microsoft.com")
    assert "pt" in linguagem.detalhe


def test_cliente_repete_em_limite_de_taxa(monkeypatch):
    chamadas = []
    esperas: list[float] = []

    def falso_urlopen(requisicao, timeout):
        chamadas.append(1)
        if len(chamadas) == 1:
            raise urllib.error.HTTPError(requisicao.full_url, 429, "Too Many Requests",
                                         {"Retry-After": "3"}, io.BytesIO(b"limite"))
        return _RespostaFalsa(b'{"ok": true}')

    monkeypatch.setattr(cliente_azure.urllib.request, "urlopen", falso_urlopen)
    resposta = requisitar("POST", "https://x.invalido", "k", corpo={"a": 1},
                          dormir=esperas.append)
    assert resposta.json() == {"ok": True}
    assert esperas == [3.0]


def test_cliente_falha_definitiva_em_chave_invalida(monkeypatch):
    def falso_urlopen(requisicao, timeout):
        raise urllib.error.HTTPError(requisicao.full_url, 401, "Unauthorized", {},
                                     io.BytesIO(b"chave invalida"))

    monkeypatch.setattr(cliente_azure.urllib.request, "urlopen", falso_urlopen)
    with pytest.raises(ErroAzure) as erro:
        requisitar("POST", "https://x.invalido", "k", dormir=lambda s: None)
    assert erro.value.status == 401


def test_banco_guarda_prescricoes_e_movimentacao(banco_limpo):
    banco_limpo.registrar_prescricoes([
        {"prontuario": "884521", "medicamento": "Enoxaparina", "dose": 40, "unidade": "mg",
         "via": "SC", "intervalo_horas": 24, "prescrito_em": "2026-09-14T22:00"},
        {"prontuario": "884521", "medicamento": "Enoxaparina", "dose": 400, "unidade": "mg",
         "via": "SC", "intervalo_horas": 24, "prescrito_em": "2026-09-15T22:00"},
    ])
    banco_limpo.registrar_movimentacao([
        {"prontuario": "884521", "registrado_em": "2026-09-14T23:00", "atividade": 0.1,
         "posicao": "deitado", "fonte": "sensor_leito"},
    ])
    assert [p["dose"] for p in banco_limpo.prescricoes("884521")] == [40, 400]
    assert banco_limpo.movimentacao("884521")[0]["posicao"] == "deitado"
    assert banco_limpo.consultar_somente_leitura("SELECT count(*) AS n FROM prescricoes")[0]["n"] == 2

    analise = banco_limpo.abrir_analise("prescricao", prontuario="884521", origem="teste")
    banco_limpo.concluir_analise(analise, {"anomalias": 1})
    assert banco_limpo.obter_analise(analise)["resumo"] == {"anomalias": 1}
