"""Analise de audio: medidas acusticas, indicadores, termos, Azure (simulada) e pipeline."""

from __future__ import annotations

import dataclasses
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from assistente_medico.config import ConfigAzure
from assistente_medico.dados.banco import BancoHospital
from assistente_medico.multimodal.audio import indicadores, roteiros, termos
from assistente_medico.multimodal.audio.acustica import MedidasVoz, medir
from assistente_medico.multimodal.audio.consulta import analisar_consulta, salvar_relatorio
from assistente_medico.multimodal.azure import fala, linguagem
from assistente_medico.multimodal.azure.cliente import RespostaAzure
from assistente_medico.multimodal.azure.linguagem import EntidadeSaude

TAXA = 16000


def _fala_sintetica(grupos: int = 4, silabas: int = 6, duracao_silaba: float = 0.14,
                    intervalo: float = 0.08, pausa: float = 0.7, f0: float = 140.0,
                    amplitude: float = 0.4) -> np.ndarray:
    """Sequencia de 'silabas' harmonicas separadas por intervalos curtos e pausas longas."""
    gerador = np.random.default_rng(0)
    partes = [np.zeros(int(0.3 * TAXA))]
    for grupo in range(grupos):
        for _ in range(silabas):
            t = np.arange(int(duracao_silaba * TAXA)) / TAXA
            tom = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
            partes.append(amplitude * tom * np.hanning(len(t)))
            partes.append(0.001 * gerador.standard_normal(int(intervalo * TAXA)))
        if grupo < grupos - 1:
            partes.append(0.001 * gerador.standard_normal(int(pausa * TAXA)))
    partes.append(np.zeros(int(0.3 * TAXA)))
    return np.concatenate(partes)


def _salvar_wav(caminho: Path, sinal: np.ndarray) -> Path:
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(TAXA)
        arquivo.writeframes((np.clip(sinal, -1, 1) * 32767).astype("<i2").tobytes())
    return caminho


def test_medidas_temporais_de_fala_sintetica():
    m = medir(_fala_sintetica(grupos=4, silabas=6), TAXA)
    assert m.pausas == 3
    assert 22 <= m.silabas <= 26
    assert 3.5 <= m.taxa_articulacao <= 5.5
    assert m.silabas_por_grupo == pytest.approx(m.silabas / 4)
    assert m.f0_media_hz == pytest.approx(140, rel=0.1)
    assert m.jitter_local is not None and m.jitter_local < 0.02
    assert "intervalos_pausa" not in m.como_dicionario()


def test_fala_entrecortada_tem_grupos_menores_e_mais_pausas():
    normal = medir(_fala_sintetica(grupos=3, silabas=8), TAXA)
    ofegante = medir(_fala_sintetica(grupos=8, silabas=3), TAXA)
    assert ofegante.silabas_por_grupo < normal.silabas_por_grupo / 2
    assert ofegante.pausas_por_minuto > normal.pausas_por_minuto


def test_audio_mudo_e_rejeitado():
    with pytest.raises(ValueError):
        medir(np.zeros(TAXA), TAXA)


def _medidas(**ajustes) -> MedidasVoz:
    base = dict(duracao_s=20, tempo_fala_s=16, pausas=4, pausa_media_s=0.6, fracao_pausa=0.2,
                pausas_por_minuto=12, silabas=80, taxa_fala=4, taxa_articulacao=5,
                silabas_por_grupo=16, irregularidade_ritmo=0.3, f0_media_hz=120,
                f0_variacao_st=3.0, intensidade_media_db=70, jitter_local=0.01,
                shimmer_local=0.05, hnr_db=18)
    return MedidasVoz(**{**base, **ajustes})


@pytest.mark.parametrize(("ajustes", "esperado"), [
    ({}, []),
    ({"intensidade_media_db": 65, "taxa_articulacao": 4.2, "pausa_media_s": 1.2}, ["fadiga"]),
    ({"silabas_por_grupo": 5, "pausas_por_minuto": 30}, ["dispneia"]),
    ({"taxa_articulacao": 3.0, "silabas_por_grupo": 8, "pausas_por_minuto": 20}, ["disartria"]),
])
def test_indicadores_contra_a_linha_de_base(ajustes, esperado):
    resultado = indicadores.avaliar(_medidas(**ajustes), _medidas())
    assert [i.condicao for i in resultado if i.suspeita] == esperado


def test_indicadores_sem_linha_de_base_usam_referencia():
    resultado = indicadores.avaliar(_medidas(silabas_por_grupo=3, pausas_por_minuto=40))
    assert [i.condicao for i in resultado if i.suspeita] == ["dispneia"]


def test_termos_locais_com_negacao_e_palavras_intercaladas():
    mencoes = termos.consolidar(termos.detectar_local(
        "Nao tenho dor no peito. A minha fala estava enrolada e eu fico cansado para respirar."))
    situacao = {m.termo.chave: m.negada for m in mencoes}
    assert situacao == {"dor_toracica": True, "deficit_neurologico": False, "dispneia": False}


def test_classificacao_de_entidades_da_azure():
    entidades = [
        EntidadeSaude("dor torácica", "SymptomOrSign", 0.9, certeza="negative"),
        EntidadeSaude("falta de ar", "SymptomOrSign", 0.9, condicionalidade="conditional"),
        EntidadeSaude("cansado", "SymptomOrSign", 0.9),
        EntidadeSaude("síncope", "Diagnosis", 0.9, codigos=("UMLS:C0039070",)),
        EntidadeSaude("metformina", "MedicationName", 0.9),
    ]
    mencoes = {m.termo.chave: m for m in termos.classificar_entidades(entidades)}
    assert set(mencoes) == {"dor_toracica", "dispneia", "fadiga", "sincope"}
    assert mencoes["dor_toracica"].negada and mencoes["dispneia"].condicional


def test_ssml_simula_cada_condicao():
    texto = "Eu estou com falta de ar desde ontem. A tosse continua."
    normal = roteiros.ssml(texto, "pt-BR-FranciscaNeural", "normal")
    dispneia = roteiros.ssml(texto, "pt-BR-FranciscaNeural", "dispneia")
    fadiga = roteiros.ssml(texto, "pt-BR-FranciscaNeural", "fadiga")
    assert "<break" not in normal and dispneia.count("<break") >= 2
    assert 'volume="-40%"' in fadiga and fadiga.count("<break") == 2
    assert roteiros.ssml("a < b", "v", "disartria").count("&lt;") == 1
    with pytest.raises(ValueError):
        roteiros.ssml(texto, "v", "rouquidao")


def test_cortes_respeitam_o_limite_e_caem_nas_pausas():
    pontos = fala.cortes(130.0, [(40, 41), (50, 51), (100, 101), (120, 121)], limite_s=55)
    assert pontos == [50.5, 100.5]
    assert fala.cortes(30.0, []) == []


CONFIG = ConfigAzure(chave_fala="k", regiao_fala="canadacentral", chave_linguagem="k",
                     endpoint_linguagem="https://recurso.cognitiveservices.azure.com")


def test_transcricao_em_trechos_reajusta_os_tempos(tmp_path, monkeypatch):
    sinal = np.concatenate([_fala_sintetica(grupos=1, silabas=4)] * 2 + [np.zeros(TAXA * 60)])
    arquivo = _salvar_wav(tmp_path / "longo.wav", sinal)
    chamadas = []

    def falso(metodo, url, chave, corpo=None, cabecalhos=None, timeout_s=30):
        chamadas.append(len(corpo))
        resposta = {"RecognitionStatus": "Success", "NBest": [{
            "Display": f"trecho {len(chamadas)}", "Confidence": 0.9,
            "Words": [{"Word": "ola", "Offset": 5_000_000, "Duration": 2_000_000}]}]}
        return RespostaAzure(200, json.dumps(resposta).encode(), {})

    monkeypatch.setattr(fala, "requisitar", falso)
    monkeypatch.setattr(fala.time, "sleep", lambda s: None)
    transcricao = fala.transcrever(arquivo, pausas=[(30.0, 31.0)], config=CONFIG)
    assert transcricao.trechos == 2 and len(chamadas) == 2
    assert transcricao.texto == "trecho 1 trecho 2"
    assert [p.inicio_s for p in transcricao.palavras] == [0.5, 31.0]
    assert transcricao.confianca == 0.9


def test_health_assincrono_com_negacao(monkeypatch):
    respostas = iter([
        RespostaAzure(202, b"", {"Operation-Location": "https://job/1"}),
        RespostaAzure(200, json.dumps({"status": "running"}).encode(), {}),
        RespostaAzure(200, json.dumps({"status": "succeeded", "tasks": {"items": [{"results": {
            "errors": [], "documents": [{"entities": [
                {"text": "dor no peito", "category": "SymptomOrSign", "confidenceScore": 0.98,
                 "offset": 9, "assertion": {"certainty": "negative"},
                 "links": [{"dataSource": "UMLS", "id": "C0008031"}]}]}]}}]}}).encode(), {}),
    ])
    monkeypatch.setattr(linguagem, "requisitar", lambda *a, **k: next(respostas))
    [entidade] = linguagem.analisar_saude("Nao tenho dor no peito", CONFIG, dormir=lambda s: None)
    assert entidade.negada and entidade.codigos == ("UMLS:C0008031",)


def test_sentimento_com_opinioes(monkeypatch):
    corpo = {"results": {"documents": [{"sentiment": "negative", "confidenceScores": {
        "positive": 0.0, "neutral": 0.05, "negative": 0.95}, "sentences": [{
            "text": "Estou com medo", "sentiment": "negative",
            "confidenceScores": {"positive": 0, "neutral": 0.05, "negative": 0.95},
            "targets": [{"text": "medo", "sentiment": "negative"}]}]}]}}
    monkeypatch.setattr(linguagem, "requisitar",
                        lambda *a, **k: RespostaAzure(200, json.dumps(corpo).encode(), {}))
    resultado = linguagem.analisar_sentimento("Estou com medo", CONFIG)
    assert resultado["sentimento"] == "negative"
    assert resultado["opinioes"] == [{"alvo": "medo", "sentimento": "negative"}]


@pytest.fixture
def banco_audio(tmp_path, prontuarios) -> BancoHospital:
    banco = BancoHospital(tmp_path / "h.db")
    banco.popular(prontuarios)
    return banco


def test_pipeline_local_pseudonimiza_e_gera_anomalias(tmp_path, banco_audio, prontuarios):
    from assistente_medico.auditoria import Auditoria
    from assistente_medico.multimodal import CentralAlertas

    paciente = next(p for p in prontuarios if p["prontuario"] == "771203")
    _salvar_wav(tmp_path / "771203_basal.wav", _fala_sintetica(grupos=3, silabas=8))
    consulta = _salvar_wav(tmp_path / "771203_consulta.wav", _fala_sintetica(grupos=3, silabas=8))
    consulta.with_suffix(".txt").write_text(
        f"Meu nome e {paciente['nome']}. Estou com dor no peito apertando. Nao tenho falta de ar.",
        encoding="utf-8")
    central = CentralAlertas(banco_audio, notificadores=[],
                             auditoria=Auditoria(banco=banco_audio, caminho_jsonl=tmp_path / "a.jsonl"))

    resultado = analisar_consulta(consulta, banco=banco_audio, central=central, usar_azure=False)

    assert resultado.prontuario == "771203"
    assert resultado.fontes["comparacao"] == "linha de base do paciente"
    assert paciente["nome"] not in resultado.texto_anonimizado
    assert "[PACIENTE_" in resultado.texto_anonimizado
    assert [i.condicao for i in resultado.indicadores if i.suspeita] == []
    tipos = {a.tipo: a.severidade for a in resultado.anomalias}
    assert tipos == {"termo_critico_dor_toracica": "critica"}
    [alerta] = banco_audio.listar_alertas(prontuario="771203")
    assert alerta["categoria"] == "anomalia_texto" and alerta["prioridade"] == "maxima"
    analise = banco_audio.obter_analise(resultado.analise_id)
    assert analise["modalidade"] == "audio" and "texto" not in (analise["resumo"]["transcricao"] or {})

    relatorio = salvar_relatorio(resultado, tmp_path / "relatorios")
    conteudo = relatorio.read_text()
    assert "dor toracica" in conteudo and "negado" in conteudo
    assert paciente["nome"] not in relatorio.with_suffix(".json").read_text()


def test_leitura_do_voiced(tmp_path):
    from assistente_medico.multimodal.audio.avaliacao import ler_voiced

    amostras = (np.sin(np.arange(800) / 5) * 1000).astype("<i4")
    (tmp_path / "voice001.dat").write_bytes(amostras.tobytes())
    (tmp_path / "voice001.hea").write_text(
        "voice001 1 8000 800\nvoice001.dat 32 1000(10)/NU 0 0 0 0 0 voice\n")
    (tmp_path / "voice001-info.txt").write_text("ID\tvoice001\nDiagnosis:\thealthy\n")
    [registro] = ler_voiced(tmp_path)
    assert registro["saudavel"] and registro["taxa"] == 8000
    assert registro["sinal"][0] == pytest.approx(-0.01)


def test_cli_audio_analisar_sem_azure(tmp_path, banco_audio, monkeypatch, capsys):
    from assistente_medico import auditoria
    from assistente_medico.cli import main
    from assistente_medico.multimodal import alertas, cli_audio
    from assistente_medico.multimodal.audio import roteiros as modulo_roteiros

    caminhos = dataclasses.replace(auditoria.CAMINHOS, logs=tmp_path / "logs")
    monkeypatch.setattr(auditoria, "CAMINHOS", caminhos)
    monkeypatch.setattr(alertas, "CAMINHOS", caminhos)
    monkeypatch.setattr(cli_audio, "preparar_banco", lambda: banco_audio)
    monkeypatch.setattr(modulo_roteiros, "PASTA_AUDIO", tmp_path / "audio")
    arquivo = _salvar_wav(tmp_path / "553119_consulta.wav", _fala_sintetica())
    arquivo.with_suffix(".txt").write_text("A minha fala esta enrolada e o braco esta fraco.")

    assert main(["audio", "analisar", str(arquivo), "--sem-azure"]) == 0
    saida = capsys.readouterr().out
    assert "sinal neurologico focal" in saida and "alerta(s)" in saida
    assert (tmp_path / "audio" / "relatorios" / "553119_consulta.md").exists()


def test_wav_compativel_converte_taxa(tmp_path):
    caminho = tmp_path / "8k.wav"
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(8000)
        arquivo.writeframes((np.sin(np.arange(8000) / 3) * 8000).astype("<i2").tobytes())
    convertido = fala.wav_compativel(caminho)
    with wave.open(io.BytesIO(convertido)) as arquivo:
        assert arquivo.getframerate() == 16000 and arquivo.getnchannels() == 1
