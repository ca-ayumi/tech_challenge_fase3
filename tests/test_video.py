"""Analise de video com esqueletos sinteticos (sem YOLO, sem arquivo de video)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from assistente_medico.dados.banco import BancoHospital
from assistente_medico.multimodal.video import areas, avaliacao, exercicio, quedas
from assistente_medico.multimodal.video import cinematica as cin
from assistente_medico.multimodal.video.analise import analisar_video
from assistente_medico.multimodal.video.extracao import (
    SequenciaPose,
    aplicar_rastreamento,
    rastrear,
    recorte_padrao,
)

FPS = 30.0
EM_PE = np.array([
    (160, 40), (155, 35), (165, 35), (150, 38), (170, 38), (145, 70), (175, 70), (140, 110),
    (180, 110), (138, 145), (182, 145), (150, 130), (170, 130), (150, 175), (170, 175),
    (150, 220), (170, 220)], dtype=float)


def _girar(pontos: np.ndarray, graus: float) -> np.ndarray:
    """Gira o corpo em torno do ponto medio dos tornozelos (cair para o lado)."""
    centro = (pontos[15] + pontos[16]) / 2
    rad = np.radians(graus)
    rotacao = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return (pontos - centro) @ rotacao.T + centro


def _sequencia(angulos: list[float], objetos: list[dict] | None = None) -> SequenciaPose:
    pontos, caixas = [], []
    for angulo in angulos:
        xy = _girar(EM_PE, angulo)
        pontos.append(np.column_stack([xy, np.full(17, 0.9)]))
        caixas.append([xy[:, 0].min() - 5, xy[:, 1].min() - 5, xy[:, 0].max() + 5, xy[:, 1].max() + 5])
    n = len(angulos)
    return SequenciaPose(origem="sintetico.mp4", fps=FPS, largura=480, altura=320,
                         tempos=np.arange(n) / FPS, pontos=np.array(pontos),
                         caixas=np.array(caixas), confianca=np.full(n, 0.9),
                         objetos=objetos or [])


def _transicao(segundos_em_pe: float, segundos_queda: float, segundos_chao: float) -> list[float]:
    em_pe = [0.0] * int(segundos_em_pe * FPS)
    caindo = list(np.linspace(0, 90, max(int(segundos_queda * FPS), 2)))
    return em_pe + caindo + [90.0] * int(segundos_chao * FPS)


def _detectar(sequencia: SequenciaPose) -> list[quedas.EventoQueda]:
    return quedas.detectar(sequencia, cin.calcular(sequencia))


def test_cinematica_de_um_corpo_em_pe_e_deitado():
    em_pe = cin.calcular(_sequencia([0.0] * 10))
    assert np.nanmax(em_pe.angulo_tronco) < 5
    assert em_pe.articulacoes["joelho_esq"][5] == pytest.approx(180, abs=1)
    deitado = cin.calcular(_sequencia([90.0] * 10))
    assert np.nanmin(deitado.angulo_tronco) > 85
    assert np.nanmin(deitado.razao_caixa) > 1.5


def test_interpolacao_preenche_so_lacunas_curtas():
    serie = np.array([1, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 8.0])
    preenchida = cin.interpolar(serie, lacuna_maxima=2)
    assert preenchida[1] == 2 and np.isnan(preenchida[4])


def test_queda_rapida_e_detectada():
    eventos = _detectar(_sequencia(_transicao(2, 0.5, 2)))
    [queda] = [e for e in eventos if e.tipo == "queda"]
    assert queda.evidencias["descida_quadril"] > 0.25
    assert queda.evidencias["velocidade_pico_alturas_s"] > 0.9
    assert 2.0 <= queda.fim_s <= 2.6


def test_deitar_devagar_nao_e_queda():
    eventos = _detectar(_sequencia(_transicao(1, 1.9, 2)))
    assert [e.tipo for e in eventos] == ["deitou_controlado"]


def test_deitar_sobre_a_cama_vira_contexto():
    cama = {"classe": "bed", "confianca": 0.9, "caixa": [100, 150, 400, 260]}
    objetos = [{"quadro": q, "objetos": [cama]} for q in range(0, 200, 5)]
    eventos = _detectar(_sequencia(_transicao(2, 0.5, 2), objetos))
    assert [e.tipo for e in eventos] == ["deitou_em_movel"]
    assert eventos[0].evidencias["movel"] == "bed"


def test_permanencia_prolongada_no_chao():
    eventos = _detectar(_sequencia(_transicao(1, 0.5, 11)))
    assert [e.tipo for e in eventos] == ["queda", "permanencia_no_chao"]


def _cinematica_exercicio(amplitudes: list[float], contralateral: list[float] | None = None,
                          tronco_extra: list[float] | None = None) -> cin.Cinematica:
    periodo = int(2 * FPS)
    n = periodo * len(amplitudes)
    t = np.arange(n) / FPS
    fase = (1 - np.cos(2 * np.pi * (np.arange(n) % periodo) / periodo)) / 2
    principal = 175 - np.repeat(amplitudes, periodo) * fase
    outra = 175 - np.repeat(contralateral or amplitudes, periodo) * fase
    tronco = 10 + np.repeat(tronco_extra or [0.0] * len(amplitudes), periodo) * fase
    articulacoes = {nome: np.full(n, 170.0) for nome in cin.ARTICULACOES}
    articulacoes["joelho_dir"] = principal
    articulacoes["joelho_esq"] = outra
    return cin.Cinematica(tempos=t, fps=FPS, angulo_tronco=tronco, razao_caixa=np.full(n, 0.4),
                          altura_caixa=np.full(n, 200.0), quadril_y=np.full(n, 130.0),
                          pe=np.zeros((n, 2)), articulacoes=articulacoes,
                          presente=np.ones(n, dtype=bool))


def test_serie_regular_sem_desvios():
    analise = exercicio.analisar(_cinematica_exercicio([80.0] * 8))
    assert analise.articulacao in {"joelho_esq", "joelho_dir"} and analise.modo == "bilateral"
    assert analise.resumo["repeticoes"] == 8
    assert analise.resumo["amplitude_media_graus"] == pytest.approx(80, abs=2)
    assert analise.resumo["cadencia_por_minuto"] == pytest.approx(30, abs=2)
    assert not analise.desvios


def test_queda_de_amplitude_ao_longo_da_serie():
    analise = exercicio.analisar(_cinematica_exercicio(list(np.linspace(85, 45, 9))))
    assert [d.tipo for d in analise.desvios] == ["queda_de_amplitude"]


def test_assimetria_e_compensacao_de_tronco():
    analise = exercicio.analisar(_cinematica_exercicio(
        [80.0] * 8, contralateral=[45.0] * 8, tronco_extra=[0, 0, 25, 25, 25, 25, 25, 25]))
    tipos = {d.tipo: d.severidade for d in analise.desvios}
    assert tipos == {"assimetria": "alta", "compensacao_tronco": "moderada"}


def test_sem_movimento_nao_inventa_repeticoes():
    estatico = _cinematica_exercicio([0.0] * 4)
    assert exercicio.analisar(estatico).articulacao is None


def test_areas_criticas_e_objetos_de_risco(tmp_path):
    configuracao = tmp_path / "areas.json"
    configuracao.write_text(json.dumps({"zonas": [{"nome": "porta", "poligono": [
        [0.25, 0.6], [0.45, 0.6], [0.45, 0.8], [0.25, 0.8]], "permanencia_s": 1.0}]}))
    faca = {"classe": "knife", "confianca": 0.8, "caixa": [0, 0, 10, 10]}
    sequencia = _sequencia([0.0] * 60, [{"quadro": q, "objetos": [faca]} for q in (0, 5, 10)])
    eventos = areas.detectar(sequencia, cin.calcular(sequencia),
                             areas.ConfigAreas.carregar(configuracao))
    assert {e.tipo for e in eventos} == {"area_critica", "objeto_risco"}
    assert areas.dentro((0.5, 0.5), ((0, 0), (1, 0), (1, 1), (0, 1)))
    assert not areas.dentro((1.5, 0.5), ((0, 0), (1, 0), (1, 1), (0, 1)))
    assert areas.objetos_vistos(sequencia) == {"knife": 3}


def test_pipeline_gera_relatorio_e_alerta(tmp_path, prontuarios):
    from assistente_medico.auditoria import Auditoria
    from assistente_medico.multimodal import CentralAlertas

    banco = BancoHospital(tmp_path / "h.db")
    banco.popular(prontuarios)
    central = CentralAlertas(banco, notificadores=[],
                             auditoria=Auditoria(banco=banco, caminho_jsonl=tmp_path / "a.jsonl"))
    resultado = analisar_video(tmp_path / "queda.mp4", prontuario="875431", banco=banco,
                               central=central, pasta_saida=tmp_path / "saida",
                               sequencia=_sequencia(_transicao(2, 0.5, 2)))
    assert resultado.tipo == "quarto"
    assert [a.tipo for a in resultado.anomalias] == ["video_queda"]
    [alerta] = banco.listar_alertas(prontuario="875431")
    assert alerta["prioridade"] == "maxima" and alerta["categoria"] == "anomalia_video"
    assert (tmp_path / "saida" / "queda_grafico.png").exists()
    relatorio = (tmp_path / "saida" / "queda.md").read_text()
    assert "Queda detectada" in relatorio and "YOLOv8s-pose" in relatorio


def test_cache_da_sequencia_ida_e_volta(tmp_path):
    original = _sequencia([0.0, 30.0, 60.0], [{"quadro": 0, "objetos": []}])
    copia = SequenciaPose.carregar(original.salvar(tmp_path / "s.npz"))
    assert copia.quadros == 3 and copia.fps == FPS
    np.testing.assert_allclose(copia.pontos, original.pontos)


def test_recorte_padrao(tmp_path):
    assert recorte_padrao(tmp_path / "fall-01-cam0.mp4", 640, 240) == "metade_direita"
    assert recorte_padrao(tmp_path / "exercicio.webm", 640, 240) is None


def test_rastreamento_ignora_objeto_parado_confundido_com_pessoa():
    quadros = 40
    caixas = np.full((quadros, 3, 4), np.nan)
    confiancas = np.zeros((quadros, 3))
    for i in range(quadros):
        caixas[i, 0] = [0, 100, 60, 240]
        confiancas[i, 0] = 0.95
        if i >= 5:
            x = 150 + 3 * i
            caixas[i, 1] = [x, 90, x + 40, 200]
            confiancas[i, 1] = 0.8
    escolha = rastrear(caixas, confiancas)
    assert (escolha[:5] == -1).all() and (escolha[5:] == 1).all()

    sequencia = _sequencia([0.0] * quadros)
    sequencia.todas_caixas, sequencia.todas_confiancas = caixas, confiancas
    sequencia.todos_pontos = np.repeat(sequencia.pontos[:, None], 3, axis=1)
    aplicar_rastreamento(sequencia)
    assert np.isnan(sequencia.caixas[0, 0]) and sequencia.caixas[10, 0] == 180


def test_avaliacao_com_gabarito(tmp_path):
    (tmp_path / "urfall-cam0-falls.csv").write_text(
        "\n".join(f"fall-01,{q},{-1 if q < 61 else 1}" for q in range(1, 120)))
    (tmp_path / "urfall-cam0-adls.csv").write_text(
        "\n".join(f"adl-02,{q},-1" for q in range(1, 120)))
    rotulos = avaliacao.gabarito(tmp_path)
    assert rotulos["fall-01"]["quadro_inicio"] == 61 and not rotulos["adl-02"]["queda"]
    sequencias = {"fall-01": _sequencia(_transicao(2, 0.5, 2)),
                  "adl-02": _sequencia(_transicao(1, 1.9, 2))}
    resultado = avaliacao.avaliar(sequencias=sequencias, rotulos=rotulos)
    assert resultado["particoes"]["calibracao (impares)"]["sensibilidade"] == 1.0
    assert resultado["particoes"]["teste (pares)"]["especificidade"] == 1.0


def test_lados_so_sao_comparados_em_repeticoes_confiaveis():
    cinematica = _cinematica_exercicio([80.0] * 8, contralateral=[40.0] * 8)
    n = len(cinematica.tempos)
    cinematica.razao_ombros_quadro = np.full(n, 0.15)
    analise = exercicio.analisar(cinematica)
    assert analise.modo == "lados_nao_comparaveis"
    assert not any(d.tipo == "assimetria" for d in analise.desvios)

    cinematica.razao_ombros_quadro = np.full(n, 0.6)
    cinematica.confianca_articulacoes = {"joelho_dir": np.full(n, 0.9), "joelho_esq": np.full(n, 0.2)}
    assert exercicio.analisar(cinematica).modo == "lados_nao_comparaveis"
    cinematica.confianca_articulacoes["joelho_esq"] = np.full(n, 0.9)
    assert any(d.tipo == "assimetria" for d in exercicio.analisar(cinematica).desvios)


def test_movimento_unico_reporta_amplitude_maxima():
    n = int(10 * FPS)
    cinematica = _cinematica_exercicio([0.0] * 5)
    cinematica.articulacoes["cotovelo_esq"] = np.linspace(170, 90, len(cinematica.tempos))
    analise = exercicio.analisar(cinematica)
    assert analise.modo == "sem_repeticoes" and analise.resumo["amplitude_maxima_graus"] > 60
    assert n > 0
