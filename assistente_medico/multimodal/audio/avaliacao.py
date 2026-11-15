"""Avaliacao da analise de audio.

1. VOICED (vozes reais): jitter, shimmer e HNR separam vozes saudaveis de
   patologicas? AUC de cada medida e de uma regressao logistica com validacao
   cruzada.
2. Condicoes vocais simuladas (TTS): calibracao em tres vozes, teste em outras
   tres, mesmo texto em todas as condicoes.
3. Consultas dos pacientes: voz, termos criticos e negacoes esperados.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ...config import CAMINHOS
from . import indicadores as ind
from .acustica import medir
from .roteiros import CONDICOES, PASTA_AUDIO, VOZES_AVALIACAO

VOZES_CALIBRACAO = 3

ESPERADO_CONSULTAS = {
    "117256": {"voz": [], "termos": [], "negados": []},
    "208731": {"voz": ["fadiga"], "termos": ["fadiga"], "negados": []},
    "559120": {"voz": ["dispneia"], "termos": ["dispneia"], "negados": []},
    "553119": {"voz": ["disartria"], "termos": ["deficit_neurologico"], "negados": []},
    "771203": {"voz": [], "termos": ["dor_toracica"], "negados": []},
    "875431": {"voz": [], "termos": [], "negados": ["dor_toracica", "dispneia"]},
}


def ler_voiced(pasta: Path | None = None) -> list[dict[str, Any]]:
    """Le os registros WFDB do VOICED (formato 32: inteiro de 32 bits little-endian)."""
    pasta = Path(pasta or CAMINHOS.multimodal / "voiced")
    registros = []
    for cabecalho in sorted(pasta.glob("voice*.hea")):
        linhas = cabecalho.read_text().splitlines()
        _, _, taxa, _ = linhas[0].split()[:4]
        campos = linhas[1].split()
        ganho_base = campos[2]
        ganho = float(ganho_base.split("(")[0])
        base = float(ganho_base.split("(")[1].split(")")[0]) if "(" in ganho_base else 0.0
        bruto = np.fromfile(pasta / campos[0], dtype="<i4").astype(np.float64)
        info = (pasta / f"{cabecalho.stem}-info.txt")
        diagnostico = "desconhecido"
        if info.exists():
            for linha in info.read_text(errors="replace").splitlines():
                if linha.lower().startswith("diagnosis"):
                    diagnostico = linha.split(":", 1)[1].strip()
        registros.append({
            "id": cabecalho.stem, "taxa": int(taxa), "sinal": (bruto - base) / ganho,
            "diagnostico": diagnostico, "saudavel": diagnostico.lower() == "healthy",
        })
    return registros


def avaliar_voiced(pasta: Path | None = None,
                   progresso: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    registros = ler_voiced(pasta)
    linhas = []
    for indice, registro in enumerate(registros, start=1):
        try:
            medidas = medir(registro["sinal"] / (np.max(np.abs(registro["sinal"])) or 1.0),
                            registro["taxa"])
        except ValueError:
            continue
        if None in (medidas.jitter_local, medidas.shimmer_local, medidas.hnr_db):
            continue
        linhas.append((registro["saudavel"], medidas.jitter_local, medidas.shimmer_local,
                       medidas.hnr_db))
        if progresso:
            progresso(indice, len(registros))

    rotulo = np.array([0 if saudavel else 1 for saudavel, *_ in linhas])
    matriz = np.array([valores for _, *valores in linhas])
    nomes = ("jitter_local", "shimmer_local", "hnr_db")
    sentidos = (1, 1, -1)
    medidas = {}
    for coluna, (nome, sentido) in enumerate(zip(nomes, sentidos, strict=True)):
        valores = matriz[:, coluna]
        medidas[nome] = {
            "auc": round(float(roc_auc_score(rotulo, sentido * valores)), 3),
            "mediana_saudavel": round(float(np.median(valores[rotulo == 0])), 4),
            "mediana_patologica": round(float(np.median(valores[rotulo == 1])), 4),
        }
    modelo = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    probabilidades = cross_val_predict(
        modelo, matriz, rotulo, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba")[:, 1]
    return {
        "dataset": "PhysioNet VOICED (vogal /a/ sustentada, 8 kHz)",
        "registros_analisados": len(linhas), "saudaveis": int((rotulo == 0).sum()),
        "patologicos": int((rotulo == 1).sum()), "medidas": medidas,
        "regressao_logistica_auc_cv5": round(float(roc_auc_score(rotulo, probabilidades)), 3),
    }


def avaliar_condicoes(pasta: Path | None = None) -> dict[str, Any]:
    pasta = Path(pasta or PASTA_AUDIO) / "avaliacao"
    resultado: dict[str, Any] = {"particoes": {}}
    for nome_particao, vozes in (("calibracao", VOZES_AVALIACAO[:VOZES_CALIBRACAO]),
                                 ("teste", VOZES_AVALIACAO[VOZES_CALIBRACAO:])):
        casos = []
        for voz in vozes:
            nome = voz.split("-")[2].replace("Neural", "").lower()
            basal = pasta / f"{nome}_basal.wav"
            if not basal.exists():
                continue
            medidas_basais = medir(basal)
            for condicao in CONDICOES:
                detectadas = [i.condicao for i in ind.avaliar(medir(pasta / f"{nome}_{condicao}.wav"),
                                                              medidas_basais) if i.suspeita]
                esperado = [] if condicao == "normal" else [condicao]
                casos.append({"voz": nome, "condicao": condicao, "detectadas": detectadas,
                              "acerto": detectadas == esperado})
        resultado["particoes"][nome_particao] = {
            "casos": casos, "acertos": sum(c["acerto"] for c in casos), "total": len(casos)}
    return resultado


def avaliar_consultas(banco, usar_azure: bool = True,
                      pasta: Path | None = None) -> dict[str, Any]:
    from .consulta import analisar_consulta

    pasta = Path(pasta or PASTA_AUDIO) / "consultas"
    casos = []
    for prontuario, esperado in ESPERADO_CONSULTAS.items():
        arquivo = pasta / f"{prontuario}_consulta.wav"
        if not arquivo.exists():
            continue
        r = analisar_consulta(arquivo, banco=banco, usar_azure=usar_azure)
        voz = sorted(i.condicao for i in r.indicadores if i.suspeita)
        presentes = sorted({m.termo.chave for m in r.mencoes if not m.negada})
        negados = sorted({m.termo.chave for m in r.mencoes if m.negada})
        casos.append({
            "prontuario": prontuario, "fontes": r.fontes,
            "voz": voz, "voz_ok": voz == sorted(esperado["voz"]),
            "termos": presentes, "termos_ok": set(esperado["termos"]) <= set(presentes),
            "negados": negados, "negados_ok": set(esperado["negados"]) <= set(negados)
            and not set(esperado["negados"]) & set(presentes),
            "confianca_transcricao": (r.transcricao or {}).get("confianca"),
            "anomalias": [f"{a.tipo} ({a.severidade})" for a in r.anomalias],
            "texto_anonimizado": r.texto_anonimizado,
        })
    return {"casos": casos,
            "acertos": sum(c["voz_ok"] and c["termos_ok"] and c["negados_ok"] for c in casos),
            "total": len(casos)}


def markdown(resultado: dict[str, Any]) -> str:
    partes = ["# Avaliacao da analise de audio\n", f"Gerado em {resultado['gerado_em']}.\n"]
    if "voiced" in resultado:
        v = resultado["voiced"]
        partes += [
            "## Qualidade vocal em vozes reais (PhysioNet VOICED)\n",
            f"{v['registros_analisados']} gravacoes ({v['saudaveis']} saudaveis, "
            f"{v['patologicos']} com patologia vocal).\n",
            "| Medida | AUC | Mediana saudavel | Mediana patologica |", "|---|---|---|---|",
        ]
        for nome, d in v["medidas"].items():
            partes.append(f"| {nome} | {d['auc']:.2f} | {d['mediana_saudavel']} | {d['mediana_patologica']} |")
        partes += ["", f"Regressao logistica com as tres medidas, validacao cruzada de 5 dobras: "
                       f"AUC {v['regressao_logistica_auc_cv5']:.2f}.", ""]
    if "condicoes" in resultado:
        partes += ["## Condicoes vocais simuladas (Azure TTS)\n",
                   "| Particao | Acertos | Erros |", "|---|---|---|"]
        for nome, d in resultado["condicoes"]["particoes"].items():
            erros = [f"{c['voz']}/{c['condicao']} -> {c['detectadas']}" for c in d["casos"] if not c["acerto"]]
            partes.append(f"| {nome} | {d['acertos']}/{d['total']} | {'; '.join(erros) or '-'} |")
        partes.append("")
    if "consultas" in resultado:
        partes += ["## Consultas dos pacientes (pipeline completo)\n",
                   "| Prontuario | Voz | Termos presentes | Negados | Conf. STT | OK |",
                   "|---|---|---|---|---|---|"]
        for c in resultado["consultas"]["casos"]:
            ok = c["voz_ok"] and c["termos_ok"] and c["negados_ok"]
            partes.append(f"| {c['prontuario']} | {', '.join(c['voz']) or '-'} | "
                          f"{', '.join(c['termos']) or '-'} | {', '.join(c['negados']) or '-'} | "
                          f"{c['confianca_transcricao'] or '-'} | {'sim' if ok else 'nao'} |")
        partes.append("")
    return "\n".join(partes)


def salvar(resultado: dict[str, Any]) -> Path:
    CAMINHOS.avaliacao.mkdir(parents=True, exist_ok=True)
    destino = CAMINHOS.avaliacao / "audio.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2, default=str) + "\n",
                       encoding="utf-8")
    (CAMINHOS.avaliacao / "audio.md").write_text(markdown(resultado), encoding="utf-8")
    return destino
