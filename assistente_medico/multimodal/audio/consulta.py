"""Pipeline de analise do audio de uma consulta.

    audio (WAV)
      |-- medidas acusticas locais (Praat) ---- comparadas a linha de base do paciente
      |                                         -> indicadores: fadiga, dispneia, disartria
      '-- Azure Speech to Text --------------- transcricao com tempo por palavra
             '-- anonimizador (local) -------- nomes e documentos viram pseudonimos
                    |-- Azure Language: sentimento e mineracao de opiniao
                    '-- Azure Text Analytics for Health: sintomas, com negacao
                           '-- catalogo de termos criticos (+ detector local de reserva)
      => anomalias (audio e texto) -> central de alertas

Sem Azure (ou com falha dela), a analise continua: a voz e medida localmente e
os termos criticos sao buscados pelo detector local no texto disponivel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ...config import AZURE, ConfigAzure
from ...dados.anonimizacao import Anonimizador
from ..azure.cliente import ErroAzure
from ..tipos import SEVERIDADES, Anomalia
from . import indicadores as ind
from . import termos
from .acustica import MedidasVoz, medir

logger = logging.getLogger("assistente_medico.multimodal.audio")


@dataclass
class ResultadoConsulta:
    prontuario: str | None
    arquivo: str
    medidas: MedidasVoz
    medidas_basais: MedidasVoz | None
    indicadores: list[ind.Indicador]
    transcricao: dict[str, Any] | None
    texto_anonimizado: str | None
    sentimento: dict[str, Any] | None
    entidades: list[dict[str, Any]]
    mencoes: list[termos.Mencao]
    anomalias: list[Anomalia]
    fontes: dict[str, str]
    avisos: list[str] = field(default_factory=list)
    analise_id: int | None = None

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "prontuario": self.prontuario, "arquivo": self.arquivo, "analise_id": self.analise_id,
            "fontes": self.fontes, "avisos": self.avisos,
            "medidas": self.medidas.como_dicionario(),
            "medidas_basais": self.medidas_basais.como_dicionario() if self.medidas_basais else None,
            "indicadores": [i.como_dicionario() for i in self.indicadores],
            "transcricao": self.transcricao, "texto_anonimizado": self.texto_anonimizado,
            "sentimento": self.sentimento, "entidades": self.entidades,
            "mencoes": [m.como_dicionario() for m in self.mencoes],
            "anomalias": [a.como_dicionario() for a in self.anomalias],
        }

    def relatorio(self) -> str:
        m, b = self.medidas, self.medidas_basais
        linhas = [f"# Analise de audio da consulta - prontuario {self.prontuario or '-'}", "",
                  f"Arquivo: `{Path(self.arquivo).name}`  ",
                  f"Transcricao: {self.fontes.get('transcricao')} | linguagem: "
                  f"{self.fontes.get('linguagem')} | comparacao: {self.fontes.get('comparacao')}", ""]
        if self.avisos:
            linhas += [f"> {aviso}" for aviso in self.avisos] + [""]
        linhas += ["## Voz", "", "| Medida | Consulta | Linha de base |", "|---|---|---|"]
        for chave, rotulo in (("taxa_articulacao", "Articulacao (silabas/s)"),
                              ("silabas_por_grupo", "Silabas por grupo respiratorio"),
                              ("pausas_por_minuto", "Pausas por minuto"),
                              ("pausa_media_s", "Pausa media (s)"),
                              ("intensidade_media_db", "Intensidade (dB)"),
                              ("f0_variacao_st", "Variacao de tom (semitons)"),
                              ("jitter_local", "Jitter local"), ("shimmer_local", "Shimmer local"),
                              ("hnr_db", "HNR (dB)")):
            atual = getattr(m, chave)
            basal = getattr(b, chave) if b else None
            linhas.append(f"| {rotulo} | {_num(atual)} | {_num(basal)} |")
        linhas += ["", "| Indicador | Suspeita | Criterios atendidos |", "|---|---|---|"]
        for indicador in self.indicadores:
            linhas.append(f"| {indicador.condicao} | {'**sim**' if indicador.suspeita else 'nao'} "
                          f"| {'; '.join(indicador.criterios) or '-'} |")
        if self.texto_anonimizado:
            linhas += ["", "## Fala transcrita (pseudonimizada)", "", f"> {self.texto_anonimizado}"]
        if self.sentimento:
            confianca = self.sentimento["confianca"]
            linhas += ["", f"Sentimento geral: **{self.sentimento['sentimento']}** "
                           f"(positivo {confianca['positive']:.2f}, neutro {confianca['neutral']:.2f}, "
                           f"negativo {confianca['negative']:.2f})"]
        if self.mencoes:
            linhas += ["", "## Termos clinicos", "", "| Termo | Trecho | Situacao | Fonte |",
                       "|---|---|---|---|"]
            for mencao in self.mencoes:
                situacao = "negado" if mencao.negada else ("condicional" if mencao.condicional
                                                           else "**presente**")
                linhas.append(f"| {mencao.termo.rotulo} | {mencao.trecho} | {situacao} | {mencao.fonte} |")
        linhas += ["", "## Anomalias", ""]
        if not self.anomalias:
            linhas.append("Nenhuma.")
        for anomalia in self.anomalias:
            linhas.append(f"- [{anomalia.severidade.upper()}] {anomalia.descricao}")
        return "\n".join(linhas) + "\n"


def _num(valor: Any) -> str:
    if valor is None:
        return "-"
    return f"{valor:.3f}" if isinstance(valor, float) and abs(valor) < 1 else f"{valor:.1f}"


def _rebaixar(severidade: str) -> str:
    return SEVERIDADES[min(SEVERIDADES.index(severidade) + 1, len(SEVERIDADES) - 1)]


def _prontuario_do_nome(caminho: Path) -> str | None:
    prefixo = caminho.stem.split("_")[0]
    return prefixo if prefixo.isdigit() else None


def analisar_consulta(caminho: str | Path, prontuario: str | None = None,
                      basal: str | Path | None = None, banco=None, central=None,
                      usar_azure: bool = True, config: ConfigAzure | None = None,
                      anonimizador: Anonimizador | None = None,
                      instante: datetime | None = None) -> ResultadoConsulta:
    caminho = Path(caminho)
    config = config or AZURE
    prontuario = prontuario or _prontuario_do_nome(caminho)
    avisos: list[str] = []
    fontes: dict[str, str] = {}

    if basal is None and prontuario:
        candidato = caminho.with_name(f"{prontuario}_basal.wav")
        basal = candidato if candidato.exists() and candidato != caminho else None
    medidas = medir(caminho)
    medidas_basais = medir(basal) if basal else None
    fontes["comparacao"] = "linha de base do paciente" if medidas_basais else "faixas de referencia"
    indicadores = ind.avaliar(medidas, medidas_basais)

    transcricao = None
    texto = None
    if usar_azure and config.fala_configurada:
        try:
            from ..azure.fala import transcrever

            resultado = transcrever(caminho, medidas.intervalos_pausa, config)
            transcricao, texto = resultado.como_dicionario(), resultado.texto
            fontes["transcricao"] = "azure_speech"
        except ErroAzure as erro:
            avisos.append(f"Transcricao Azure indisponivel: {erro}")
    if texto is None and caminho.with_suffix(".txt").exists():
        texto = caminho.with_suffix(".txt").read_text(encoding="utf-8").strip()
        transcricao = {"texto": texto, "fonte": "arquivo_texto"}
        fontes["transcricao"] = "arquivo de texto (sem Azure)"
    fontes.setdefault("transcricao", "indisponivel")

    if anonimizador is None:
        nomes = [p["nome"] for p in banco.listar_pacientes()] if banco is not None else []
        anonimizador = Anonimizador(nomes_conhecidos=nomes)
    texto_anonimizado = anonimizador.anonimizar_texto(texto) if texto else None

    sentimento, entidades, mencoes = None, [], []
    if texto_anonimizado:
        if usar_azure and config.linguagem_configurada:
            try:
                from ..azure.linguagem import analisar_saude, analisar_sentimento

                sentimento = analisar_sentimento(texto_anonimizado, config)
                encontradas = analisar_saude(texto_anonimizado, config)
                entidades = [e.como_dicionario() for e in encontradas]
                mencoes = termos.classificar_entidades(encontradas)
                fontes["linguagem"] = "azure_language"
            except ErroAzure as erro:
                avisos.append(f"Azure Language indisponivel: {erro}")
        cobertos = {m.termo.chave for m in mencoes}
        reserva = [m for m in termos.detectar_local(texto_anonimizado)
                   if m.termo.chave not in cobertos]
        mencoes = termos.consolidar(mencoes + reserva)
        fontes.setdefault("linguagem", "detector local")

    if instante is None:
        from ..relogio import agora as relogio

        instante = relogio(banco)
    agora = instante.isoformat(timespec="seconds")
    referencia = "a linha de base do paciente" if medidas_basais else "faixas de referencia"
    anomalias = []
    for indicador in indicadores:
        if not indicador.suspeita:
            continue
        rotulo, severidade = ind.CONDICOES[indicador.condicao]
        anomalias.append(Anomalia(
            modalidade="audio", tipo=f"voz_{indicador.condicao}", severidade=severidade,
            descricao=f"{rotulo} (comparada com {referencia}): {'; '.join(indicador.criterios)}",
            prontuario=prontuario, inicio=agora, fim=agora, escore=indicador.escore,
            detector="acustica_praat", evidencias={
                **indicador.como_dicionario(), "arquivo": caminho.name,
                "confianca_transcricao": (transcricao or {}).get("confianca")},
        ))
    for mencao in mencoes:
        if mencao.negada:
            continue
        severidade = _rebaixar(mencao.termo.severidade) if mencao.condicional else mencao.termo.severidade
        anomalias.append(Anomalia(
            modalidade="texto", tipo=f"termo_critico_{mencao.termo.chave}", severidade=severidade,
            descricao=f"Paciente relata {mencao.termo.rotulo} na consulta: \"{mencao.trecho}\"",
            prontuario=prontuario, inicio=agora, fim=agora, escore=1.0,
            detector="azure_text_analytics_health" if mencao.fonte == "azure_health" else "termos_locais",
            evidencias=mencao.como_dicionario(),
        ))
    if sentimento and sentimento["sentimento"] == "negative" and sentimento["confianca"]["negative"] >= 0.9:
        anomalias.append(Anomalia(
            modalidade="texto", tipo="sofrimento_referido", severidade="baixa",
            descricao="Relato com sentimento fortemente negativo (Azure Language)",
            prontuario=prontuario, inicio=agora, fim=agora,
            escore=sentimento["confianca"]["negative"], detector="azure_sentimento",
            evidencias={"opinioes": sentimento.get("opinioes", [])},
        ))

    resultado = ResultadoConsulta(
        prontuario=prontuario, arquivo=str(caminho), medidas=medidas,
        medidas_basais=medidas_basais, indicadores=indicadores, transcricao=transcricao,
        texto_anonimizado=texto_anonimizado, sentimento=sentimento, entidades=entidades,
        mencoes=mencoes, anomalias=anomalias, fontes=fontes, avisos=avisos,
    )
    if banco is not None:
        resultado.analise_id = banco.abrir_analise("audio", prontuario=prontuario,
                                                   origem=caminho.name)
        if central is not None:
            central.processar(anomalias, analise_id=resultado.analise_id)
        resumo = resultado.como_dicionario()
        if resumo["transcricao"]:
            resumo["transcricao"] = {k: v for k, v in resumo["transcricao"].items() if k != "texto"}
        banco.concluir_analise(resultado.analise_id, resumo)
    return resultado


def salvar_relatorio(resultado: ResultadoConsulta, pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    base = pasta / Path(resultado.arquivo).stem
    dados = resultado.como_dicionario()
    if dados["transcricao"]:
        dados["transcricao"] = {k: v for k, v in dados["transcricao"].items() if k != "texto"}
    base.with_suffix(".json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    base.with_suffix(".md").write_text(resultado.relatorio(), encoding="utf-8")
    return base.with_suffix(".md")
