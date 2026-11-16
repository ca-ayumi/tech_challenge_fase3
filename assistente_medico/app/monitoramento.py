"""Aba Monitoramento do Streamlit: risco multimodal, alertas consolidados e evidencias."""

from __future__ import annotations

import json
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from assistente_medico.app.envio import secao_envio
from assistente_medico.config import AZURE, MONITORAMENTO
from assistente_medico.multimodal import demonstracao, fusao
from assistente_medico.multimodal.alertas import CentralAlertas
from assistente_medico.multimodal.relogio import agora

NIVEL = {"critico": "🔴 crítico", "alto": "🟠 alto", "moderado": "🟡 moderado", "baixo": "🟢 baixo"}
PRIORIDADE = {"maxima": "🔴", "alta": "🟠", "media": "🟡", "informativa": "🔵"}
CORES_SEVERIDADE = {"critica": "#c53030", "alta": "#dd6b20", "moderada": "#b7791f", "baixa": "#3182ce"}
MODALIDADES = {"sinais_vitais": "Sinais vitais", "prescricao": "Prescrição",
               "movimentacao": "Movimentação", "audio": "Voz", "texto": "Fala (conteúdo)",
               "video": "Vídeo", "multimodal": "Fusão"}
VITAIS = {"fc": "FC (bpm)", "fr": "FR (irpm)", "saturacao": "SpO₂ (%)", "pas": "PAS (mmHg)",
          "temperatura": "Temperatura (°C)"}


def _executar_demonstracao(banco) -> None:
    coluna_azure, coluna_video, coluna_botao = st.columns([2, 2, 2])
    usar_azure = coluna_azure.checkbox(
        "Usar Azure (Speech + Language)",
        value=AZURE.fala_configurada and AZURE.linguagem_configurada,
        disabled=not (AZURE.fala_configurada and AZURE.linguagem_configurada),
        help="Sem as chaves no .env, o áudio é analisado só localmente.")
    usar_video = coluna_video.checkbox("Analisar vídeo (YOLOv8)", value=True)
    if coluna_botao.button("▶ Executar demonstração completa", type="primary"):
        with st.status("Executando o fluxo multimodal...", expanded=True) as status:
            demonstracao.executar(banco, usar_azure=usar_azure, usar_video=usar_video,
                                  progresso=st.write)
            status.update(label="Demonstração concluída", state="complete", expanded=False)
        st.rerun()
    if st.button("Recalcular fusão", help="Reavalia o risco e emite alertas consolidados novos."):
        fusao.consolidar(banco, CentralAlertas(banco))
        st.rerun()


def _alertas_consolidados(banco, pacientes: dict) -> None:
    st.markdown("#### Alertas consolidados para a equipe")
    abertos = [a for a in banco.listar_alertas(limite=200) if a["categoria"] == "anomalia_multimodal"]
    if not abertos:
        st.info("Nenhum alerta consolidado aberto.")
    for alerta in abertos:
        paciente = pacientes.get(alerta["prontuario"], {})
        with st.container(border=True):
            texto, botao = st.columns([7, 1])
            titulo, _, evidencias = alerta["mensagem"].partition(". Evidencias: ")
            texto.markdown(f"{PRIORIDADE.get(alerta['prioridade'], '')} **{paciente.get('leito', '-')} · "
                           f"{paciente.get('nome', alerta['prontuario'])}** — {titulo}")
            if evidencias:
                texto.caption(" · ".join(e.strip() for e in evidencias.split(" | ")))
            texto.caption(f"Alerta #{alerta['id']} · {alerta['fonte']} · emitido {alerta['criado_em']}")
            if botao.button("Reconhecer", key=f"reconhecer_{alerta['id']}"):
                CentralAlertas(banco, notificadores=[]).reconhecer(
                    alerta["id"], profissional=st.session_state.get("usuario") or "nao identificado",
                    perfil=st.session_state.get("perfil"))
                st.rerun()

    reconhecidos = [a for a in banco.listar_alertas(status="reconhecido", limite=20)
                    if a["categoria"] == "anomalia_multimodal"]
    if reconhecidos:
        with st.expander(f"Reconhecidos ({len(reconhecidos)}): ciclo detecção → alerta → resposta"):
            linhas = []
            for alerta in reconhecidos:
                trilha = banco.trilha_auditoria(f"ALERTA-{alerta['id']}")
                resposta = trilha[-1] if trilha else {}
                linhas.append({"Alerta": alerta["id"],
                               "Leito": pacientes.get(alerta["prontuario"], {}).get("leito"),
                               "Emitido": alerta["criado_em"],
                               "Reconhecido por": resposta.get("usuario"),
                               "Perfil": resposta.get("perfil"),
                               "Reconhecido em": resposta.get("registrado_em"),
                               "Mensagem": alerta["mensagem"][:90]})
            st.dataframe(linhas, width="stretch", hide_index=True)


def _linha_do_tempo(anomalias: list[dict]) -> None:
    if not anomalias:
        st.info("Sem anomalias para este paciente nas últimas 24 h.")
        return
    tabela = pd.DataFrame([{
        "instante": fusao.instante(a), "modalidade": MODALIDADES.get(a["modalidade"], a["modalidade"]),
        "severidade": a["severidade"], "tipo": a["tipo"], "descricao": a["descricao"],
        "detector": a.get("detector"),
    } for a in anomalias if fusao.instante(a)])
    grafico = alt.Chart(tabela).mark_circle(size=160, opacity=0.85).encode(
        x=alt.X("instante:T", title=None),
        y=alt.Y("modalidade:N", title=None, sort=list(MODALIDADES.values())),
        color=alt.Color("severidade:N", scale=alt.Scale(domain=list(CORES_SEVERIDADE),
                                                         range=list(CORES_SEVERIDADE.values())),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["instante:T", "modalidade", "severidade", "tipo", "descricao", "detector"],
    ).properties(height=260)
    st.altair_chart(grafico, width="stretch")
    st.dataframe(tabela.sort_values("instante", ascending=False)[
        ["instante", "modalidade", "severidade", "descricao"]],
        width="stretch", hide_index=True)


def _sinais_vitais(banco, prontuario: str, anomalias: list[dict]) -> None:
    serie = banco.serie_sinais_vitais(prontuario)
    if len(serie) < 2:
        st.info("Sem série de sinais vitais para este paciente.")
        return
    tabela = pd.DataFrame(serie)
    tabela["instante"] = pd.to_datetime(tabela["aferido_em"])
    longa = tabela.melt(id_vars="instante", value_vars=[c for c in VITAIS if c in tabela],
                        var_name="variavel", value_name="valor").dropna()
    longa["variavel"] = longa["variavel"].map(VITAIS)
    marcas = pd.DataFrame([{"instante": fusao.instante(a), "descricao": a["descricao"]}
                           for a in anomalias if a["modalidade"] == "sinais_vitais" and fusao.instante(a)])
    graficos = []
    for variavel in longa["variavel"].unique():
        linha = alt.Chart(longa[longa["variavel"] == variavel]).mark_line(color="#2b6cb0").encode(
            x=alt.X("instante:T", title=None),
            y=alt.Y("valor:Q", title=variavel, scale=alt.Scale(zero=False)),
            tooltip=["instante:T", "valor"]).properties(height=95)
        if not marcas.empty:
            linha = linha + alt.Chart(marcas).mark_rule(color="#c53030", opacity=0.35).encode(
                x="instante:T", tooltip=["instante:T", "descricao"])
        graficos.append(linha)
    st.altair_chart(alt.vconcat(*graficos), width="stretch")
    st.caption("Linhas vermelhas: anomalias de sinais vitais detectadas.")


def _audio(prontuario: str) -> None:
    from assistente_medico.multimodal.audio.roteiros import PASTA_AUDIO

    consulta = PASTA_AUDIO / "consultas" / f"{prontuario}_consulta.wav"
    if not consulta.exists():
        st.info("Sem gravação de consulta para este paciente.")
        return
    st.audio(str(consulta))
    relatorio = PASTA_AUDIO / "relatorios" / f"{prontuario}_consulta.md"
    if relatorio.exists():
        st.markdown(relatorio.read_text(encoding="utf-8").split("\n", 1)[1])
    else:
        st.caption("Execute a demonstração (ou `audio consultas`) para gerar a análise.")


def _video(prontuario: str) -> None:
    from pathlib import Path

    from assistente_medico.multimodal.video.analise import PASTA_RELATORIOS

    resultados = []
    for arquivo in sorted(PASTA_RELATORIOS.glob("*.json")):
        try:
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if dados.get("prontuario") == prontuario:
            resultados.append(dados)
    if not resultados:
        st.info("Sem vídeo analisado para este paciente.")
        return
    for dados in resultados:
        st.markdown(f"**{Path(dados['arquivo']).name}** · {dados['tipo']} · {dados['duracao_s']} s")
        for anomalia in dados["anomalias"]:
            st.markdown(f"- **[{anomalia['severidade'].upper()}]** {anomalia['descricao']}")
        gerados = dados.get("arquivos_gerados", {})
        if "video anotado" in gerados and Path(gerados["video anotado"]).exists():
            st.video(gerados["video anotado"])
        imagens = [c for c in gerados.values() if c.endswith((".jpg", ".png")) and Path(c).exists()]
        if imagens:
            st.image(imagens, width=330)


def aba_monitoramento(banco) -> None:
    referencia: datetime = agora(banco)
    st.subheader("Monitoramento multimodal")
    st.caption(f"Relógio: {referencia:%d/%m/%Y %H:%M} "
               f"({'cenário sintético' if MONITORAMENTO.relogio == 'cenario' else 'hora real'}) · "
               "fusão tardia de sinais vitais, prescrições, movimentação, voz, fala e vídeo")
    with st.expander("Executar demonstração", expanded=not banco.listar_anomalias(limite=1)):
        _executar_demonstracao(banco)
    with st.expander("📤 Enviar gravação para análise (vídeo ou áudio)",
                     expanded="envio_resultado" in st.session_state):
        secao_envio(banco)

    pacientes = {p["prontuario"]: p for p in banco.listar_pacientes()}
    avaliacoes = fusao.avaliar_todos(banco, referencia)
    consolidados = [a for a in banco.listar_alertas(limite=500) if a["categoria"] == "anomalia_multimodal"]
    individuais = [a for a in banco.listar_alertas(limite=500)
                   if a["categoria"].startswith("anomalia_") and a["categoria"] != "anomalia_multimodal"]
    metricas = st.columns(4)
    metricas[0].metric("Pacientes em risco crítico", sum(a.nivel == "critico" for a in avaliacoes))
    metricas[1].metric("Pacientes em risco alto", sum(a.nivel == "alto" for a in avaliacoes))
    metricas[2].metric("Alertas consolidados abertos", len(consolidados))
    metricas[3].metric("Alertas por modalidade", len(individuais),
                       help="Alertas individuais dos detectores, agrupados nos consolidados.")

    if avaliacoes:
        st.markdown("#### Risco por paciente")
        st.dataframe([{
            "Leito": pacientes.get(a.prontuario, {}).get("leito", a.prontuario),
            "Paciente": pacientes.get(a.prontuario, {}).get("nome", "-"),
            "Risco": round(a.escore),
            "Nível": NIVEL.get(a.nivel, a.nivel),
            "Modalidades": ", ".join(MODALIDADES.get(m, m) for m in a.modalidades),
            "Correlações": "; ".join(c.regra.titulo for c in a.correlacoes) or "-",
        } for a in avaliacoes], width="stretch", hide_index=True,
            column_config={"Risco": st.column_config.ProgressColumn(
                "Risco", min_value=0, max_value=100, format="%d")})

    _alertas_consolidados(banco, pacientes)

    if not avaliacoes:
        return
    st.markdown("#### Detalhe do paciente")
    opcoes = [a.prontuario for a in avaliacoes]
    escolhido = st.selectbox(
        "Paciente", opcoes,
        format_func=lambda p: f"{pacientes.get(p, {}).get('leito', p)} · "
                              f"{pacientes.get(p, {}).get('nome', '')}")
    avaliacao = next(a for a in avaliacoes if a.prontuario == escolhido)
    st.caption(f"Risco {avaliacao.escore:.0f}/100 · contribuição por modalidade: " + ", ".join(
        f"{MODALIDADES.get(m, m)} {v:.0f}" for m, v in sorted(avaliacao.contribuicoes.items(),
                                                              key=lambda x: -x[1])))
    abas = st.tabs(["Linha do tempo", "Sinais vitais", "Áudio da consulta", "Vídeo"])
    with abas[0]:
        _linha_do_tempo(avaliacao.anomalias)
    with abas[1]:
        _sinais_vitais(banco, escolhido, avaliacao.anomalias)
    with abas[2]:
        _audio(escolhido)
    with abas[3]:
        _video(escolhido)
