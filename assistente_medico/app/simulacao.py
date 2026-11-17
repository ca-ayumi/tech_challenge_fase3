"""Simulacao em tempo real e teste da Azure na aba Monitoramento (sem terminal)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from assistente_medico.config import AZURE
from assistente_medico.multimodal import demonstracao
from assistente_medico.multimodal.alertas import CentralAlertas
from assistente_medico.multimodal.anomalias import (
    DetectorSinaisVitais,
    ParametrosSinaisVitais,
    sintetico,
)
from assistente_medico.multimodal.tempo_real import Monitor

PRIORIDADE = {"maxima": "🔴", "alta": "🟠", "media": "🟡", "informativa": "🔵"}
EVENTOS = {"dessaturacao": "Queda de saturação", "taquicardia": "Taquicardia",
           "taquipneia": "Taquipneia", "piora_combinada": "Piora combinada",
           "artefato": "Sensor solto (artefato)", "": "Nenhum (registro limpo)"}
SERIES = {"saturacao": "SpO₂ (%)", "fc": "FC (bpm)", "fr": "FR (irpm)"}
CENARIOS = {"bidmc": "Paciente real de UTI (PhysioNet), segundo a segundo",
            "internacao": "Internação de 24 h, todos os pacientes"}


def secao_azure() -> None:
    from assistente_medico.multimodal.azure.diagnostico import verificar_azure

    st.caption("Speech (Speech to Text e Text to Speech) e Language (sentimento e Text Analytics "
               "for Health), camada gratuita F0. As chaves ficam no .env e nunca aparecem aqui.")
    if st.button("Testar conexão com a Azure"):
        with st.spinner("Chamando os serviços..."):
            st.session_state["azure_status"] = verificar_azure()
    for status in st.session_state.get("azure_status", []):
        nome = {"speech": "Speech", "language": "Language"}.get(status.servico, status.servico)
        if status.disponivel:
            st.success(f"**{nome}** respondendo em {status.latencia_ms} ms · {status.detalhe}")
        elif status.configurado:
            st.error(f"**{nome}** configurado, mas sem resposta: {status.detalhe}")
        else:
            st.warning(f"**{nome}** não configurado: {status.detalhe}. O sistema usa o modo local.")
    if "azure_status" not in st.session_state:
        st.caption(f"Speech {'configurado' if AZURE.fala_configurada else 'não configurado'} · "
                   f"Language {'configurado' if AZURE.linguagem_configurada else 'não configurado'}")


def _linha_alerta(leitura, resultado, leitos: dict[str, str]) -> str:
    a = resultado.anomalia
    return (f"{PRIORIDADE.get(a.prioridade, '')} `{leitura.instante:%H:%M:%S}` "
            f"**{leitos.get(a.prontuario, a.prontuario)}** · {a.modalidade} · {a.descricao}")


def _grafico(pontos: list[dict], inicio_evento) -> alt.Chart:
    tabela = pd.DataFrame(pontos).melt(id_vars="instante", var_name="sinal", value_name="valor").dropna()
    tabela["sinal"] = tabela["sinal"].map(SERIES)
    graficos = []
    for sinal in SERIES.values():
        camada = alt.Chart(tabela[tabela["sinal"] == sinal]).mark_line(color="#2b6cb0").encode(
            x=alt.X("instante:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
            y=alt.Y("valor:Q", title=sinal, scale=alt.Scale(zero=False))).properties(height=110)
        if inicio_evento is not None:
            camada = camada + alt.Chart(pd.DataFrame({"instante": [inicio_evento]})).mark_rule(
                color="#c53030", strokeDash=[4, 4]).encode(x="instante:T")
        graficos.append(camada)
    return alt.vconcat(*graficos)


def _executar(banco, cenario: str, opcoes: dict) -> dict:
    leitos = {p["prontuario"]: p["leito"] for p in banco.listar_pacientes()}
    banco.limpar_monitoramento()
    central = CentralAlertas(banco)
    metricas = st.empty()
    grafico = st.empty()
    st.markdown("**Alertas na ordem em que chegam à central**")
    feed = st.empty()
    linhas: list[str] = []
    pontos: list[dict] = []
    contagem = {"leituras": 0, "alertas": 0}
    injecao = None

    def painel(leitura) -> None:
        colunas = metricas.columns(3)
        colunas[0].metric("Horário simulado", f"{leitura.instante:%d/%m %H:%M:%S}")
        colunas[1].metric("Leituras processadas", contagem["leituras"])
        colunas[2].metric("Alertas emitidos", contagem["alertas"])

    def ao_processar(leitura, resultados) -> None:
        contagem["leituras"] += 1
        novos = [r for r in resultados if r.emitido]
        contagem["alertas"] += len(novos)
        linhas.extend(_linha_alerta(leitura, r, leitos) for r in novos)
        if cenario == "bidmc":
            pontos.append({"instante": leitura.instante,
                           **{k: leitura.valores.get(k) for k in SERIES}})
        passo = 5 if cenario == "bidmc" else 40
        if novos or contagem["leituras"] % passo == 0:
            painel(leitura)
            if cenario == "bidmc" and pontos:
                grafico.altair_chart(_grafico(pontos, injecao.inicio if injecao else None),
                                     width="stretch")
            feed.markdown("\n\n".join(reversed(linhas[-14:])) or "_Nenhum alerta ainda._")

    if cenario == "bidmc":
        leituras, injecao = demonstracao.leituras_bidmc(opcoes["registro"], opcoes["evento"] or None)
        monitor = Monitor(central, [DetectorSinaisVitais(ParametrosSinaisVitais.alta_frequencia())])
        resumo = monitor.reproduzir(leituras, aceleracao=opcoes["aceleracao"], ao_processar=ao_processar)
    else:
        dados = sintetico.preparar_monitoramento(banco)
        resumo = demonstracao.fluxo(banco, central, dados, horas=24, com_queda=opcoes["queda"],
                                    aceleracao=opcoes["aceleracao"], ao_processar=ao_processar)
    if contagem["leituras"]:
        feed.markdown("\n\n".join(reversed(linhas)) or "_Nenhum alerta._")
    r = resumo.como_dicionario()
    st.success(f"Simulação concluída: {r['leituras']} leituras, {r['alertas_emitidos']} alertas emitidos, "
               f"{r['suprimidos']} repetições suprimidas e {r['abaixo_do_limiar']} achados abaixo do "
               "limiar. Os alertas já aparecem mais abaixo, nesta aba.")
    return {"cenario": cenario, "linhas": linhas, "resumo": resumo.como_dicionario(),
            "injecao": injecao.inicio if injecao else None, "pontos": pontos}


def secao_simulacao(banco) -> None:
    st.caption("Reproduz um fluxo de leituras como se chegasse agora, respeitando o intervalo real "
               "entre elas (acelerado). Os detectores decidem a cada leitura e a central filtra o que "
               "vira alerta. Cada simulação começa limpando o histórico de alertas do monitoramento.")
    cenario = st.radio("Cenário", list(CENARIOS), format_func=CENARIOS.get, horizontal=True,
                       key="simulacao_cenario")
    opcoes: dict = {}
    if cenario == "bidmc":
        registros = demonstracao.numeros_bidmc()
        if not registros:
            st.warning("BIDMC ausente: rode `make baixar-dados`.")
            return
        c1, c2, c3 = st.columns(3)
        opcoes["registro"] = c1.selectbox("Registro BIDMC", registros,
                                          index=registros.index(7) if 7 in registros else 0,
                                          help="Paciente real de UTI, sinais a 1 Hz (cerca de 8 min).")
        opcoes["evento"] = c2.selectbox("Evento injetado", list(EVENTOS), format_func=EVENTOS.get)
        opcoes["aceleracao"] = c3.select_slider("Velocidade", [5, 10, 20, 40, 80], value=20,
                                                format_func=lambda v: f"{v}×")
        st.caption("Paciente exibido como ENF-210. A linha vermelha tracejada marca o início do evento "
                   "injetado.")
    else:
        c1, c2 = st.columns(2)
        velocidades = {1800: "30 min por segundo", 3600: "1 h por segundo", 7200: "2 h por segundo"}
        opcoes["aceleracao"] = c1.select_slider("Velocidade", list(velocidades), value=3600,
                                                format_func=velocidades.get)
        opcoes["queda"] = c2.checkbox("Encaixar a queda real do UR Fall no ENF-221", value=True)
    if st.button("▶ Iniciar simulação", type="primary"):
        st.session_state["simulacao_ultima"] = _executar(banco, cenario, opcoes)
        return
    ultima = st.session_state.get("simulacao_ultima")
    if ultima:
        r = ultima["resumo"]
        st.caption(f"Última simulação ({CENARIOS[ultima['cenario']].split(',')[0].lower()}): "
                   f"{r['leituras']} leituras · {r['alertas_emitidos']} alertas · "
                   f"{r['suprimidos']} repetições suprimidas · {r['abaixo_do_limiar']} abaixo do limiar")
        if ultima["pontos"]:
            st.altair_chart(_grafico(ultima["pontos"], ultima["injecao"]), width="stretch")
        st.markdown("\n\n".join(reversed(ultima["linhas"])) or "_Nenhum alerta._")
