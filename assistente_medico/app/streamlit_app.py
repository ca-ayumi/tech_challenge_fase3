"""Interface web do assistente medico (Streamlit).

Cinco abas, espelhando o que a instituicao precisa ver:

- **Assistente**: a conversa, com a base da resposta sempre visivel;
- **Painel**: pacientes internados e alertas abertos;
- **Validacoes**: fila de sugestoes pendentes de decisao humana;
- **Auditoria**: trilha completa das ultimas interacoes;
- **Politicas**: limites de atuacao vigentes.

Executar com:
    streamlit run assistente_medico/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):  # execucao direta pelo streamlit
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistente_medico.auditoria import Auditoria, configurar_logging
from assistente_medico.config import MODELO, NOME_ASSISTENTE, VERSAO_ASSISTENTE
from assistente_medico.dados.banco import preparar_banco
from assistente_medico.dados.contexto_paciente import montar_contexto
from assistente_medico.explicabilidade import Explicacao, FonteUtilizada
from assistente_medico.seguranca.politicas import resumo_politicas
from assistente_medico.seguranca.regras_clinicas import avaliar

CORES_PRIORIDADE = {
    "maxima": "🔴", "alta": "🟠", "media": "🟡", "informativa": "🔵",
}

st.set_page_config(page_title=NOME_ASSISTENTE, page_icon="🏥", layout="wide")


@st.cache_resource(show_spinner="Carregando modelo e base institucional...")
def carregar_assistente(backend: str):
    from assistente_medico.grafo import AssistenteMedico, Dependencias
    from assistente_medico.llm import criar_modelo_chat

    configurar_logging()
    return AssistenteMedico(Dependencias(
        banco=preparar_banco(), modelo=criar_modelo_chat(backend)
    ))


@st.cache_resource
def carregar_banco():
    return preparar_banco()


def bloco_explicacao(explicacao: dict) -> None:
    if not explicacao:
        return
    objeto = Explicacao(
        fontes=[FonteUtilizada(**f) for f in explicacao.get("fontes", [])],
        campos_prontuario=explicacao.get("campos_prontuario", []),
        regras_aplicadas=[r for r in explicacao.get("regras_aplicadas", []) if r],
        citadas=explicacao.get("referencias_citadas", []),
        recuperadas=explicacao.get("referencias_recuperadas", []),
        nao_fundamentadas=explicacao.get("citacoes_nao_fundamentadas", []),
    )
    with st.expander("Base da resposta (explicabilidade)", expanded=objeto.tem_alerta_de_fonte):
        st.markdown(objeto.como_markdown())


def aba_assistente(assistente, banco) -> None:
    coluna_conversa, coluna_lateral = st.columns([3, 2])

    with coluna_lateral:
        st.subheader("Contexto do paciente")
        pacientes = banco.listar_pacientes()
        opcoes = ["(nenhum)"] + [
            f"{p['leito']} — {p['nome']} ({p['prontuario']})" for p in pacientes
        ]
        escolha = st.selectbox("Paciente em foco", opcoes, key="paciente_escolhido")
        prontuario = None
        if escolha != "(nenhum)":
            prontuario = escolha.split("(")[-1].rstrip(")")
            contexto = montar_contexto(banco, prontuario)
            if contexto:
                st.caption(f"Leito {contexto.leito} · {contexto.unidade}")
                st.metric("Exames pendentes", len(contexto.exames_pendentes))
                alertas = avaliar(contexto)
                if alertas:
                    st.markdown("**Pendencias detectadas**")
                    for alerta in alertas:
                        icone = CORES_PRIORIDADE.get(alerta.prioridade, "⚪")
                        st.markdown(f"{icone} {alerta.mensagem}  \n`{alerta.fonte}`")
                else:
                    st.success("Sem pendencias pelas regras vigentes.")
                with st.expander("Dados enviados ao modelo"):
                    st.code(contexto.como_texto(), language="text")

    with coluna_conversa:
        st.subheader("Conversa")
        for mensagem in st.session_state.get("historico", []):
            with st.chat_message(mensagem["papel"]):
                st.markdown(mensagem["texto"])
                if mensagem.get("explicacao"):
                    bloco_explicacao(mensagem["explicacao"])
                if mensagem.get("rodape"):
                    st.caption(mensagem["rodape"])

        pergunta = st.chat_input("Pergunta clinica ou sobre protocolo institucional")
        if pergunta:
            st.session_state.setdefault("historico", []).append(
                {"papel": "user", "texto": pergunta}
            )
            with st.chat_message("user"):
                st.markdown(pergunta)

            with st.chat_message("assistant"):
                with st.spinner("Consultando protocolos e prontuario..."):
                    estado = assistente.responder(
                        pergunta,
                        perfil=st.session_state.get("perfil", "medico"),
                        usuario=st.session_state.get("usuario", "nao identificado"),
                        prontuario=prontuario,
                    )
                st.markdown(estado["resposta"])
                bloco_explicacao(estado.get("explicacao", {}))

                partes = [f"interacao `{estado['interacao']}`",
                          f"categoria `{estado['categoria']}`",
                          f"etapas: {len(estado.get('etapas', []))}"]
                if estado.get("violacoes"):
                    partes.append("politicas: "
                                  + ", ".join(v["codigo"] for v in estado["violacoes"]))
                if estado.get("validacao_id"):
                    partes.append(f"validacao pendente #{estado['validacao_id']}")
                rodape = " · ".join(partes)
                st.caption(rodape)

                if estado.get("alertas"):
                    st.warning(f"{len(estado['alertas'])} alerta(s) emitido(s) para a equipe.")

            st.session_state["historico"].append({
                "papel": "assistant", "texto": estado["resposta"],
                "explicacao": estado.get("explicacao", {}), "rodape": rodape,
            })


def aba_painel(banco) -> None:
    st.subheader("Pacientes internados")
    pacientes = banco.listar_pacientes()
    linhas = []
    for paciente in pacientes:
        contexto = montar_contexto(banco, paciente["prontuario"])
        alertas = avaliar(contexto) if contexto else []
        maior = alertas[0].prioridade if alertas else "-"
        linhas.append({
            "Leito": paciente["leito"],
            "Unidade": paciente["unidade"],
            "Paciente": paciente["nome"],
            "Motivo": paciente["motivo_internacao"][:44],
            "Exames pendentes": len(contexto.exames_pendentes) if contexto else 0,
            "Pendencias": len(alertas),
            "Maior prioridade": f"{CORES_PRIORIDADE.get(maior, '')} {maior}",
        })
    st.dataframe(linhas, use_container_width=True, hide_index=True)

    st.subheader("Alertas abertos no painel institucional")
    alertas_registrados = banco.listar_alertas(limite=50)
    if not alertas_registrados:
        st.info("Nenhum alerta registrado ainda. Faca uma pergunta sobre um paciente "
                "na aba Assistente para que as regras sejam aplicadas.")
    else:
        st.dataframe([
            {
                "Prioridade": f"{CORES_PRIORIDADE.get(a['prioridade'], '')} {a['prioridade']}",
                "Prontuario": a["prontuario"],
                "Mensagem": a["mensagem"],
                "Fonte": a["fonte"],
                "Criado em": a["criado_em"],
            }
            for a in alertas_registrados
        ], use_container_width=True, hide_index=True)


def aba_validacoes(banco) -> None:
    st.subheader("Validacao humana obrigatoria")
    st.caption("Toda sugestao gerada pelo assistente entra como pendente e so tem efeito "
               "apos decisao registrada de profissional habilitado (PROT-GOV-010 §5).")

    status = st.radio("Status", ["pendente", "aceita", "recusada", "modificada"],
                      horizontal=True)
    validacoes = banco.listar_validacoes(status=status, limite=30)
    if not validacoes:
        st.info(f"Nenhuma validacao com status '{status}'.")
        return

    for validacao in validacoes:
        with st.expander(
            f"#{validacao['id']} · prontuario {validacao['prontuario'] or '-'} · "
            f"{validacao['criado_em']}"
        ):
            st.markdown(validacao["sugestao"])
            st.caption(f"interacao `{validacao['interacao']}` · fontes: {validacao['fontes']}")
            if status == "pendente":
                profissional = st.text_input(
                    "Profissional responsavel", key=f"prof_{validacao['id']}",
                    placeholder="Dra. Fulana de Tal (CRM-SP 00.000)",
                )
                justificativa = st.text_area(
                    "Justificativa (obrigatoria em recusa ou modificacao)",
                    key=f"just_{validacao['id']}",
                )
                colunas = st.columns(3)
                for coluna, decisao in zip(colunas, ["aceita", "modificada", "recusada"], strict=True):
                    if coluna.button(decisao.capitalize(), key=f"{decisao}_{validacao['id']}"):
                        if not profissional.strip():
                            st.error("Informe o profissional responsavel.")
                        elif decisao != "aceita" and not justificativa.strip():
                            st.error("Justificativa obrigatoria para recusa ou modificacao.")
                        else:
                            banco.decidir_validacao(
                                validacao["id"], decisao, profissional, justificativa or None
                            )
                            Auditoria(banco=banco).registrar(
                                validacao["interacao"], "decisao_humana",
                                {"validacao_id": validacao["id"], "status": decisao,
                                 "justificativa": justificativa},
                                usuario=profissional,
                            )
                            st.success(f"Validacao {validacao['id']} registrada como {decisao}.")
                            st.rerun()


def aba_auditoria(banco) -> None:
    st.subheader("Trilha de auditoria")
    auditoria = Auditoria(banco=banco)
    interacoes = auditoria.ultimas_interacoes(limite=25)
    if not interacoes:
        st.info("Nenhuma interacao registrada ainda.")
        return

    escolhida = st.selectbox(
        "Interacao",
        [i["interacao"] for i in interacoes],
        format_func=lambda x: next(
            f"{i['registrado_em']} · {x} · {i['eventos']} eventos"
            for i in interacoes if i["interacao"] == x
        ),
    )
    eventos = auditoria.trilha(escolhida)
    st.caption(f"{len(eventos)} eventos · dados pessoais mascarados no registro "
               "(PROT-GOV-010 §6 e §8)")
    for evento in eventos:
        with st.expander(f"{evento['registrado_em']} · {evento['evento']}"):
            st.json(evento["conteudo"])


def aba_politicas() -> None:
    st.subheader("Limites de atuacao vigentes")
    st.caption("Politicas derivadas do PROT-GOV-010. Entrada: avaliadas antes de chamar o "
               "modelo. Saida: avaliadas sobre o texto gerado, antes da entrega.")
    st.dataframe(resumo_politicas(), use_container_width=True, hide_index=True)


def main() -> None:
    banco = carregar_banco()

    with st.sidebar:
        st.title("🏥 " + NOME_ASSISTENTE)
        st.caption(f"versao {VERSAO_ASSISTENTE}")
        st.session_state["perfil"] = st.selectbox(
            "Perfil", ["medico", "enfermeiro", "farmaceutico", "residente", "paciente"]
        )
        st.session_state["usuario"] = st.text_input("Identificacao", value="dra.helena")
        backend = st.selectbox(
            "Backend do modelo", ["mlx", "eco", "transformers"],
            index=["mlx", "eco", "transformers"].index(MODELO.backend)
            if MODELO.backend in ("mlx", "eco", "transformers") else 0,
            help="'mlx' usa a LLM ajustada por LoRA. 'eco' e o backend deterministico, "
                 "sem rede neural, util para demonstrar o fluxo sem carregar o modelo.",
        )
        if st.button("Limpar conversa"):
            st.session_state["historico"] = []
            st.rerun()
        st.divider()
        st.caption("Este assistente nao prescreve, nao informa dose e nao autoriza alta. "
                   "Toda sugestao exige validacao humana.")

    assistente = carregar_assistente(backend)

    abas = st.tabs(["Assistente", "Painel", "Validacoes", "Auditoria", "Politicas"])
    with abas[0]:
        aba_assistente(assistente, banco)
    with abas[1]:
        aba_painel(banco)
    with abas[2]:
        aba_validacoes(banco)
    with abas[3]:
        aba_auditoria(banco)
    with abas[4]:
        aba_politicas()


main()
