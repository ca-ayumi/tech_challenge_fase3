"""Secao "Enviar gravacao" da aba Monitoramento: upload de video ou audio pela equipe."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from assistente_medico.config import AZURE
from assistente_medico.multimodal import envio

CONFIGURACOES_AREAS = Path(envio.__file__).parent / "video" / "configuracoes"
TIPOS_VIDEO = {"exercicio": "Sessão de fisioterapia (repetições, amplitude, compensações)",
               "quarto": "Câmera do quarto (queda, áreas críticas, objetos de risco)",
               "auto": "Detectar automaticamente"}
PRIORIDADE = {"maxima": "🔴", "alta": "🟠", "media": "🟡", "informativa": "🔵"}


def _rotulo_paciente(pacientes: dict, prontuario: str) -> str:
    p = pacientes.get(prontuario, {})
    return f"{p.get('leito', prontuario)} · {p.get('nome', '')}"


def _barra():
    barra = st.progress(0.0, text="Preparando...")
    return lambda fracao, texto: barra.progress(min(max(fracao, 0.0), 1.0), text=texto)


def _formulario_video(banco, prontuario: str, arquivo) -> None:
    coluna_tipo, coluna_areas = st.columns(2)
    tipo = coluna_tipo.selectbox("Tipo de análise", list(TIPOS_VIDEO), format_func=TIPOS_VIDEO.get)
    opcoes_areas = ["nenhuma"] + sorted(c.name for c in CONFIGURACOES_AREAS.glob("*.json"))
    areas = coluna_areas.selectbox(
        "Áreas críticas da câmera", opcoes_areas, disabled=tipo == "exercicio",
        help="Polígonos configurados por câmera (ex.: porta de saída do quarto). "
             "Arquivos em multimodal/video/configuracoes/.")
    anotar = st.checkbox("Gerar vídeo anotado com o esqueleto", value=True)
    if st.button("Analisar vídeo", type="primary"):
        try:
            caminho = envio.salvar_envio(arquivo.name, arquivo.getvalue(), "video")
            resultado = envio.analisar_video_enviado(
                banco, caminho, prontuario, tipo=tipo,
                areas=CONFIGURACOES_AREAS / areas if areas != "nenhuma" and tipo != "exercicio" else None,
                anotar=anotar, progresso=_barra())
        except envio.EnvioInvalido as erro:
            st.error(str(erro))
            return
        except ImportError:
            st.error("A análise de vídeo precisa do ultralytics: rode `make instalar-video`.")
            return
        except ValueError as erro:
            st.error(f"Não foi possível ler o vídeo: {erro}")
            return
        st.session_state["envio_resultado"] = resultado


def _formulario_audio(banco, prontuario: str, arquivo) -> None:
    azure_ok = AZURE.fala_configurada and AZURE.linguagem_configurada
    usar_azure = st.checkbox("Transcrever e analisar o texto com a Azure (Speech + Language)",
                             value=azure_ok, disabled=not azure_ok,
                             help="Sem a Azure, só a voz é analisada (Praat, local).")
    registrada = envio.basal_do_paciente(prontuario)
    opcoes = (["registrada"] if registrada else []) + ["enviar", "referencia"]
    rotulos = {"registrada": "Gravação de base já registrada do paciente",
               "enviar": "Enviar uma gravação de base", "referencia": "Faixas de referência (sem base)"}
    escolha = st.radio("Comparar a voz com", opcoes, format_func=rotulos.get, horizontal=True,
                       help="A comparação com a própria voz do paciente é mais sensível; "
                            "só use a base registrada se a gravação enviada for da mesma pessoa.")
    arquivo_basal = None
    if escolha == "enviar":
        arquivo_basal = st.file_uploader("Gravação de base", type=[e[1:] for e in envio.EXTENSOES_AUDIO],
                                         key="envio_basal")
    if st.button("Analisar áudio", type="primary"):
        try:
            caminho = envio.salvar_envio(arquivo.name, arquivo.getvalue(), "audio")
            basal = registrada if escolha == "registrada" else None
            if escolha == "enviar" and arquivo_basal is not None:
                basal = envio.salvar_envio(arquivo_basal.name, arquivo_basal.getvalue(), "audio")
            resultado = envio.analisar_audio_enviado(banco, caminho, prontuario, basal=basal,
                                                     usar_azure=usar_azure, progresso=_barra())
        except envio.EnvioInvalido as erro:
            st.error(str(erro))
            return
        except ValueError as erro:
            st.error(f"Não foi possível analisar o áudio: {erro}")
            return
        st.session_state["envio_resultado"] = resultado


def _mostrar_video(resultado: envio.ResultadoEnvio) -> None:
    analise = resultado.analise
    texto = analise.relatorio().split("\n## Arquivos", 1)[0]
    st.markdown(texto.split("\n", 1)[1])
    gerados = analise.arquivos_gerados
    if "video anotado" in gerados and Path(gerados["video anotado"]).exists():
        st.video(gerados["video anotado"])
    imagens = [c for c in gerados.values() if c.endswith((".jpg", ".png")) and Path(c).exists()]
    if imagens:
        st.image(imagens, width=330)


def _mostrar_audio(resultado: envio.ResultadoEnvio) -> None:
    st.audio(str(resultado.arquivo))
    if resultado.analise.fontes.get("transcricao") == "indisponivel":
        st.caption("Sem transcrição: só a voz foi analisada.")
    st.markdown(resultado.analise.relatorio().split("\n", 1)[1])


def _mostrar_resultado(pacientes: dict, resultado: envio.ResultadoEnvio) -> None:
    st.success(f"{resultado.arquivo.name.split('_', 1)[-1]} analisado para "
               f"{_rotulo_paciente(pacientes, resultado.prontuario)}. Relatório salvo em "
               f"`{resultado.relatorio}`.")
    if resultado.consolidados_novos:
        for alerta in resultado.consolidados_novos:
            st.error(f"{PRIORIDADE.get(alerta['prioridade'], '')} Alerta consolidado #{alerta['id']}: "
                     f"{alerta['mensagem'].split('. Evidencias: ')[0]}")
    if resultado.alertas_novos:
        st.warning(f"{len(resultado.alertas_novos)} alerta(s) enviado(s) à central: " + "; ".join(
            a["mensagem"][:110] for a in resultado.alertas_novos[:4]))
    elif resultado.analise.anomalias:
        st.info("Os desvios abaixo ficaram registrados, mas não geraram alerta novo "
                "(abaixo da severidade mínima ou já alertados nos últimos 30 min).")
    else:
        st.info("Nenhum desvio relevante: nada foi enviado à central de alertas.")
    if resultado.midia == "video":
        _mostrar_video(resultado)
    else:
        _mostrar_audio(resultado)
    if st.button("Fechar resultado"):
        st.session_state.pop("envio_resultado", None)
        st.rerun()


def secao_envio(banco) -> None:
    st.caption("A equipe envia a gravação de uma sessão de fisioterapia, da câmera do quarto ou de "
               "uma consulta. O sistema gera o relatório automático, manda os achados para a central "
               "de alertas e recalcula o risco do paciente.")
    pacientes = {p["prontuario"]: p for p in banco.listar_pacientes()}
    prontuario = st.selectbox("Paciente da gravação", list(pacientes),
                              format_func=lambda p: _rotulo_paciente(pacientes, p), key="envio_paciente")
    arquivo = st.file_uploader(
        "Gravação (vídeo ou áudio)", key="envio_arquivo",
        type=[e[1:] for e in envio.EXTENSOES_VIDEO + envio.EXTENSOES_AUDIO],
        help="O tipo é reconhecido pela extensão: vídeo (mp4, mov, avi, webm, ogv, mkv) ou "
             "áudio (wav, mp3, flac).")
    midia = envio.midia_do_arquivo(arquivo.name) if arquivo is not None else None
    if midia == "video":
        st.caption("🎥 Vídeo reconhecido: análise de pose (YOLOv8-pose) e objetos (YOLOv8).")
        _formulario_video(banco, prontuario, arquivo)
    elif midia == "audio":
        st.caption("🎙️ Áudio reconhecido: voz (Praat) e texto (Azure Speech + Language).")
        _formulario_audio(banco, prontuario, arquivo)
    resultado = st.session_state.get("envio_resultado")
    if resultado is not None:
        st.divider()
        _mostrar_resultado(pacientes, resultado)
