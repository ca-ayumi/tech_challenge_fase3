"""Regras deterministicas de vigilancia clinica.

Estas regras nao dependem do modelo de linguagem: sao verificacoes objetivas
sobre os dados estruturados do prontuario, com a secao do protocolo que as
justifica. Isso separa o que precisa ser confiavel (gatilho, prazo, conflito de
alergia) do que e gerado por linguagem natural (a redacao da resposta).

Cada regra devolve um ``Alerta`` com prioridade, mensagem e fonte citavel.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..dados.contexto_paciente import ContextoPaciente, formatar_intervalo, minutos_desde

PRIORIDADES = ("maxima", "alta", "media", "informativa")


@dataclass(frozen=True)
class Alerta:
    """Alerta objetivo emitido para a equipe assistencial."""

    prioridade: str
    categoria: str
    mensagem: str
    fonte: str
    regra: str
    dados: dict[str, Any] = field(default_factory=dict)

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "prioridade": self.prioridade,
            "categoria": self.categoria,
            "mensagem": self.mensagem,
            "fonte": self.fonte,
            "regra": self.regra,
            "dados": self.dados,
        }

    def como_linha(self) -> str:
        return f"[{self.prioridade.upper()}] {self.mensagem} (fonte: {self.fonte})"


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in sem_acento if not unicodedata.combining(c)).strip().lower()


def _numero(valor: Any) -> float | None:
    """Converte resultados textuais como '5,2' ou 'Leucocitos 19.400/mm3' em numero."""
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip().replace(",", ".")
    numero = ""
    for caractere in texto:
        if caractere.isdigit() or (caractere == "." and numero and "." not in numero):
            numero += caractere
        elif numero:
            break
    try:
        return float(numero) if numero else None
    except ValueError:
        return None


def pressao_arterial_media(pas: Any, pad: Any) -> float | None:
    sistolica, diastolica = _numero(pas), _numero(pad)
    if sistolica is None or diastolica is None:
        return None
    return round((sistolica + 2 * diastolica) / 3, 1)


def _exame_por_codigo(contexto: ContextoPaciente, prefixo: str,
                      apenas_liberados: bool = True) -> list[dict[str, Any]]:
    fonte = contexto.exames_liberados if apenas_liberados else (
        contexto.exames_liberados + contexto.exames_pendentes)
    return [e for e in fonte if str(e.get("codigo", "")).upper().startswith(prefixo)]


# --------------------------------------------------------------------- regras
def regra_sepse_antimicrobiano(contexto: ContextoPaciente) -> list[Alerta]:
    if not contexto.tem_protocolo("PROT-SEP-001"):
        return []
    protocolo = next(p for p in contexto.protocolos if p["protocolo"] == "PROT-SEP-001")
    decorridos = minutos_desde(protocolo.get("aberto_em"))
    evento = contexto.evento("antimicrobiano_administrado")
    if evento and not evento.get("ocorreu") and decorridos is not None and decorridos > 60:
        return [Alerta(
            prioridade="alta",
            categoria="prazo_protocolo",
            mensagem=(
                "Protocolo de sepse aberto "
                f"{formatar_intervalo(decorridos)} sem registro de administracao da primeira "
                "dose de antimicrobiano; o pacote da primeira hora esta vencido."
            ),
            fonte="PROT-SEP-001 §4",
            regra="sepse_antimicrobiano_60min",
            dados={"minutos_desde_abertura": decorridos},
        )]
    return []


def regra_sepse_coletas_pendentes(contexto: ContextoPaciente) -> list[Alerta]:
    if not contexto.tem_protocolo("PROT-SEP-001"):
        return []
    alertas = []
    for exame in contexto.exames_pendentes:
        decorridos = minutos_desde(exame.get("solicitado_em"))
        if decorridos is None:
            continue
        codigo = str(exame.get("codigo", "")).upper()
        if codigo.startswith("LAC") and decorridos > 60:
            alertas.append(Alerta(
                prioridade="alta", categoria="exame_pendente",
                mensagem=(f"Lactato arterial pendente {formatar_intervalo(decorridos)}, "
                          "acima do prazo do pacote da primeira hora."),
                fonte="PROT-SEP-001 §4", regra="sepse_lactato_pendente",
                dados={"exame": exame.get("nome"), "minutos": decorridos},
            ))
        elif codigo.startswith("HMC") and decorridos > 90:
            alertas.append(Alerta(
                prioridade="alta", categoria="exame_pendente",
                mensagem=(f"{exame.get('nome')} pendente {formatar_intervalo(decorridos)}, "
                          "acima do limite de 90 minutos previsto no protocolo."),
                fonte="PROT-SEP-001 §9", regra="sepse_hemocultura_pendente",
                dados={"exame": exame.get("nome"), "minutos": decorridos},
            ))
    return alertas


def regra_lactato_elevado(contexto: ContextoPaciente) -> list[Alerta]:
    # Interessa o lactato mais recente: uma curva em queda nao deve disparar
    # alerta por causa do primeiro valor da serie.
    coletas = sorted(
        _exame_por_codigo(contexto, "LAC"),
        key=lambda e: e.get("liberado_em") or "",
        reverse=True,
    )
    for exame in coletas[:1]:
        valor = _numero(exame.get("resultado"))
        if valor is not None and valor > 4:
            return [Alerta(
                prioridade="alta", categoria="resultado_critico",
                mensagem=(f"Lactato de {exame.get('resultado')} mmol/L acima de 4: criterio de "
                          "gravidade e de avaliacao para terapia intensiva."),
                fonte="PROT-SEP-001 §8", regra="lactato_maior_4",
                dados={"lactato": valor},
            )]
    return []


def regra_hipotensao(contexto: ContextoPaciente) -> list[Alerta]:
    sinal = contexto.ultimo_sinal()
    if not sinal:
        return []
    pam = pressao_arterial_media(sinal.get("pas"), sinal.get("pad"))
    if pam is not None and pam < 65:
        return [Alerta(
            prioridade="alta", categoria="sinal_vital",
            mensagem=(f"Pressao arterial media estimada em {pam} mmHg, abaixo da meta de "
                      "65 mmHg."),
            fonte="PROT-SEP-001 §7", regra="pam_menor_65",
            dados={"pam": pam},
        )]
    return []


def regra_supra_st(contexto: ContextoPaciente) -> list[Alerta]:
    for exame in _exame_por_codigo(contexto, "ECG"):
        resultado = _normalizar(exame.get("resultado"))
        if "supra" in resultado:
            evento = contexto.evento("hemodinamica_ativada")
            if evento and not evento.get("ocorreu"):
                return [Alerta(
                    prioridade="maxima", categoria="prazo_protocolo",
                    mensagem=("Eletrocardiograma com supradesnivelamento de ST sem registro de "
                              "ativacao da hemodinamica; meta institucional de porta-balao e de "
                              "90 minutos."),
                    fonte="PROT-CAR-002 §5", regra="supra_st_sem_hemodinamica",
                    dados={"ecg": exame.get("resultado")},
                )]
            return [Alerta(
                prioridade="alta", categoria="resultado_critico",
                mensagem="Eletrocardiograma com supradesnivelamento de ST.",
                fonte="PROT-CAR-002 §9", regra="supra_st",
                dados={"ecg": exame.get("resultado")},
            )]
    return []


def regra_avc_tomografia(contexto: ContextoPaciente) -> list[Alerta]:
    if not contexto.tem_protocolo("PROT-NEU-003"):
        return []
    alertas = []
    for exame in contexto.exames_pendentes:
        if str(exame.get("codigo", "")).upper().startswith("TC"):
            decorridos = minutos_desde(exame.get("solicitado_em"))
            if decorridos is not None and decorridos > 25:
                alertas.append(Alerta(
                    prioridade="alta", categoria="exame_pendente",
                    mensagem=(f"Tomografia de cranio pendente {formatar_intervalo(decorridos)}, "
                              "acima do prazo de 25 minutos do Codigo AVC."),
                    fonte="PROT-NEU-003 §8", regra="avc_tc_25min",
                    dados={"minutos": decorridos},
                ))
    evento = contexto.evento("ultimo_momento_visto_bem")
    if evento and evento.get("registrado_em"):
        decorridos = minutos_desde(evento["registrado_em"])
        if decorridos is not None:
            restante = 270 - decorridos     # 4,5 horas de janela de trombolise
            if 0 < restante <= 30:
                alertas.append(Alerta(
                    prioridade="maxima", categoria="janela_terapeutica",
                    mensagem=(f"Janela de trombolise se encerra em {restante} minutos "
                              f"(ultimo momento visto bem {formatar_intervalo(decorridos)})."),
                    fonte="PROT-NEU-003 §5", regra="avc_janela_trombolise",
                    dados={"minutos_restantes": restante},
                ))
            elif restante <= 0:
                alertas.append(Alerta(
                    prioridade="alta", categoria="janela_terapeutica",
                    mensagem=("Janela de trombolise endovenosa de 4,5 horas ja encerrada; "
                              "avaliar elegibilidade para trombectomia."),
                    fonte="PROT-NEU-003 §5", regra="avc_janela_encerrada",
                    dados={"minutos_desde_inicio": decorridos},
                ))
    return alertas


def regra_cetoacidose_potassio(contexto: ContextoPaciente) -> list[Alerta]:
    if not contexto.tem_protocolo("PROT-END-005"):
        return []
    evento = contexto.evento("insulina_iniciada")
    for exame in _exame_por_codigo(contexto, "K"):
        valor = _numero(exame.get("resultado"))
        if valor is not None and valor < 3.3:
            prioridade = "maxima" if (evento and evento.get("ocorreu")) else "alta"
            complemento = (" com insulina ja iniciada" if evento and evento.get("ocorreu") else "")
            return [Alerta(
                prioridade=prioridade, categoria="seguranca_medicamentosa",
                mensagem=(f"Potassio serico de {exame.get('resultado')} mEq/L, abaixo de "
                          f"3,3 mEq/L{complemento}: o protocolo exige reposicao de potassio "
                          "antes da insulinoterapia."),
                fonte="PROT-END-005 §4", regra="cetoacidose_potassio_baixo",
                dados={"potassio": valor, "insulina_iniciada": bool(evento and evento.get("ocorreu"))},
            )]
    return []


def regra_conflito_alergia(contexto: ContextoPaciente) -> list[Alerta]:
    alertas = []
    alergias = [(_normalizar(a), a) for a in contexto.alergias]
    for medicacao in contexto.medicacoes:
        normalizada = _normalizar(medicacao)
        for alergia_normalizada, alergia in alergias:
            if not alergia_normalizada:
                continue
            if alergia_normalizada in normalizada or normalizada in alergia_normalizada:
                alertas.append(Alerta(
                    prioridade="maxima", categoria="seguranca_medicamentosa",
                    mensagem=(f"Conflito de alergia: '{medicacao}' consta como medicacao em uso e "
                              f"'{alergia}' esta na lista de alergias registradas do paciente."),
                    fonte="PROT-ALE-007 §7", regra="conflito_alergia_medicacao",
                    dados={"medicacao": medicacao, "alergia": alergia},
                ))
    return alertas


def regra_registro_alergia_pendente(contexto: ContextoPaciente) -> list[Alerta]:
    if not contexto.tem_protocolo("PROT-ALE-007"):
        return []
    evento = contexto.evento("registro_alergia_atualizado")
    if evento and not evento.get("ocorreu"):
        return [Alerta(
            prioridade="alta", categoria="prazo_protocolo",
            mensagem=("Episodio de anafilaxia sem registro de alergia atualizado no cadastro; "
                      "o protocolo exige o registro em ate 2 horas."),
            fonte="PROT-ALE-007 §7", regra="anafilaxia_registro_alergia",
        )]
    return []


def regra_avaliacao_tev(contexto: ContextoPaciente) -> list[Alerta]:
    evento = contexto.evento("avaliacao_risco_tev")
    if evento is None or evento.get("ocorreu"):
        return []
    if contexto.minutos_internado is not None and contexto.minutos_internado > 24 * 60:
        return [Alerta(
            prioridade="media", categoria="prazo_protocolo",
            mensagem=(f"Paciente internado {formatar_intervalo(contexto.minutos_internado)} sem "
                      "avaliacao de risco de tromboembolismo venoso registrada; o prazo e de "
                      "24 horas."),
            fonte="PROT-TEV-004 §7", regra="tev_sem_avaliacao_24h",
            dados={"minutos_internado": contexto.minutos_internado},
        )]
    return []


def regra_reavaliacao_antimicrobiano(contexto: ContextoPaciente) -> list[Alerta]:
    evento = contexto.evento("reavaliacao_antimicrobiano")
    if evento is None or evento.get("ocorreu"):
        return []
    if contexto.minutos_internado is not None and contexto.minutos_internado > 72 * 60:
        return [Alerta(
            prioridade="media", categoria="prazo_protocolo",
            mensagem=(f"Antimicrobiano em curso {formatar_intervalo(contexto.minutos_internado)} "
                      "sem reavaliacao registrada; o protocolo exige reavaliacao entre 48 e "
                      "72 horas."),
            fonte="PROT-ATB-003 §3", regra="atb_sem_reavaliacao_72h",
            dados={"minutos_internado": contexto.minutos_internado},
        )]
    return []


def regra_descalonamento_disponivel(contexto: ContextoPaciente) -> list[Alerta]:
    for exame in _exame_por_codigo(contexto, "URO"):
        resultado = _normalizar(exame.get("resultado"))
        if "sensivel" in resultado:
            return [Alerta(
                prioridade="media", categoria="racionalizacao",
                mensagem=("Cultura com agente identificado e perfil de sensibilidade disponivel: "
                          "o descalonamento guiado por cultura e obrigatorio."),
                fonte="PROT-ATB-003 §3", regra="descalonamento_por_cultura",
                dados={"cultura": exame.get("resultado")},
            )]
    return []


def regra_funcao_renal(contexto: ContextoPaciente) -> list[Alerta]:
    for exame in _exame_por_codigo(contexto, "CLCR"):
        valor = _numero(exame.get("resultado"))
        if valor is not None and valor < 30:
            return [Alerta(
                prioridade="media", categoria="seguranca_medicamentosa",
                mensagem=(f"Clearance de creatinina estimado em {exame.get('resultado')} mL/min: "
                          "situacao especial que exige revisao de dose pelo medico assistente. "
                          "O assistente nao informa o valor do ajuste."),
                fonte="PROT-TEV-004 §5", regra="clearance_menor_30",
                dados={"clearance": valor},
            )]
    return []


def regra_gatilhos_uti(contexto: ContextoPaciente) -> list[Alerta]:
    sinal = contexto.ultimo_sinal()
    if not sinal:
        return []
    gatilhos = []
    frequencia = _numero(sinal.get("fr"))
    saturacao = _numero(sinal.get("saturacao"))
    glasgow = _numero(sinal.get("glasgow"))
    pam = pressao_arterial_media(sinal.get("pas"), sinal.get("pad"))

    if frequencia is not None and frequencia > 30:
        gatilhos.append(f"frequencia respiratoria de {frequencia:.0f} irpm")
    if saturacao is not None and saturacao < 90:
        gatilhos.append(f"saturacao de {saturacao:.0f}%")
    if pam is not None and pam < 65:
        gatilhos.append(f"pressao arterial media de {pam} mmHg")
    if glasgow is not None and glasgow <= 12:
        gatilhos.append(f"escala de coma de Glasgow de {glasgow:.0f}")

    if gatilhos and not contexto.tem_protocolo("PROT-UTI-002"):
        return [Alerta(
            prioridade="alta", categoria="acionamento",
            mensagem=("Gatilho objetivo de acionamento da terapia intensiva presente ("
                      + ", ".join(gatilhos) + ") sem solicitacao de avaliacao registrada."),
            fonte="PROT-UTI-002 §3", regra="gatilho_uti_sem_solicitacao",
            dados={"gatilhos": gatilhos},
        )]
    return []


REGRAS: tuple[Callable[[ContextoPaciente], list[Alerta]], ...] = (
    regra_conflito_alergia,
    regra_supra_st,
    regra_cetoacidose_potassio,
    regra_avc_tomografia,
    regra_sepse_antimicrobiano,
    regra_sepse_coletas_pendentes,
    regra_lactato_elevado,
    regra_hipotensao,
    regra_gatilhos_uti,
    regra_registro_alergia_pendente,
    regra_avaliacao_tev,
    regra_reavaliacao_antimicrobiano,
    regra_descalonamento_disponivel,
    regra_funcao_renal,
)


def avaliar(contexto: ContextoPaciente) -> list[Alerta]:
    """Aplica todas as regras e devolve os alertas ordenados por prioridade."""
    alertas: list[Alerta] = []
    for regra in REGRAS:
        try:
            alertas.extend(regra(contexto))
        except Exception:  # pragma: no cover - uma regra defeituosa nao derruba o fluxo
            continue
    ordem = {prioridade: indice for indice, prioridade in enumerate(PRIORIDADES)}
    return sorted(alertas, key=lambda alerta: ordem.get(alerta.prioridade, 99))
