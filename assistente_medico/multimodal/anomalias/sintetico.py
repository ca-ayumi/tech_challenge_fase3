"""Historico sintetico de internacao para os 12 pacientes do hospital ficticio.

Gera, de forma deterministica, tres series por paciente, ancoradas na data de
admissao do prontuario:

- sinais vitais horarios (com pressao arterial, que o BIDMC nao tem);
- historico de prescricoes, coerente com as medicacoes em uso;
- sensor de leito a cada 15 min (atividade e posicao).

Sobre essa base normal, injeta cenarios clinicos com rotulo, que servem para
medir os detectores e para a demonstracao. Tudo e ficticio.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ...config import CAMINHOS
from .prescricoes import normalizar

PASTA_PADRAO = CAMINHOS.dados_brutos / "monitoramento"
HORAS_MAXIMAS = 72
HORAS_MINIMAS = 12

DOSES_HABITUAIS = {
    "metformina": (850, "mg", "VO", 12),
    "losartana": (50, "mg", "VO", 24),
    "sinvastatina": (20, "mg", "VO", 24),
    "varfarina": (5, "mg", "VO", 24),
    "atenolol": (50, "mg", "VO", 24),
    "levotiroxina": (75, "mcg", "VO", 24),
    "ceftriaxona": (1000, "mg", "IV", 12),
    "azitromicina": (500, "mg", "IV", 24),
    "furosemida": (40, "mg", "IV", 12),
    "cefazolina": (2000, "mg", "IV", 8),
    "amoxicilina com clavulanato": (875, "mg", "VO", 12),
    "piperacilina com tazobactam": (4500, "mg", "IV", 6),
    "noradrenalina": (0.1, "mcg/kg/min", "IV", None),
    "insulina regular em infusao continua": (7, "UI/h", "IV", None),
    "insulina nph": (10, "UI", "SC", 12),
    "dipirona": (1000, "mg", "IV", 6),
}

BASES = {
    "443902": dict(pas=128, pad=78, fc=82, fr=16, saturacao=97, temperatura=36.9),
    "559120": dict(pas=124, pad=74, fc=88, fr=18, saturacao=95, temperatura=37.4),
    "310445": dict(pas=118, pad=72, fc=80, fr=16, saturacao=98, temperatura=36.8),
    "208731": dict(pas=142, pad=84, fc=74, fr=17, saturacao=96, temperatura=36.5),
    "875431": dict(pas=132, pad=80, fc=76, fr=17, saturacao=96, temperatura=36.6),
    "117256": dict(pas=122, pad=76, fc=78, fr=15, saturacao=98, temperatura=36.7),
}
RUIDO = dict(pas=4.0, pad=3.0, fc=3.0, fr=1.0, saturacao=0.6, temperatura=0.15)

ACAMADOS = {"624190", "662480"}


@dataclass
class _Paciente:
    dados: dict[str, Any]
    inicio: datetime
    fim: datetime
    idade: int

    @property
    def prontuario(self) -> str:
        return self.dados["prontuario"]

    def h(self, horas_antes_do_fim: float) -> datetime:
        return self.fim - timedelta(hours=horas_antes_do_fim)


def _iso(instante: datetime) -> str:
    return instante.isoformat(timespec="minutes")


def _referencia(prontuarios: list[dict[str, Any]]) -> datetime:
    instantes = [datetime.fromisoformat(s["aferido_em"])
                 for p in prontuarios for s in p.get("sinais_vitais", [])]
    return max(instantes).replace(minute=0, second=0, microsecond=0)


def _pacientes(prontuarios: list[dict[str, Any]]) -> list[_Paciente]:
    fim = _referencia(prontuarios)
    saida = []
    for dados in prontuarios:
        admissao = datetime.fromisoformat(dados["data_admissao"])
        inicio = max(admissao, fim - timedelta(hours=HORAS_MAXIMAS)).replace(minute=0)
        nascimento = datetime.fromisoformat(dados["data_nascimento"])
        idade = fim.year - nascimento.year - ((fim.month, fim.day) < (nascimento.month, nascimento.day))
        saida.append(_Paciente(dados, inicio, fim, idade))
    return saida


def _rampa(instante: datetime, inicio: datetime, fim: datetime) -> float:
    if instante <= inicio:
        return 0.0
    if instante >= fim:
        return 1.0
    return (instante - inicio) / (fim - inicio)


def _cenarios_vitais(p: _Paciente) -> list[tuple[str, datetime, datetime, dict[str, float]]]:
    """(evento, inicio, fim, variacao total no fim da rampa)."""
    return {
        "443902": [("deterioracao_septica", p.h(10), p.h(0),
                    dict(temperatura=2.3, fc=36, pas=-32, pad=-16, fr=6))],
        "559120": [("insuficiencia_respiratoria", p.h(8), p.h(0),
                    dict(fr=11, saturacao=-7, fc=18))],
        "310445": [("febre_apos_suspensao", p.h(12), p.h(6), dict(temperatura=2.0, fc=22, fr=4))],
    }.get(p.prontuario, [])


def _sinais_vitais(p: _Paciente, gerador: random.Random) -> list[dict[str, Any]]:
    base = BASES.get(p.prontuario)
    if base is None or p.fim - p.inicio < timedelta(hours=HORAS_MINIMAS):
        return []
    cenarios = _cenarios_vitais(p)
    leituras = []
    instante = p.inicio
    while instante <= p.fim:
        valores = {}
        for variavel, valor in base.items():
            delta = sum(c[3].get(variavel, 0.0) * _rampa(instante, c[1], c[2]) for c in cenarios)
            ruido = gerador.gauss(0, RUIDO[variavel])
            valores[variavel] = valor + delta + ruido
        valores["saturacao"] = min(round(valores["saturacao"]), 100)
        valores["temperatura"] = round(valores["temperatura"], 1)
        for variavel in ("pas", "pad", "fc", "fr"):
            valores[variavel] = round(valores[variavel])
        leituras.append({"prontuario": p.prontuario, "aferido_em": _iso(instante),
                         "glasgow": 15, **valores})
        instante += timedelta(hours=1)
    return leituras


def _prescricao(p: _Paciente, medicamento: str, instante: datetime, dose: float | None = None,
                intervalo: float | None | str = "habitual", situacao: str = "ativa",
                unidade: str | None = None) -> dict[str, Any]:
    habitual = DOSES_HABITUAIS.get(normalizar(medicamento), (None, None, None, None))
    return {
        "prontuario": p.prontuario, "medicamento": medicamento,
        "dose": habitual[0] if dose is None else dose,
        "unidade": unidade or habitual[1], "via": habitual[2],
        "intervalo_horas": habitual[3] if intervalo == "habitual" else intervalo,
        "prescrito_em": _iso(instante), "prescritor": "Equipe assistente (ficticio)",
        "situacao": situacao,
    }


def _prescricoes(p: _Paciente) -> list[dict[str, Any]]:
    admissao = max(datetime.fromisoformat(p.dados["data_admissao"]), p.inicio)
    lista = [_prescricao(p, m, admissao) for m in p.dados.get("medicacoes_em_uso", [])
             if normalizar(m) in DOSES_HABITUAIS]
    pid = p.prontuario
    if pid == "559120":
        lista += [_prescricao(p, "Ceftriaxona", p.h(30), situacao="suspensa"),
                  _prescricao(p, "Amoxicilina com clavulanato", p.h(30))]
    elif pid == "443902":
        lista.append(_prescricao(p, "Cefazolina", p.h(36), dose=20000))
    elif pid == "208731":
        lista += [_prescricao(p, "Furosemida", p.h(40), dose=60),
                  _prescricao(p, "Insulina NPH", p.h(26), dose=100)]
    elif pid == "117256":
        lista += [_prescricao(p, "Enoxaparina", admissao, dose=40, unidade="mg", intervalo=24),
                  _prescricao(p, "Heparina", p.h(20), dose=5000, unidade="UI", intervalo=8)]
    elif pid == "310445":
        lista.append(_prescricao(p, "Ceftriaxona", p.h(20), situacao="suspensa"))
    elif pid == "624190":
        lista += [_prescricao(p, "Noradrenalina", p.h(h), dose=d)
                  for h, d in ((18, 0.15), (10, 0.2), (4, 0.3))]
        lista.append(_prescricao(p, "Morfina", p.h(5), dose=2, unidade="mg", intervalo=4))
    elif pid == "662480":
        lista += [_prescricao(p, "Insulina regular em infusao continua", p.h(h), dose=d)
                  for h, d in ((8, 5), (5, 4))]
        lista.append(_prescricao(p, "Insulina regular em infusao continua", p.h(3),
                                 situacao="suspensa"))
    elif pid == "990117":
        lista.append(_prescricao(p, "Dipirona", admissao + timedelta(minutes=30),
                                 situacao="suspensa"))
    return sorted(lista, key=lambda item: item["prescrito_em"])


def _noite_recente(p: _Paciente) -> datetime:
    """01:00 da madrugada mais recente que ainda deixa 6 h de serie depois."""
    candidata = p.fim.replace(hour=1, minute=0)
    while p.fim - candidata < timedelta(hours=6):
        candidata -= timedelta(days=1)
    return candidata


def _movimentacao(p: _Paciente, gerador: random.Random) -> list[dict[str, Any]]:
    if p.fim - p.inicio < timedelta(hours=24):
        return []
    decubitos = ["decubito_dorsal", "decubito_lateral_direito", "decubito_lateral_esquerdo"]
    acamado = p.prontuario in ACAMADOS
    imobilidade = (p.h(14), p.h(8)) if p.prontuario == "208731" else None
    noite = _noite_recente(p) if p.prontuario == "875431" else None

    registros = []
    instante = p.inicio
    posicao, desde = "decubito_dorsal", instante
    while instante <= p.fim:
        noturno = instante.hour >= 22 or instante.hour < 7
        if (noturno or acamado) and posicao not in decubitos:
            posicao, desde = decubitos[0], instante
        if noturno or acamado:
            atividade = gerador.uniform(0.02, 0.08)
            if instante - desde >= timedelta(hours=2):
                posicao = decubitos[(decubitos.index(posicao) + 1) % 3] if posicao in decubitos else decubitos[0]
                desde = instante
        else:
            atividade = gerador.uniform(0.15, 0.45)
            if instante - desde >= timedelta(minutes=gerador.choice([45, 60, 90])):
                opcoes = ["sentado", "fora_do_leito", "sentado", *decubitos]
                posicao = gerador.choice([o for o in opcoes if o != posicao])
                desde = instante

        if imobilidade and imobilidade[0] <= instante <= imobilidade[1]:
            posicao, atividade = "decubito_dorsal", gerador.uniform(0.0, 0.03)
        if noite and noite <= instante < noite + timedelta(hours=2):
            atividade = gerador.uniform(0.5, 0.8)
            if noite + timedelta(minutes=105) <= instante < noite + timedelta(hours=2):
                posicao = "fora_do_leito"
        registros.append({"prontuario": p.prontuario, "registrado_em": _iso(instante),
                          "atividade": round(atividade, 3), "posicao": posicao,
                          "fonte": "sensor_leito"})
        instante += timedelta(minutes=15)
    return registros


def _rotulos(pacientes: list[_Paciente]) -> list[dict[str, Any]]:
    rotulos = []
    por_id = {p.prontuario: p for p in pacientes}

    def rotulo(pid, modalidade, evento, inicio, fim, tipos):
        rotulos.append({"prontuario": pid, "modalidade": modalidade, "evento": evento,
                        "inicio": _iso(inicio), "fim": _iso(fim), "tipos_esperados": tipos})

    vitais = ["news2_elevado", "parametro_critico", "desvio_basal", "tendencia",
              "padrao_multivariado"]
    for p in pacientes:
        for evento, inicio, fim, _ in _cenarios_vitais(p):
            if BASES.get(p.prontuario) and p.fim - p.inicio >= timedelta(hours=HORAS_MINIMAS):
                rotulo(p.prontuario, "sinais_vitais", evento, inicio, fim + timedelta(hours=1), vitais)

    def instante_de(pid, horas):
        return por_id[pid].h(horas)

    if "559120" in por_id:
        t = instante_de("559120", 30)
        rotulo("559120", "prescricao", "alergia_penicilina", t, t, ["conflito_alergia"])
    if "443902" in por_id:
        t = instante_de("443902", 36)
        rotulo("443902", "prescricao", "dose_10x", t, t, ["salto_dose", "acima_referencia"])
    if "208731" in por_id:
        t = instante_de("208731", 26)
        rotulo("208731", "prescricao", "insulina_10x", t, t, ["salto_dose"])
        rotulo("208731", "movimentacao", "imobilidade", instante_de("208731", 14),
               instante_de("208731", 8), ["imobilidade_prolongada"])
    if "117256" in por_id:
        t = instante_de("117256", 20)
        rotulo("117256", "prescricao", "duplo_anticoagulante", t, t, ["duplicidade_terapeutica"])
    if "310445" in por_id:
        t = instante_de("310445", 20)
        rotulo("310445", "prescricao", "suspensao_antimicrobiano", t, t, ["suspensao_sem_substituto"])
    if "624190" in por_id:
        t = instante_de("624190", 5)
        rotulo("624190", "prescricao", "alergia_morfina", t, t, ["conflito_alergia"])
    if "662480" in por_id:
        t = instante_de("662480", 3)
        rotulo("662480", "prescricao", "suspensao_insulina", t, t, ["suspensao_sem_substituto"])
    if "990117" in por_id:
        t = max(datetime.fromisoformat(por_id["990117"].dados["data_admissao"]), por_id["990117"].inicio)
        rotulo("990117", "prescricao", "alergia_dipirona", t, t, ["conflito_alergia"])
    if "875431" in por_id and por_id["875431"].fim - por_id["875431"].inicio >= timedelta(hours=24):
        noite = _noite_recente(por_id["875431"])
        rotulo("875431", "movimentacao", "agitacao", noite, noite + timedelta(hours=2),
               ["agitacao_noturna"])
        rotulo("875431", "movimentacao", "saida_noturna", noite + timedelta(minutes=105),
               noite + timedelta(hours=2), ["saida_leito_noturna"])
    return rotulos


def gerar(prontuarios: list[dict[str, Any]], semente: int = 42) -> dict[str, Any]:
    pacientes = _pacientes(prontuarios)
    sinais, prescricoes, movimentacao = [], [], []
    for p in pacientes:
        gerador = random.Random(f"{semente}-{p.prontuario}")
        sinais.extend(_sinais_vitais(p, gerador))
        prescricoes.extend(_prescricoes(p))
        movimentacao.extend(_movimentacao(p, gerador))
    return {
        "sinais_vitais_horarios": sinais,
        "prescricoes": prescricoes,
        "movimentacao": movimentacao,
        "rotulos": _rotulos(pacientes),
        "risco_queda": sorted(p.prontuario for p in pacientes if p.idade >= 65),
    }


def salvar(dados: dict[str, Any], pasta: Path | None = None) -> Path:
    pasta = Path(pasta or PASTA_PADRAO)
    pasta.mkdir(parents=True, exist_ok=True)
    for nome in ("sinais_vitais_horarios", "prescricoes", "movimentacao"):
        with (pasta / f"{nome}.jsonl").open("w", encoding="utf-8") as arquivo:
            for item in dados[nome]:
                arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")
    (pasta / "rotulos.json").write_text(
        json.dumps({"risco_queda": dados["risco_queda"], "eventos": dados["rotulos"]},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pasta


def carregar(pasta: Path | None = None) -> dict[str, Any]:
    pasta = Path(pasta or PASTA_PADRAO)

    def jsonl(nome: str) -> list[dict[str, Any]]:
        caminho = pasta / f"{nome}.jsonl"
        if not caminho.exists():
            return []
        return [json.loads(linha) for linha in caminho.read_text(encoding="utf-8").splitlines()
                if linha.strip()]

    rotulos_arquivo = pasta / "rotulos.json"
    rotulos = (json.loads(rotulos_arquivo.read_text(encoding="utf-8"))
               if rotulos_arquivo.exists() else {"risco_queda": [], "eventos": []})
    return {
        "sinais_vitais_horarios": jsonl("sinais_vitais_horarios"),
        "prescricoes": jsonl("prescricoes"),
        "movimentacao": jsonl("movimentacao"),
        "rotulos": rotulos["eventos"],
        "risco_queda": rotulos["risco_queda"],
    }


def carregar_no_banco(banco, dados: dict[str, Any]) -> dict[str, int]:
    """Grava as series de internacao nas tabelas do hospital."""
    return {
        "sinais_vitais": banco.registrar_sinais_vitais(dados["sinais_vitais_horarios"]),
        "prescricoes": banco.registrar_prescricoes(dados["prescricoes"]),
        "movimentacao": banco.registrar_movimentacao(dados["movimentacao"]),
    }


def preparar_monitoramento(banco, forcar: bool = False,
                           pasta: Path | None = None) -> dict[str, Any]:
    """Garante as series sinteticas em disco e no banco, sem duplicar registros."""
    from ...dados.banco import carregar_prontuarios_brutos

    pasta = Path(pasta or PASTA_PADRAO)
    if forcar or not (pasta / "rotulos.json").exists():
        salvar(gerar(carregar_prontuarios_brutos()), pasta)
    dados = carregar(pasta)
    ja_carregado = banco.consultar_somente_leitura(
        "SELECT count(*) AS total FROM prescricoes")[0]["total"] > 0
    if forcar or not ja_carregado:
        if ja_carregado:
            banco.popular(carregar_prontuarios_brutos())
        dados["carregados"] = carregar_no_banco(banco, dados)
    return dados
