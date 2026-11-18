"""Deteccao online de anomalias em series temporais de sinais vitais.

Quatro camadas, cada uma cobrindo uma falha das outras:

``limiar``       NEWS2: valores absolutos perigosos para qualquer paciente.
``basal``        z-score robusto (mediana/MAD) contra a linha de base do
                 *proprio* paciente: pega a mudanca brusca de quem ainda esta
                 dentro da faixa normal (FC 70 -> 105).
``tendencia``    inclinacao linear numa janela: piora gradual, antes de cruzar
                 qualquer limiar. E o sinal precoce de risco.
``multivariado`` Isolation Forest treinado na linha de base do paciente: pega
                 combinacoes atipicas em que nenhum parametro isolado chama
                 atencao (FC um pouco alta + SpO2 um pouco baixa + FR alta).

Antes de tudo, a qualidade do sinal: valor zero, ausente ou fisiologicamente
impossivel e tratado como falha de sensor, nunca entra nas estatisticas e nunca
vira alarme clinico. A linha de base congela enquanto ha desvio ativo naquele
parametro, para nao "aprender" a anomalia e deixar de ve-la.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..tempo_real import Leitura
from ..tipos import Anomalia
from . import news2
from .episodios import GerenciadorEpisodios

VARIAVEIS = ("fc", "fr", "saturacao", "pas", "pad", "temperatura")

NOMES = {
    "fc": "frequencia cardiaca", "fr": "frequencia respiratoria", "saturacao": "SpO2",
    "pas": "pressao sistolica", "pad": "pressao diastolica", "temperatura": "temperatura",
    "glasgow": "Glasgow",
}
UNIDADES = {"fc": "bpm", "fr": "irpm", "saturacao": "%", "pas": "mmHg", "pad": "mmHg",
            "temperatura": "C", "glasgow": ""}

FAIXA_FISIOLOGICA = {"fc": (20, 250), "fr": (3, 70), "saturacao": (50, 100),
                     "pas": (40, 280), "pad": (20, 180), "temperatura": (30.0, 43.5)}

ESCALA_MINIMA = {"fc": 3.0, "fr": 1.5, "saturacao": 1.0, "pas": 5.0, "pad": 4.0,
                 "temperatura": 0.2}

SO_QUEDA_E_RUIM = {"saturacao"}

PIORA_TENDENCIA = {"fc": 20.0, "fr": 6.0, "saturacao": -4.0, "pas": -20.0, "temperatura": 1.0}

CAMADAS = frozenset({"limiar", "basal", "tendencia", "multivariado"})


@dataclass(frozen=True)
class ParametrosSinaisVitais:
    janela_consolidacao: timedelta
    janela_base: timedelta
    min_amostras_base: int
    janela_tendencia: timedelta
    min_amostras_tendencia: int
    persistencia_limiar: timedelta
    persistencia_desvio: timedelta
    persistencia_falha: timedelta
    falha_prolongada: timedelta
    min_amostras_multivariado: int
    passo_multivariado: timedelta
    tolerancia_normal: int
    limiar_z: float = 4.0
    limiar_z_alto: float = 6.0
    margem_multivariado: float = 0.05
    persistencia_multivariado: timedelta = timedelta(seconds=45)
    desvio_minimo_multivariado: float = 2.0
    camadas: frozenset[str] = CAMADAS
    semente: int = 42

    @classmethod
    def alta_frequencia(cls, **ajustes: Any) -> ParametrosSinaisVitais:
        """Monitor de beira de leito, uma leitura por segundo."""
        base = cls(
            janela_consolidacao=timedelta(seconds=15),
            janela_base=timedelta(minutes=10),
            min_amostras_base=120,
            janela_tendencia=timedelta(minutes=3),
            min_amostras_tendencia=90,
            persistencia_limiar=timedelta(seconds=30),
            persistencia_desvio=timedelta(seconds=20),
            persistencia_falha=timedelta(seconds=30),
            falha_prolongada=timedelta(minutes=2),
            min_amostras_multivariado=120,
            passo_multivariado=timedelta(seconds=5),
            tolerancia_normal=30,
        )
        return replace(base, **ajustes)

    @classmethod
    def horario(cls, **ajustes: Any) -> ParametrosSinaisVitais:
        """Aferição de enfermagem na enfermaria, uma leitura por hora."""
        base = cls(
            janela_consolidacao=timedelta(0),
            janela_base=timedelta(hours=24),
            min_amostras_base=6,
            janela_tendencia=timedelta(hours=6),
            min_amostras_tendencia=4,
            persistencia_limiar=timedelta(0),
            persistencia_desvio=timedelta(0),
            persistencia_falha=timedelta(hours=1),
            falha_prolongada=timedelta(hours=4),
            min_amostras_multivariado=12,
            passo_multivariado=timedelta(0),
            tolerancia_normal=2,
        )
        return replace(base, **ajustes)


@dataclass
class _ModeloMultivariado:
    variaveis: tuple[str, ...]
    centro: np.ndarray
    escala: np.ndarray
    floresta: Any
    limiar: float
    ajustado_em: datetime


@dataclass
class _Estado:
    episodios: GerenciadorEpisodios
    recentes: dict[str, deque] = field(default_factory=dict)
    base: dict[str, deque] = field(default_factory=dict)
    tendencia: dict[str, deque] = field(default_factory=dict)
    vetores: deque = field(default_factory=lambda: deque(maxlen=2000))
    invalido_desde: dict[str, datetime] = field(default_factory=dict)
    modelo: _ModeloMultivariado | None = None
    ultimo_multivariado: datetime | None = None


def _podar(fila: deque, instante: datetime, janela: timedelta) -> None:
    while fila and instante - fila[0][0] > janela:
        fila.popleft()


def _valido(variavel: str, valor: Any) -> bool:
    if valor is None:
        return False
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return False
    if math.isnan(numero):
        return False
    minimo, maximo = FAIXA_FISIOLOGICA[variavel]
    return minimo <= numero <= maximo


def _iso(instante: datetime) -> str:
    return instante.isoformat(timespec="seconds")


def _fmt(variavel: str, valor: float) -> str:
    casas = 1 if variavel == "temperatura" else 0
    return f"{valor:.{casas}f} {UNIDADES[variavel]}".strip()


class DetectorSinaisVitais:
    nome = "sinais_vitais"
    modalidades = ("sinais_vitais",)

    def __init__(self, parametros: ParametrosSinaisVitais | None = None) -> None:
        self.parametros = parametros or ParametrosSinaisVitais.alta_frequencia()
        self._estados: dict[str, _Estado] = {}

    def _estado(self, prontuario: str) -> _Estado:
        if prontuario not in self._estados:
            self._estados[prontuario] = _Estado(
                episodios=GerenciadorEpisodios(self.parametros.tolerancia_normal))
        return self._estados[prontuario]

    def _anomalia(self, leitura: Leitura, inicio: datetime, tipo: str, severidade: str,
                  descricao: str, detector: str, escore: float,
                  evidencias: dict[str, Any]) -> Anomalia:
        return Anomalia(
            modalidade="sinais_vitais", tipo=tipo, severidade=severidade, descricao=descricao,
            prontuario=leitura.prontuario, inicio=_iso(inicio), fim=_iso(leitura.instante),
            escore=round(float(escore), 3), detector=detector, evidencias=evidencias,
        )

    def atualizar(self, leitura: Leitura) -> list[Anomalia]:
        p = self.parametros
        estado = self._estado(leitura.prontuario)
        instante = leitura.instante
        achados: list[Anomalia] = []

        validos: dict[str, float] = {}
        for variavel in VARIAVEIS:
            if variavel not in leitura.valores:
                continue
            valor = leitura.valores[variavel]
            if _valido(variavel, valor):
                validos[variavel] = float(valor)
                estado.invalido_desde.pop(variavel, None)
                estado.episodios.negativo(("falha", variavel))
                estado.recentes.setdefault(variavel, deque()).append((instante, float(valor)))
            else:
                achados.extend(self._qualidade(leitura, estado, variavel))

        consolidados: dict[str, float] = {}
        for variavel, fila in estado.recentes.items():
            _podar(fila, instante, p.janela_consolidacao)
            if fila:
                consolidados[variavel] = float(np.median([v for _, v in fila]))
        if not consolidados:
            return achados

        if "limiar" in p.camadas:
            valores_news = dict(consolidados)
            if _valido_glasgow(leitura.valores.get("glasgow")):
                valores_news["glasgow"] = float(leitura.valores["glasgow"])
            achados.extend(self._limiar(leitura, estado, valores_news))
        if "basal" in p.camadas:
            achados.extend(self._basal(leitura, estado, validos, consolidados))
        if "tendencia" in p.camadas:
            achados.extend(self._tendencia(leitura, estado, consolidados))
        if "multivariado" in p.camadas:
            achados.extend(self._multivariado(leitura, estado, validos, consolidados))
        return achados

    def _qualidade(self, leitura: Leitura, estado: _Estado, variavel: str) -> list[Anomalia]:
        p = self.parametros
        desde = estado.invalido_desde.setdefault(variavel, leitura.instante)
        duracao = leitura.instante - desde
        if duracao < p.persistencia_falha:
            return []
        severidade = "moderada" if duracao >= p.falha_prolongada else "baixa"
        inicio = estado.episodios.positivo(("falha", variavel), leitura.instante, severidade)
        if inicio is None:
            return []
        segundos = int(duracao.total_seconds())
        return [self._anomalia(
            leitura, inicio, f"falha_sensor_{variavel}", severidade,
            f"Leitura de {NOMES[variavel]} invalida ha {segundos // 60} min {segundos % 60} s "
            "(possivel sensor desconectado ou mal posicionado)",
            "qualidade_sinal", 1.0, {"variavel": variavel, "segundos": segundos,
                                     "valor_recebido": leitura.valores.get(variavel)},
        )]

    def _limiar(self, leitura: Leitura, estado: _Estado,
                valores: dict[str, float]) -> list[Anomalia]:
        p = self.parametros
        escore = news2.calcular(valores)
        achados = []
        for parametro, pontos in escore["subescores"].items():
            chave = ("parametro", parametro)
            if pontos < 3:
                estado.episodios.negativo(chave)
                continue
            inicio = estado.episodios.positivo(chave, leitura.instante, "alta",
                                               p.persistencia_limiar)
            if inicio is not None:
                valor = valores[parametro]
                achados.append(self._anomalia(
                    leitura, inicio, f"parametro_critico_{parametro}", "alta",
                    f"{NOMES[parametro]} em {_fmt(parametro, valor)}: faixa de pontuacao "
                    "maxima do NEWS2 neste parametro",
                    "news2", 1.0, {"parametro": parametro, "valor": valor, "subescore": pontos},
                ))

        if escore["total"] >= 5:
            severidade = "critica" if escore["total"] >= 7 else "alta"
            inicio = estado.episodios.positivo(("news2",), leitura.instante, severidade,
                                               p.persistencia_limiar)
            if inicio is not None:
                parcial = (f" (parcial: sem {', '.join(escore['parametros_ausentes'])})"
                           if escore["parcial"] else "")
                achados.append(self._anomalia(
                    leitura, inicio, "news2_elevado", severidade,
                    f"NEWS2 {escore['total']}{parcial}: risco {escore['faixa']} de deterioracao",
                    "news2", min(escore["total"] / 10, 1.0), escore,
                ))
        else:
            estado.episodios.negativo(("news2",))
        return achados

    def _basal(self, leitura: Leitura, estado: _Estado, validos: dict[str, float],
               consolidados: dict[str, float]) -> list[Anomalia]:
        p = self.parametros
        achados = []
        for variavel, valor in consolidados.items():
            chave = ("desvio", variavel)
            fila = estado.base.setdefault(variavel, deque())
            _podar(fila, leitura.instante, p.janela_base)
            if len(fila) >= p.min_amostras_base:
                historico = np.fromiter((v for _, v in fila), dtype=float)
                mediana = float(np.median(historico))
                mad = float(np.median(np.abs(historico - mediana)))
                escala = max(1.4826 * mad, ESCALA_MINIMA[variavel])
                z = (valor - mediana) / escala
                anormal = abs(z) >= p.limiar_z and not (variavel in SO_QUEDA_E_RUIM and z > 0)
                if anormal:
                    severidade = "alta" if abs(z) >= p.limiar_z_alto else "moderada"
                    inicio = estado.episodios.positivo(chave, leitura.instante, severidade,
                                                       p.persistencia_desvio)
                    if inicio is not None:
                        sentido = "subiu" if z > 0 else "caiu"
                        achados.append(self._anomalia(
                            leitura, inicio, f"desvio_basal_{variavel}", severidade,
                            f"{NOMES[variavel]} {sentido} para {_fmt(variavel, valor)}, fora da "
                            f"linha de base deste paciente ({_fmt(variavel, mediana)}; "
                            f"z = {z:+.1f})",
                            "zscore_robusto", min(abs(z) / 10, 1.0),
                            {"variavel": variavel, "valor": valor, "mediana_base": mediana,
                             "escala": round(escala, 3), "z": round(z, 2),
                             "amostras_base": len(fila)},
                        ))
                else:
                    estado.episodios.negativo(chave)
            if variavel in validos and not estado.episodios.ativo(chave):
                fila.append((leitura.instante, validos[variavel]))
        return achados

    def _tendencia(self, leitura: Leitura, estado: _Estado,
                   consolidados: dict[str, float]) -> list[Anomalia]:
        p = self.parametros
        achados = []
        for variavel, piora in PIORA_TENDENCIA.items():
            if variavel not in consolidados:
                continue
            chave = ("tendencia", variavel)
            fila = estado.tendencia.setdefault(variavel, deque())
            fila.append((leitura.instante, consolidados[variavel]))
            _podar(fila, leitura.instante, p.janela_tendencia)
            duracao = (fila[-1][0] - fila[0][0]).total_seconds() if fila else 0.0
            if len(fila) < p.min_amostras_tendencia or duracao < 0.6 * p.janela_tendencia.total_seconds():
                continue
            tempos = np.fromiter(((t - fila[0][0]).total_seconds() for t, _ in fila), dtype=float)
            valores = np.fromiter((v for _, v in fila), dtype=float)
            inclinacao = float(np.polyfit(tempos, valores, 1)[0])
            variacao = inclinacao * duracao
            terco = max(len(valores) // 3, 1)
            confirmacao = float(np.median(valores[-terco:]) - np.median(valores[:terco]))
            no_sentido = math.copysign(1.0, piora)
            if variacao * no_sentido >= abs(piora) and confirmacao * no_sentido >= abs(piora) / 2:
                inicio = estado.episodios.positivo(chave, leitura.instante, "moderada")
                if inicio is not None:
                    minutos = int(round(duracao / 60))
                    janela = f"{minutos} min" if minutos < 120 else f"{minutos // 60} h"
                    achados.append(self._anomalia(
                        leitura, inicio, f"tendencia_{variavel}", "moderada",
                        f"Tendencia de piora de {NOMES[variavel]}: {variacao:+.1f} "
                        f"{UNIDADES[variavel]} em {janela}",
                        "tendencia_linear", min(abs(variacao) / (2 * abs(piora)), 1.0),
                        {"variavel": variavel, "variacao": round(variacao, 2),
                         "inclinacao_por_hora": round(inclinacao * 3600, 2),
                         "janela_s": int(duracao), "amostras": len(fila)},
                    ))
            else:
                estado.episodios.negativo(chave)
        return achados

    def _multivariado(self, leitura: Leitura, estado: _Estado, validos: dict[str, float],
                      consolidados: dict[str, float]) -> list[Anomalia]:
        p = self.parametros
        chave = ("multivariado",)
        presentes = tuple(v for v in VARIAVEIS if v in validos)
        algum_ativo = any(estado.episodios.ativo(("desvio", v)) for v in presentes)
        if len(presentes) >= 2 and not algum_ativo and not estado.episodios.ativo(chave):
            estado.vetores.append((leitura.instante, presentes,
                                   np.array([validos[v] for v in presentes])))

        modelo = estado.modelo
        if modelo is None or (leitura.instante - modelo.ajustado_em > p.janela_base
                              and not estado.episodios.ativo(chave)):
            modelo = self._ajustar(estado, leitura.instante) or modelo
        if modelo is None:
            return []
        if (estado.ultimo_multivariado is not None
                and leitura.instante - estado.ultimo_multivariado < p.passo_multivariado):
            return []
        if not all(v in consolidados for v in modelo.variaveis):
            return []
        estado.ultimo_multivariado = leitura.instante

        vetor = np.array([consolidados[v] for v in modelo.variaveis])
        padronizado = (vetor - modelo.centro) / modelo.escala
        pontuacao = float(modelo.floresta.score_samples(padronizado.reshape(1, -1))[0])
        if (pontuacao >= modelo.limiar
                or float(np.max(np.abs(padronizado))) < p.desvio_minimo_multivariado):
            estado.episodios.negativo(chave)
            return []
        inicio = estado.episodios.positivo(chave, leitura.instante, "moderada",
                                           p.persistencia_multivariado)
        if inicio is None:
            return []
        ordem = np.argsort(-np.abs(padronizado))
        principais = [f"{NOMES[modelo.variaveis[i]]} ({padronizado[i]:+.1f} desvios)"
                      for i in ordem[:2]]
        return [self._anomalia(
            leitura, inicio, "padrao_multivariado", "moderada",
            "Combinacao de sinais vitais atipica para este paciente (Isolation Forest); "
            f"maiores contribuicoes: {', '.join(principais)}",
            "isolation_forest", min((modelo.limiar - pontuacao) * 5, 1.0),
            {"pontuacao": round(pontuacao, 4), "limiar": round(modelo.limiar, 4),
             "desvios_padronizados": {modelo.variaveis[i]: round(float(padronizado[i]), 2)
                                      for i in range(len(modelo.variaveis))}},
        )]

    def _ajustar(self, estado: _Estado, instante: datetime) -> _ModeloMultivariado | None:
        from sklearn.ensemble import IsolationForest

        p = self.parametros
        recentes = [(t, nomes, v) for t, nomes, v in estado.vetores
                    if instante - t <= p.janela_base]
        if len(recentes) < p.min_amostras_multivariado:
            return None
        variaveis = recentes[-1][1]
        matriz = np.array([v for _, nomes, v in recentes if nomes == variaveis])
        if len(matriz) < p.min_amostras_multivariado:
            return None
        centro = np.median(matriz, axis=0)
        mad = np.median(np.abs(matriz - centro), axis=0)
        escala = np.maximum(1.4826 * mad, [ESCALA_MINIMA[v] for v in variaveis])
        padronizada = (matriz - centro) / escala
        gerador = np.random.default_rng(p.semente)
        treino = np.vstack([padronizada + gerador.normal(0, 0.5, padronizada.shape)
                            for _ in range(3)])
        floresta = IsolationForest(n_estimators=100, random_state=p.semente).fit(treino)
        limiar = float(np.quantile(floresta.score_samples(padronizada), 0.005)) - p.margem_multivariado
        estado.modelo = _ModeloMultivariado(variaveis, centro, escala, floresta, limiar, instante)
        return estado.modelo


def _valido_glasgow(valor: Any) -> bool:
    try:
        return 3 <= float(valor) <= 15
    except (TypeError, ValueError):
        return False
