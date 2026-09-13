"""Explicabilidade das respostas.

O requisito institucional (PROT-GOV-010 §7) e que toda afirmacao relevante
indique sua origem. Este modulo monta esse rastro e faz uma verificacao que o
rotulo "Fontes" sozinho nao faz: confere se cada referencia citada pelo modelo
realmente estava entre os documentos recuperados.

Uma citacao que aparece na resposta mas nao foi recuperada e uma citacao
**nao fundamentada** — o caso classico de um modelo de linguagem inventar um
numero de protocolo plausivel. Ela e sinalizada na interface e registrada na
auditoria em vez de passar despercebida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dados.preprocessamento import Trecho
from .finetuning.formato import referencias_citadas, separar_secoes
from .rag.recuperador import TrechoRecuperado


@dataclass
class FonteUtilizada:
    """Uma fonte que sustenta a resposta."""

    referencia: str
    titulo: str
    tipo: str          # protocolo | prontuario | regra
    detalhe: str = ""
    score: float | None = None
    versao: str = ""

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "referencia": self.referencia,
            "titulo": self.titulo,
            "tipo": self.tipo,
            "detalhe": self.detalhe,
            "score": self.score,
            "versao": self.versao,
        }


@dataclass
class Explicacao:
    """Rastro completo do que sustentou a resposta."""

    fontes: list[FonteUtilizada] = field(default_factory=list)
    campos_prontuario: list[str] = field(default_factory=list)
    regras_aplicadas: list[str] = field(default_factory=list)
    citadas: list[str] = field(default_factory=list)
    recuperadas: list[str] = field(default_factory=list)
    nao_fundamentadas: list[str] = field(default_factory=list)

    @property
    def cobertura(self) -> float:
        """Fracao das referencias citadas que estavam entre as recuperadas."""
        if not self.citadas:
            return 1.0
        fundamentadas = len(self.citadas) - len(self.nao_fundamentadas)
        return round(fundamentadas / len(self.citadas), 3)

    @property
    def tem_alerta_de_fonte(self) -> bool:
        return bool(self.nao_fundamentadas) or not self.fontes

    def como_dicionario(self) -> dict[str, Any]:
        return {
            "fontes": [f.como_dicionario() for f in self.fontes],
            "campos_prontuario": self.campos_prontuario,
            "regras_aplicadas": self.regras_aplicadas,
            "referencias_citadas": self.citadas,
            "referencias_recuperadas": self.recuperadas,
            "citacoes_nao_fundamentadas": self.nao_fundamentadas,
            "cobertura_das_citacoes": self.cobertura,
        }

    def como_markdown(self) -> str:
        """Bloco legivel para a interface e para o relatorio."""
        linhas = ["#### Base da resposta"]
        if self.fontes:
            for fonte in self.fontes:
                sufixo = f" (relevancia {fonte.score:.2f})" if fonte.score is not None else ""
                versao = f" — versao {fonte.versao}" if fonte.versao else ""
                linhas.append(f"- **{fonte.referencia}** — {fonte.titulo}{versao}{sufixo}")
        else:
            linhas.append("- Nenhum documento institucional recuperado.")

        if self.campos_prontuario:
            linhas.append("")
            linhas.append("**Campos do prontuario consultados:** "
                          + ", ".join(f"`{campo}`" for campo in self.campos_prontuario))
        if self.regras_aplicadas:
            linhas.append("")
            linhas.append("**Regras institucionais aplicadas:** "
                          + ", ".join(self.regras_aplicadas))
        if self.nao_fundamentadas:
            linhas.append("")
            linhas.append("> **Atencao:** a resposta cita "
                          + ", ".join(f"`{r}`" for r in self.nao_fundamentadas)
                          + ", que nao esta entre os documentos recuperados. "
                            "Confira a citacao antes de usar.")
        linhas.append("")
        linhas.append(f"_Cobertura das citacoes: {self.cobertura:.0%}_")
        return "\n".join(linhas)


def _normalizar_referencia(referencia: str) -> str:
    return referencia.split("—")[0].split(" - ")[0].strip().upper().replace(" ", "")


def montar_explicacao(
    resposta: str,
    recuperados: list[TrechoRecuperado] | None = None,
    campos_prontuario: list[str] | None = None,
    regras_aplicadas: list[str] | None = None,
    trechos_extras: list[Trecho] | None = None,
) -> Explicacao:
    """Monta a explicacao comparando o que foi citado com o que foi recuperado."""
    recuperados = recuperados or []
    estruturada = separar_secoes(resposta)

    # Referencias citadas: as do bloco "Fontes" mais as que aparecem no corpo.
    citadas: list[str] = []
    for texto in [*estruturada.fontes, estruturada.corpo]:
        for referencia in referencias_citadas(texto):
            if referencia not in citadas:
                citadas.append(referencia)

    disponiveis: dict[str, FonteUtilizada] = {}
    for item in recuperados:
        chave = _normalizar_referencia(item.referencia)
        disponiveis[chave] = FonteUtilizada(
            referencia=item.trecho.referencia,
            titulo=f"{item.trecho.titulo_documento} — {item.trecho.titulo_secao}",
            tipo="protocolo", detalhe=item.motivo, score=round(item.score, 2),
            versao=item.trecho.versao,
        )
    for trecho in trechos_extras or []:
        chave = _normalizar_referencia(trecho.referencia)
        disponiveis.setdefault(chave, FonteUtilizada(
            referencia=trecho.referencia,
            titulo=f"{trecho.titulo_documento} — {trecho.titulo_secao}",
            tipo="protocolo", detalhe="citado por regra institucional",
            versao=trecho.versao,
        ))

    # Uma citacao ao documento inteiro (PROT-SEP-001) e considerada fundamentada
    # se qualquer secao daquele documento foi recuperada.
    documentos_recuperados = {chave.split("§")[0] for chave in disponiveis}

    fontes: list[FonteUtilizada] = []
    nao_fundamentadas: list[str] = []
    for referencia in citadas:
        chave = _normalizar_referencia(referencia)
        if chave in disponiveis:
            fonte = disponiveis[chave]
            if fonte not in fontes:
                fontes.append(fonte)
        elif chave.split("§")[0] in documentos_recuperados:
            fontes.append(FonteUtilizada(
                referencia=referencia, titulo="Secao do documento recuperado",
                tipo="protocolo", detalhe="documento recuperado, secao nao conferida",
            ))
        else:
            nao_fundamentadas.append(referencia)

    # Fontes recuperadas que o modelo nao chegou a citar continuam listadas:
    # o auditor precisa ver o que o sistema consultou, nao so o que citou.
    for fonte in disponiveis.values():
        if fonte not in fontes:
            fontes.append(fonte)

    if campos_prontuario:
        fontes.append(FonteUtilizada(
            referencia="Prontuario eletronico", tipo="prontuario",
            titulo="Registros estruturados do paciente",
            detalhe=", ".join(campos_prontuario),
        ))

    return Explicacao(
        fontes=fontes,
        campos_prontuario=campos_prontuario or [],
        regras_aplicadas=regras_aplicadas or [],
        citadas=citadas,
        recuperadas=[f.referencia for f in disponiveis.values()],
        nao_fundamentadas=nao_fundamentadas,
    )
