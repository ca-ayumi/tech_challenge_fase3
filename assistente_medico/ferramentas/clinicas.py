"""Ferramentas LangChain do assistente.

As ferramentas sao a unica porta pela qual o assistente toca dados do hospital.
Isso e deliberado: cada acesso fica nomeado, tipado e passivel de registro na
auditoria, em vez de o modelo receber um despejo de dados no prompt.

Elas se dividem em dois grupos:

- **leitura** (consultar prontuario, exames, protocolos, alertas): sem efeito
  colateral, liberadas para qualquer perfil autorizado;
- **escrita** (registrar alerta, abrir pedido de validacao): tem efeito no
  sistema e por isso registram autor, horario e origem.

Nenhuma ferramenta prescreve, calcula dose ou autoriza alta — essas acoes nao
existem como capacidade do sistema, e nao apenas como instrucao no prompt.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from ..dados.banco import BancoHospital
from ..dados.contexto_paciente import montar_contexto
from ..rag.recuperador import RecuperadorProtocolos
from ..seguranca.regras_clinicas import avaliar


def _json(dado: Any) -> str:
    return json.dumps(dado, ensure_ascii=False, indent=2, default=str)


def criar_ferramentas(
    banco: BancoHospital,
    recuperador: RecuperadorProtocolos,
    interacao: str | None = None,
    usuario: str | None = None,
) -> list[StructuredTool]:
    """Cria as ferramentas ligadas a uma instancia de banco e de recuperador."""

    def buscar_paciente(termo: str) -> str:
        """Localiza um paciente por numero de prontuario, leito ou parte do nome.

        Use antes de qualquer consulta clinica, para obter o numero de prontuario.
        Devolve a lista de pacientes que casam com o termo.
        """
        encontrados = banco.buscar_paciente(termo)
        if not encontrados:
            return f"Nenhum paciente encontrado para '{termo}'."
        return _json(encontrados)

    def consultar_prontuario(prontuario: str) -> str:
        """Devolve a situacao clinica consolidada de um paciente.

        Inclui idade, unidade, leito, motivo da internacao, comorbidades, alergias,
        medicacoes em uso, protocolos ativos, sinais vitais recentes e exames
        liberados e pendentes. Nao inclui identificadores diretos (nome, CPF,
        telefone, endereco), por minimizacao de dados.
        """
        contexto = montar_contexto(banco, prontuario)
        if contexto is None:
            return f"Prontuario {prontuario} nao encontrado."
        return contexto.como_texto()

    def listar_exames_pendentes(prontuario: str) -> str:
        """Lista os exames ainda pendentes de um paciente, com o tempo de espera.

        Marca quais sao criticos segundo o cadastro do exame.
        """
        contexto = montar_contexto(banco, prontuario)
        if contexto is None:
            return f"Prontuario {prontuario} nao encontrado."
        if not contexto.exames_pendentes:
            return f"Nenhum exame pendente para o paciente do leito {contexto.leito}."
        return _json([
            {
                "exame": exame["nome"],
                "codigo": exame["codigo"],
                "critico": bool(exame.get("critico")),
                "solicitado_em": exame.get("solicitado_em"),
            }
            for exame in contexto.exames_pendentes
        ])

    def verificar_pendencias_de_protocolo(prontuario: str) -> str:
        """Aplica as regras institucionais de vigilancia e devolve os alertas do paciente.

        Cada alerta traz prioridade, categoria, mensagem e a secao do protocolo que
        o justifica. As regras sao deterministicas, calculadas sobre os dados
        estruturados, e nao dependem do modelo de linguagem.
        """
        contexto = montar_contexto(banco, prontuario)
        if contexto is None:
            return f"Prontuario {prontuario} nao encontrado."
        alertas = avaliar(contexto)
        if not alertas:
            return (f"Nenhuma pendencia identificada para o paciente do leito "
                    f"{contexto.leito} pelas regras vigentes.")
        return _json([alerta.como_dicionario() for alerta in alertas])

    def consultar_protocolo(consulta: str) -> str:
        """Recupera as secoes de protocolo institucional relevantes para uma pergunta.

        Devolve os trechos com seu identificador citavel (por exemplo
        'PROT-SEP-001 §4'), que deve ser usado no bloco Fontes da resposta.
        """
        recuperados = recuperador.recuperar(consulta)
        if not recuperados:
            return "Nenhum trecho de protocolo institucional recuperado para esta consulta."
        return "\n\n".join(item.trecho.como_contexto() for item in recuperados)

    def consultar_base_estruturada(sql: str) -> str:
        """Executa uma consulta SELECT somente-leitura na base assistencial.

        Tabelas disponiveis: pacientes, comorbidades, alergias, medicacoes,
        sinais_vitais, exames, protocolos_ativos, eventos, evolucoes, alertas.
        Comandos de escrita e consultas as tabelas de governanca sao recusados.
        Exemplo: SELECT leito, motivo_internacao FROM pacientes WHERE unidade LIKE '%Intensiva%'
        """
        try:
            linhas = banco.consultar_somente_leitura(sql)
        except ValueError as erro:
            return f"Consulta recusada: {erro}"
        if not linhas:
            return "A consulta nao retornou linhas."
        return _json(linhas)

    def registrar_alerta_equipe(prontuario: str, prioridade: str, mensagem: str,
                                fonte: str = "") -> str:
        """Registra um alerta para a equipe assistencial no painel institucional.

        Prioridade deve ser uma de: maxima, alta, media, informativa. A fonte deve
        ser a secao do protocolo que justifica o alerta.
        """
        if prioridade not in {"maxima", "alta", "media", "informativa"}:
            return ("Prioridade invalida. Use: maxima, alta, media ou informativa.")
        identificador = banco.registrar_alerta(
            prioridade=prioridade, categoria="assistente", mensagem=mensagem,
            prontuario=prontuario, fonte=fonte or None, interacao=interacao,
        )
        return f"Alerta {identificador} registrado com prioridade {prioridade}."

    def solicitar_validacao_humana(sugestao: str, prontuario: str = "",
                                   fontes: str = "") -> str:
        """Abre um pedido de validacao humana para uma sugestao de conduta.

        Toda sugestao gerada pelo assistente entra como pendente e so tem efeito
        apos aceite, recusa ou modificacao registrada por profissional habilitado
        (PROT-GOV-010 §5).
        """
        lista_fontes = [f.strip() for f in fontes.split(";") if f.strip()] if fontes else []
        identificador = banco.registrar_validacao(
            interacao=interacao or "sem-interacao", sugestao=sugestao,
            prontuario=prontuario or None, fontes=lista_fontes,
        )
        return (f"Pedido de validacao {identificador} aberto e pendente de decisao "
                "de profissional habilitado.")

    definicoes = [
        (buscar_paciente, "leitura"),
        (consultar_prontuario, "leitura"),
        (listar_exames_pendentes, "leitura"),
        (verificar_pendencias_de_protocolo, "leitura"),
        (consultar_protocolo, "leitura"),
        (consultar_base_estruturada, "leitura"),
        (registrar_alerta_equipe, "escrita"),
        (solicitar_validacao_humana, "escrita"),
    ]

    ferramentas: list[StructuredTool] = []
    for funcao, tipo in definicoes:
        ferramenta = StructuredTool.from_function(
            func=funcao,
            name=funcao.__name__,
            description=(funcao.__doc__ or "").strip(),
        )
        ferramenta.metadata = {"tipo_acesso": tipo, "usuario": usuario}
        ferramentas.append(ferramenta)
    return ferramentas


def ferramentas_de_leitura(ferramentas: list[StructuredTool]) -> list[StructuredTool]:
    return [f for f in ferramentas if (f.metadata or {}).get("tipo_acesso") == "leitura"]
