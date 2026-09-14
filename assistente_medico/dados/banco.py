"""Banco de dados estruturado do hospital (SQLite).

Simula o sistema de prontuario eletronico consultado pelo assistente em tempo
de execucao. Guarda tambem as tabelas de governanca: alertas emitidos,
validacoes humanas pendentes e a trilha de auditoria das interacoes.

O banco de execucao guarda os dados identificados, como no sistema real; a
anonimizacao se aplica ao corpus de treinamento e aos registros de auditoria
(PROT-GOV-010 §6 e §8).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import CAMINHOS

ESQUEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    prontuario          TEXT PRIMARY KEY,
    nome                TEXT NOT NULL,
    cpf                 TEXT,
    cns                 TEXT,
    rg                  TEXT,
    data_nascimento     TEXT,
    sexo                TEXT,
    telefone            TEXT,
    email               TEXT,
    endereco            TEXT,
    unidade             TEXT,
    leito               TEXT,
    data_admissao       TEXT,
    motivo_internacao   TEXT,
    medico_responsavel  TEXT
);

CREATE TABLE IF NOT EXISTS comorbidades (
    prontuario TEXT NOT NULL,
    descricao  TEXT NOT NULL,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS alergias (
    prontuario TEXT NOT NULL,
    agente     TEXT NOT NULL,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS medicacoes (
    prontuario TEXT NOT NULL,
    descricao  TEXT NOT NULL,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS sinais_vitais (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prontuario  TEXT NOT NULL,
    aferido_em  TEXT NOT NULL,
    pas         REAL, pad REAL, fc REAL, fr REAL,
    temperatura REAL, saturacao REAL, glasgow INTEGER,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS exames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prontuario    TEXT NOT NULL,
    codigo        TEXT NOT NULL,
    nome          TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('pendente', 'liberado', 'cancelado')),
    solicitado_em TEXT,
    liberado_em   TEXT,
    resultado     TEXT,
    unidade       TEXT,
    referencia    TEXT,
    critico       INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS protocolos_ativos (
    prontuario TEXT NOT NULL,
    protocolo  TEXT NOT NULL,
    aberto_em  TEXT,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS eventos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prontuario    TEXT NOT NULL,
    tipo          TEXT NOT NULL,
    ocorreu       INTEGER NOT NULL DEFAULT 0,
    registrado_em TEXT,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS evolucoes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    prontuario TEXT NOT NULL,
    data_hora  TEXT,
    autor      TEXT,
    texto      TEXT,
    FOREIGN KEY (prontuario) REFERENCES pacientes (prontuario)
);

CREATE TABLE IF NOT EXISTS alertas (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    interacao  TEXT,
    prontuario TEXT,
    prioridade TEXT NOT NULL,
    categoria  TEXT NOT NULL,
    mensagem   TEXT NOT NULL,
    fonte      TEXT,
    criado_em  TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'aberto'
);

CREATE TABLE IF NOT EXISTS validacoes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    interacao     TEXT NOT NULL,
    prontuario    TEXT,
    sugestao      TEXT NOT NULL,
    fontes        TEXT,
    status        TEXT NOT NULL DEFAULT 'pendente',
    profissional  TEXT,
    decidido_em   TEXT,
    justificativa TEXT,
    criado_em     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auditoria (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    interacao   TEXT NOT NULL,
    registrado_em TEXT NOT NULL,
    evento      TEXT NOT NULL,
    usuario     TEXT,
    perfil      TEXT,
    prontuario  TEXT,
    conteudo    TEXT
);

CREATE INDEX IF NOT EXISTS idx_exames_prontuario ON exames (prontuario, status);
CREATE INDEX IF NOT EXISTS idx_sinais_prontuario ON sinais_vitais (prontuario, aferido_em);
CREATE INDEX IF NOT EXISTS idx_auditoria_interacao ON auditoria (interacao);
CREATE INDEX IF NOT EXISTS idx_alertas_prontuario ON alertas (prontuario, status);
"""

TABELAS_CONSULTAVEIS = {
    "pacientes", "comorbidades", "alergias", "medicacoes", "sinais_vitais",
    "exames", "protocolos_ativos", "eventos", "evolucoes", "alertas",
}


class BancoHospital:
    """Acesso ao banco do hospital. Metodos de leitura sao usados pelas tools."""

    def __init__(self, caminho: Path | None = None) -> None:
        self.caminho = Path(caminho or CAMINHOS.banco)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def conectar(self) -> Iterator[sqlite3.Connection]:
        conexao = sqlite3.connect(self.caminho)
        conexao.row_factory = sqlite3.Row
        conexao.execute("PRAGMA foreign_keys = ON")
        try:
            yield conexao
            conexao.commit()
        finally:
            conexao.close()

    def criar_esquema(self) -> None:
        with self.conectar() as conexao:
            conexao.executescript(ESQUEMA)

    def existe(self) -> bool:
        if not self.caminho.exists():
            return False
        with self.conectar() as conexao:
            linha = conexao.execute(
                "SELECT count(*) AS total FROM sqlite_master WHERE type='table' AND name='pacientes'"
            ).fetchone()
            if not linha or not linha["total"]:
                return False
            return conexao.execute("SELECT count(*) AS total FROM pacientes").fetchone()["total"] > 0

    def popular(self, prontuarios: list[dict[str, Any]], limpar: bool = True) -> int:
        """Carrega os prontuarios sinteticos no banco."""
        self.criar_esquema()
        with self.conectar() as conexao:
            if limpar:
                for tabela in ("evolucoes", "eventos", "protocolos_ativos", "exames",
                               "sinais_vitais", "medicacoes", "alergias", "comorbidades",
                               "pacientes"):
                    conexao.execute(f"DELETE FROM {tabela}")

            for paciente in prontuarios:
                conexao.execute(
                    """INSERT OR REPLACE INTO pacientes
                       (prontuario, nome, cpf, cns, rg, data_nascimento, sexo, telefone,
                        email, endereco, unidade, leito, data_admissao, motivo_internacao,
                        medico_responsavel)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        paciente["prontuario"], paciente["nome"], paciente.get("cpf"),
                        paciente.get("cns"), paciente.get("rg"), paciente.get("data_nascimento"),
                        paciente.get("sexo"), paciente.get("telefone"), paciente.get("email"),
                        paciente.get("endereco"), paciente.get("unidade"), paciente.get("leito"),
                        paciente.get("data_admissao"), paciente.get("motivo_internacao"),
                        paciente.get("medico_responsavel"),
                    ),
                )
                prontuario = paciente["prontuario"]
                conexao.executemany(
                    "INSERT INTO comorbidades (prontuario, descricao) VALUES (?,?)",
                    [(prontuario, c) for c in paciente.get("comorbidades", [])],
                )
                conexao.executemany(
                    "INSERT INTO alergias (prontuario, agente) VALUES (?,?)",
                    [(prontuario, a) for a in paciente.get("alergias", [])],
                )
                conexao.executemany(
                    "INSERT INTO medicacoes (prontuario, descricao) VALUES (?,?)",
                    [(prontuario, m) for m in paciente.get("medicacoes_em_uso", [])],
                )
                conexao.executemany(
                    """INSERT INTO sinais_vitais
                       (prontuario, aferido_em, pas, pad, fc, fr, temperatura, saturacao, glasgow)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    [
                        (prontuario, s["aferido_em"], s.get("pas"), s.get("pad"), s.get("fc"),
                         s.get("fr"), s.get("temperatura"), s.get("saturacao"), s.get("glasgow"))
                        for s in paciente.get("sinais_vitais", [])
                    ],
                )
                conexao.executemany(
                    """INSERT INTO exames
                       (prontuario, codigo, nome, status, solicitado_em, liberado_em,
                        resultado, unidade, referencia, critico)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (prontuario, e["codigo"], e["nome"], e["status"], e.get("solicitado_em"),
                         e.get("liberado_em"), e.get("resultado"), e.get("unidade"),
                         e.get("referencia"), int(bool(e.get("critico"))))
                        for e in paciente.get("exames", [])
                    ],
                )
                conexao.executemany(
                    "INSERT INTO protocolos_ativos (prontuario, protocolo, aberto_em) VALUES (?,?,?)",
                    [
                        (prontuario, p["protocolo"], p.get("aberto_em"))
                        for p in paciente.get("protocolos_ativos", [])
                    ],
                )
                conexao.executemany(
                    "INSERT INTO eventos (prontuario, tipo, ocorreu, registrado_em) VALUES (?,?,?,?)",
                    [
                        (prontuario, ev["tipo"], int(bool(ev.get("ocorreu"))), ev.get("registrado_em"))
                        for ev in paciente.get("eventos", [])
                    ],
                )
                conexao.executemany(
                    "INSERT INTO evolucoes (prontuario, data_hora, autor, texto) VALUES (?,?,?,?)",
                    [
                        (prontuario, ev.get("data_hora"), ev.get("autor"), ev.get("texto"))
                        for ev in paciente.get("evolucoes", [])
                    ],
                )
        return len(prontuarios)

    def _consultar(self, sql: str, parametros: tuple = ()) -> list[dict[str, Any]]:
        with self.conectar() as conexao:
            return [dict(linha) for linha in conexao.execute(sql, parametros).fetchall()]

    def listar_pacientes(self) -> list[dict[str, Any]]:
        return self._consultar(
            "SELECT prontuario, nome, unidade, leito, motivo_internacao, data_admissao "
            "FROM pacientes ORDER BY unidade, leito"
        )

    def buscar_paciente(self, termo: str) -> list[dict[str, Any]]:
        """Localiza paciente por prontuario, leito ou parte do nome."""
        termo = (termo or "").strip()
        if not termo:
            return []
        return self._consultar(
            "SELECT prontuario, nome, unidade, leito FROM pacientes "
            "WHERE prontuario = ? OR leito = ? OR upper(leito) = upper(?) "
            "OR lower(nome) LIKE lower(?) ORDER BY nome",
            (termo, termo, termo, f"%{termo}%"),
        )

    def obter_paciente(self, prontuario: str) -> dict[str, Any] | None:
        linhas = self._consultar("SELECT * FROM pacientes WHERE prontuario = ?", (prontuario,))
        return linhas[0] if linhas else None

    def comorbidades(self, prontuario: str) -> list[str]:
        return [r["descricao"] for r in self._consultar(
            "SELECT descricao FROM comorbidades WHERE prontuario = ?", (prontuario,))]

    def alergias(self, prontuario: str) -> list[str]:
        return [r["agente"] for r in self._consultar(
            "SELECT agente FROM alergias WHERE prontuario = ?", (prontuario,))]

    def medicacoes(self, prontuario: str) -> list[str]:
        return [r["descricao"] for r in self._consultar(
            "SELECT descricao FROM medicacoes WHERE prontuario = ?", (prontuario,))]

    def sinais_vitais(self, prontuario: str, limite: int = 3) -> list[dict[str, Any]]:
        return self._consultar(
            "SELECT * FROM sinais_vitais WHERE prontuario = ? ORDER BY aferido_em DESC LIMIT ?",
            (prontuario, limite),
        )

    def exames(self, prontuario: str, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self._consultar(
                "SELECT * FROM exames WHERE prontuario = ? AND status = ? "
                "ORDER BY critico DESC, solicitado_em",
                (prontuario, status),
            )
        return self._consultar(
            "SELECT * FROM exames WHERE prontuario = ? ORDER BY critico DESC, solicitado_em",
            (prontuario,),
        )

    def exames_pendentes(self, prontuario: str) -> list[dict[str, Any]]:
        return self.exames(prontuario, status="pendente")

    def protocolos_ativos(self, prontuario: str) -> list[dict[str, Any]]:
        return self._consultar(
            "SELECT protocolo, aberto_em FROM protocolos_ativos WHERE prontuario = ?",
            (prontuario,),
        )

    def eventos(self, prontuario: str) -> list[dict[str, Any]]:
        return self._consultar("SELECT * FROM eventos WHERE prontuario = ?", (prontuario,))

    def evolucoes(self, prontuario: str, limite: int = 3) -> list[dict[str, Any]]:
        return self._consultar(
            "SELECT * FROM evolucoes WHERE prontuario = ? ORDER BY data_hora DESC LIMIT ?",
            (prontuario, limite),
        )

    def consultar_somente_leitura(self, sql: str, limite: int = 50) -> list[dict[str, Any]]:
        """Executa uma consulta SELECT restrita as tabelas assistenciais.

        Rejeita qualquer comando que nao seja um unico SELECT, que toque tabelas
        de governanca (auditoria, validacoes) ou que contenha ponto e virgula.
        """
        normalizado = " ".join(sql.strip().split())
        if ";" in normalizado.rstrip(";"):
            raise ValueError("Apenas uma instrucao por consulta.")
        normalizado = normalizado.rstrip(";")
        if not normalizado.lower().startswith("select"):
            raise ValueError("Somente consultas SELECT sao permitidas.")
        proibidos = ("insert", "update", "delete", "drop", "alter", "create", "attach", "pragma")
        if any(f" {palavra} " in f" {normalizado.lower()} " for palavra in proibidos):
            raise ValueError("Comando de escrita detectado e bloqueado.")
        tabelas_citadas = {
            palavra.strip("(),")
            for palavra in normalizado.lower().split()
        } & {t for t in TABELAS_CONSULTAVEIS | {"auditoria", "validacoes"}}
        if tabelas_citadas - TABELAS_CONSULTAVEIS:
            raise ValueError("Consulta a tabela de governanca nao permitida por esta ferramenta.")
        if " limit " not in normalizado.lower():
            normalizado = f"{normalizado} LIMIT {limite}"
        return self._consultar(normalizado)

    def registrar_alerta(self, prioridade: str, categoria: str, mensagem: str,
                         prontuario: str | None = None, fonte: str | None = None,
                         interacao: str | None = None) -> int:
        with self.conectar() as conexao:
            cursor = conexao.execute(
                """INSERT INTO alertas
                   (interacao, prontuario, prioridade, categoria, mensagem, fonte, criado_em)
                   VALUES (?,?,?,?,?,?,?)""",
                (interacao, prontuario, prioridade, categoria, mensagem, fonte,
                 datetime.now().isoformat(timespec="seconds")),
            )
            return int(cursor.lastrowid)

    def listar_alertas(self, prontuario: str | None = None, status: str = "aberto",
                       limite: int = 50) -> list[dict[str, Any]]:
        if prontuario:
            return self._consultar(
                "SELECT * FROM alertas WHERE prontuario = ? AND status = ? "
                "ORDER BY criado_em DESC LIMIT ?",
                (prontuario, status, limite),
            )
        return self._consultar(
            "SELECT * FROM alertas WHERE status = ? ORDER BY criado_em DESC LIMIT ?",
            (status, limite),
        )

    def registrar_validacao(self, interacao: str, sugestao: str,
                            prontuario: str | None = None,
                            fontes: list[str] | None = None) -> int:
        with self.conectar() as conexao:
            cursor = conexao.execute(
                """INSERT INTO validacoes (interacao, prontuario, sugestao, fontes, criado_em)
                   VALUES (?,?,?,?,?)""",
                (interacao, prontuario, sugestao, json.dumps(fontes or [], ensure_ascii=False),
                 datetime.now().isoformat(timespec="seconds")),
            )
            return int(cursor.lastrowid)

    def decidir_validacao(self, validacao_id: int, status: str, profissional: str,
                          justificativa: str | None = None) -> None:
        if status not in {"aceita", "recusada", "modificada"}:
            raise ValueError("Status de validacao invalido.")
        with self.conectar() as conexao:
            conexao.execute(
                """UPDATE validacoes
                   SET status = ?, profissional = ?, justificativa = ?, decidido_em = ?
                   WHERE id = ?""",
                (status, profissional, justificativa,
                 datetime.now().isoformat(timespec="seconds"), validacao_id),
            )

    def listar_validacoes(self, status: str | None = "pendente",
                          limite: int = 50) -> list[dict[str, Any]]:
        if status:
            return self._consultar(
                "SELECT * FROM validacoes WHERE status = ? ORDER BY criado_em DESC LIMIT ?",
                (status, limite),
            )
        return self._consultar(
            "SELECT * FROM validacoes ORDER BY criado_em DESC LIMIT ?", (limite,))

    def registrar_auditoria(self, interacao: str, evento: str, conteudo: dict[str, Any],
                            usuario: str | None = None, perfil: str | None = None,
                            prontuario: str | None = None) -> None:
        with self.conectar() as conexao:
            conexao.execute(
                """INSERT INTO auditoria
                   (interacao, registrado_em, evento, usuario, perfil, prontuario, conteudo)
                   VALUES (?,?,?,?,?,?,?)""",
                (interacao, datetime.now().isoformat(timespec="seconds"), evento, usuario,
                 perfil, prontuario, json.dumps(conteudo, ensure_ascii=False, default=str)),
            )

    def trilha_auditoria(self, interacao: str | None = None,
                         limite: int = 100) -> list[dict[str, Any]]:
        if interacao:
            return self._consultar(
                "SELECT * FROM auditoria WHERE interacao = ? ORDER BY id", (interacao,))
        return self._consultar(
            "SELECT * FROM auditoria ORDER BY id DESC LIMIT ?", (limite,))


def carregar_prontuarios_brutos(caminho: Path | None = None) -> list[dict[str, Any]]:
    caminho = Path(caminho or CAMINHOS.dados_brutos / "prontuarios.jsonl")
    with caminho.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def preparar_banco(caminho: Path | None = None, forcar: bool = False) -> BancoHospital:
    """Garante um banco populado, criando-o a partir dos prontuarios sinteticos."""
    banco = BancoHospital(caminho)
    if forcar or not banco.existe():
        banco.popular(carregar_prontuarios_brutos())
    return banco
