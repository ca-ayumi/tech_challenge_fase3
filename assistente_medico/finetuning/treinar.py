"""Pipeline de fine-tuning por LoRA.

Dois backends, mesma receita e mesmo dataset:

- ``mlx``: roda em Apple Silicon com ``mlx_lm.lora``. E o caminho usado na
  demonstracao, porque treina o modelo inteiro em minutos num MacBook.
- ``peft``: roda em GPU NVIDIA (Colab) com transformers + peft + trl.

A escolha por LoRA, e nao por ajuste completo dos pesos, e deliberada: o
adaptador tem poucos megabytes, pode ser versionado junto do codigo, e permite
reverter o ajuste sem trocar o modelo base — o que importa num contexto
hospitalar em que o modelo base e auditado separadamente.

Uso:
    python -m assistente_medico.finetuning.treinar --iteracoes 400
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import CAMINHOS, MODELO, TREINO


def _escrever_yaml(dados: dict[str, Any], destino: Path) -> Path:
    """Escreve um YAML simples (um nivel de aninhamento), sem dependencia extra."""
    linhas = []
    for chave, valor in dados.items():
        if isinstance(valor, dict):
            linhas.append(f"{chave}:")
            for subchave, subvalor in valor.items():
                linhas.append(f"  {subchave}: {_formatar_yaml(subvalor)}")
        else:
            linhas.append(f"{chave}: {_formatar_yaml(valor)}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return destino


def _formatar_yaml(valor: Any) -> str:
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if valor is None:
        return "null"
    if isinstance(valor, str):
        return f'"{valor}"'
    return str(valor)


def montar_configuracao(
    modelo: str,
    dados: Path,
    adaptador: Path,
    iteracoes: int,
    lote: int,
    camadas: int,
    rank: int,
    escala: float,
    taxa_aprendizado: float,
    max_seq: int,
    semente: int,
) -> dict[str, Any]:
    return {
        "model": modelo,
        "train": True,
        "data": str(dados),
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "num_layers": camadas,
        "batch_size": lote,
        "iters": iteracoes,
        "val_batches": 10,
        "learning_rate": taxa_aprendizado,
        "steps_per_report": 20,
        "steps_per_eval": 50,
        "adapter_path": str(adaptador),
        "save_every": 50,
        "max_seq_length": max_seq,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "seed": semente,
        "lora_parameters": {"rank": rank, "scale": escala, "dropout": 0.05},
    }


def treinar_mlx(configuracao: dict[str, Any], caminho_config: Path,
                caminho_log: Path) -> dict[str, Any]:
    """Executa o treinamento com mlx_lm.lora e captura as metricas do log."""
    _escrever_yaml(configuracao, caminho_config)
    comando = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(caminho_config)]

    inicio = time.perf_counter()
    caminho_log.parent.mkdir(parents=True, exist_ok=True)
    with caminho_log.open("w", encoding="utf-8") as log:
        processo = subprocess.Popen(
            comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        linhas: list[str] = []
        for linha in processo.stdout:  # type: ignore[union-attr]
            log.write(linha)
            log.flush()
            linhas.append(linha)
            if linha.startswith(("Iter", "Trainable", "Loading", "Starting")):
                print(linha.rstrip())
        codigo = processo.wait()
    duracao = time.perf_counter() - inicio

    if codigo != 0:
        raise RuntimeError(
            f"Treinamento falhou (codigo {codigo}). Log completo em {caminho_log}"
        )
    return {"duracao_segundos": round(duracao, 1), "metricas": extrair_metricas(linhas)}


_METRICAS_LOG = {
    "iteracao": re.compile(r"\bIter\s+(\d+)"),
    "perda_treino": re.compile(r"\bTrain loss\s+([\d.]+)"),
    "perda_validacao": re.compile(r"\bVal loss\s+([\d.]+)"),
    "taxa_aprendizado": re.compile(r"\bLearning Rate\s+([\d.e+-]+)"),
    "tokens_por_segundo": re.compile(r"\bTokens/sec\s+([\d.]+)"),
    "memoria_pico_gb": re.compile(r"\bPeak mem\s+([\d.]+)\s*GB"),
    "tokens_treinados": re.compile(r"\bTrained Tokens\s+(\d+)"),
}


def extrair_metricas(linhas: list[str]) -> dict[str, Any]:
    """Extrai as curvas de perda de treino e de validacao do log do mlx_lm."""
    treino: list[dict[str, float]] = []
    validacao: list[dict[str, float]] = []

    for linha in linhas:
        linha = linha.strip()
        if not linha.startswith("Iter"):
            continue
        registro: dict[str, float] = {}
        for nome, padrao in _METRICAS_LOG.items():
            achado = padrao.search(linha)
            if achado:
                try:
                    registro[nome] = float(achado.group(1))
                except ValueError:
                    continue
        if "perda_validacao" in registro:
            validacao.append(registro)
        elif "perda_treino" in registro:
            treino.append(registro)

    resumo: dict[str, Any] = {"treino": treino, "validacao": validacao}
    if validacao:
        melhor = min(validacao, key=lambda r: r["perda_validacao"])
        resumo["perda_validacao_inicial"] = validacao[0]["perda_validacao"]
        resumo["perda_validacao_final"] = validacao[-1]["perda_validacao"]
        resumo["melhor_perda_validacao"] = melhor["perda_validacao"]
        resumo["melhor_iteracao"] = melhor.get("iteracao")
    if treino:
        resumo["perda_treino_inicial"] = treino[0]["perda_treino"]
        resumo["perda_treino_final"] = treino[-1]["perda_treino"]
        resumo["tokens_por_segundo_medio"] = round(
            sum(r.get("tokens_por_segundo", 0.0) for r in treino) / len(treino), 1
        )
        resumo["memoria_pico_gb"] = max(
            (r.get("memoria_pico_gb", 0.0) for r in treino), default=0.0
        )
    return resumo


def promover_melhor_checkpoint(adaptador: Path, metricas: dict[str, Any]) -> dict[str, Any]:
    """Entrega o checkpoint de menor perda de validacao, nao o da ultima iteracao.

    Com poucos exemplos o sobreajuste comeca antes do fim do treino, e a ultima
    iteracao costuma ser pior que o melhor ponto da curva. Como ``save_every``
    esta alinhado com ``steps_per_eval``, todo ponto avaliado tem um checkpoint
    correspondente e o melhor pode ser promovido a adaptador oficial.
    """
    melhor_iteracao = metricas.get("melhor_iteracao")
    entregue = adaptador / "adapters.safetensors"
    if melhor_iteracao is None or not entregue.exists():
        return {"checkpoint_entregue": "ultima_iteracao", "selecao": "indisponivel"}

    iteracao = int(melhor_iteracao)
    candidato = adaptador / f"{iteracao:07d}_adapters.safetensors"
    if not candidato.exists():
        return {"checkpoint_entregue": "ultima_iteracao",
                "selecao": f"sem checkpoint para a iteracao {iteracao}"}

    entregue.write_bytes(candidato.read_bytes())
    return {
        "checkpoint_entregue": iteracao,
        "perda_validacao_entregue": metricas.get("melhor_perda_validacao"),
        "selecao": "menor perda de validacao",
    }


def treinar_peft(configuracao: dict[str, Any]) -> dict[str, Any]:
    """Treinamento equivalente em GPU NVIDIA, com transformers + peft + trl."""
    from .backend_peft import treinar as treinar_com_peft

    return treinar_com_peft(configuracao)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "peft"], default="mlx")
    parser.add_argument("--modelo", default=None, help="Modelo base (padrao: config).")
    parser.add_argument("--dados", type=Path, default=CAMINHOS.dados_processados)
    parser.add_argument("--adaptador", type=Path, default=CAMINHOS.adaptador)
    parser.add_argument("--iteracoes", type=int, default=TREINO.iteracoes)
    parser.add_argument("--lote", type=int, default=TREINO.lote)
    parser.add_argument("--camadas", type=int, default=TREINO.camadas_lora)
    parser.add_argument("--rank", type=int, default=TREINO.rank)
    parser.add_argument("--escala", type=float, default=TREINO.escala)
    parser.add_argument("--taxa-aprendizado", type=float, default=TREINO.taxa_aprendizado)
    parser.add_argument("--max-seq", type=int, default=TREINO.max_tokens_sequencia)
    parser.add_argument("--semente", type=int, default=TREINO.semente)
    args = parser.parse_args()

    if not (args.dados / "train.jsonl").exists():
        raise SystemExit(
            f"Dataset nao encontrado em {args.dados}. "
            "Rode antes: python -m assistente_medico.dados.construir_dataset"
        )

    modelo = args.modelo or (MODELO.modelo_base_mlx if args.backend == "mlx"
                             else MODELO.modelo_base_hf)
    configuracao = montar_configuracao(
        modelo=modelo, dados=args.dados, adaptador=args.adaptador,
        iteracoes=args.iteracoes, lote=args.lote, camadas=args.camadas,
        rank=args.rank, escala=args.escala, taxa_aprendizado=args.taxa_aprendizado,
        max_seq=args.max_seq, semente=args.semente,
    )

    CAMINHOS.garantir()
    caminho_config = args.adaptador / "config_treino.yaml"
    caminho_log = CAMINHOS.logs / "treino_lora.log"

    print(f"Backend: {args.backend} | modelo base: {modelo}")
    print(f"Dataset: {args.dados} | adaptador: {args.adaptador}")

    if args.backend == "mlx":
        resultado = treinar_mlx(configuracao, caminho_config, caminho_log)
    else:
        resultado = treinar_peft(configuracao)

    selecao = promover_melhor_checkpoint(args.adaptador, resultado.get("metricas", {}))

    metadados = {
        "concluido_em": datetime.now().isoformat(timespec="seconds"),
        "selecao_checkpoint": selecao,
        "backend": args.backend,
        "modelo_base": modelo,
        "hiperparametros": {
            "iteracoes": args.iteracoes, "lote": args.lote, "camadas_lora": args.camadas,
            "rank": args.rank, "escala": args.escala,
            "taxa_aprendizado": args.taxa_aprendizado,
            "max_seq_length": args.max_seq, "mask_prompt": True, "semente": args.semente,
        },
        **resultado,
    }
    destino = args.adaptador / "metadados_treino.json"
    destino.write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")

    metricas = resultado.get("metricas", {})
    if metricas.get("validacao"):
        primeira = metricas["validacao"][0].get("perda_validacao")
        ultima = metricas["validacao"][-1].get("perda_validacao")
        print(f"Perda de validacao: {primeira} -> {ultima}")
    if isinstance(selecao.get("checkpoint_entregue"), int):
        print(f"Checkpoint entregue: iteracao {selecao['checkpoint_entregue']} "
              f"(val loss {selecao['perda_validacao_entregue']})")
    print(f"Treino concluido em {resultado.get('duracao_segundos')}s")
    print(f"Adaptador salvo em {args.adaptador}")
    print(f"Metadados: {destino}")


if __name__ == "__main__":
    main()
