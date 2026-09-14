"""Fine-tuning por LoRA com transformers + peft + trl.

Caminho alternativo ao MLX, para quem for reproduzir o treinamento em GPU
NVIDIA (Google Colab, por exemplo) ou em CPU. Consome exatamente o mesmo
dataset gerado por ``assistente_medico.dados.construir_dataset``: os arquivos
``train.jsonl`` e ``valid.jsonl`` em formato de chat.

As dependencias nao entram no requirements principal porque torch pesa mais de
2 GB e nao e necessario no caminho MLX. Instale com:

    pip install -r requirements-treino-cuda.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def carregar_dataset(diretorio: Path):
    """Carrega train/valid no formato de chat como um ``DatasetDict``."""
    from datasets import Dataset, DatasetDict

    def ler(nome: str) -> Dataset:
        caminho = diretorio / nome
        registros = [json.loads(linha) for linha in caminho.open(encoding="utf-8")
                     if linha.strip()]
        return Dataset.from_list(registros)

    return DatasetDict({"train": ler("train.jsonl"), "validation": ler("valid.jsonl")})


def treinar(configuracao: dict[str, Any]) -> dict[str, Any]:
    """Executa o fine-tuning com SFTTrainer e salva o adaptador."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    modelo_base = configuracao["model"]
    destino = Path(configuracao["adapter_path"])
    destino.mkdir(parents=True, exist_ok=True)

    tokenizador = AutoTokenizer.from_pretrained(modelo_base)
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token

    modelo = AutoModelForCausalLM.from_pretrained(
        modelo_base,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    parametros_lora = configuracao.get("lora_parameters", {})
    config_lora = LoraConfig(
        r=int(parametros_lora.get("rank", 16)),
        lora_alpha=int(parametros_lora.get("scale", 20)),
        lora_dropout=float(parametros_lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dados = carregar_dataset(Path(configuracao["data"]))

    argumentos = SFTConfig(
        output_dir=str(destino / "checkpoints"),
        max_steps=int(configuracao.get("iters", 400)),
        per_device_train_batch_size=int(configuracao.get("batch_size", 2)),
        learning_rate=float(configuracao.get("learning_rate", 1e-5)),
        logging_steps=int(configuracao.get("steps_per_report", 20)),
        eval_strategy="steps",
        eval_steps=int(configuracao.get("steps_per_eval", 50)),
        save_steps=int(configuracao.get("save_every", 100)),
        max_length=int(configuracao.get("max_seq_length", 1280)),
        gradient_checkpointing=bool(configuracao.get("grad_checkpoint", True)),
        seed=int(configuracao.get("seed", 42)),
        report_to=[],
        completion_only_loss=True,
    )

    treinador = SFTTrainer(
        model=modelo,
        args=argumentos,
        train_dataset=dados["train"],
        eval_dataset=dados["validation"],
        peft_config=config_lora,
        processing_class=tokenizador,
    )

    resultado = treinador.train()
    treinador.save_model(str(destino))
    tokenizador.save_pretrained(str(destino))

    historico = [
        {chave: valor for chave, valor in registro.items()
         if chave in {"step", "loss", "eval_loss", "learning_rate"}}
        for registro in treinador.state.log_history
    ]
    return {
        "duracao_segundos": round(resultado.metrics.get("train_runtime", 0.0), 1),
        "metricas": {
            "treino": [h for h in historico if "loss" in h],
            "validacao": [h for h in historico if "eval_loss" in h],
        },
    }
