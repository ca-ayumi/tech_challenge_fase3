"""Baixa os datasets publicos usados no monitoramento multimodal.

Os arquivos vao para ``dados/multimodal/<dataset>/`` (fora do Git), junto de
um ``ORIGEM.md`` com a fonte, a licenca e a citacao exigida.

Uso:
    python scripts/baixar_datasets.py bidmc urfall
    python scripts/baixar_datasets.py --todos
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistente_medico.config import CAMINHOS  # noqa: E402

AGENTE = "tech-challenge-academico/1.0 (download de dataset publico)"


@dataclass(frozen=True)
class Dataset:
    nome: str
    descricao: str
    origem: str
    licenca: str
    citacao: str
    arquivos: Callable[[], list[tuple[str, str]]]


def _bidmc() -> list[tuple[str, str]]:
    base = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv"
    return [(f"{base}/bidmc_{i:02d}_Numerics.csv", f"bidmc_{i:02d}_Numerics.csv")
            for i in range(1, 54)]


def _urfall_acelerometro() -> list[tuple[str, str]]:
    base = "http://fenix.ur.edu.pl/~mkepski/ds/data"
    quedas = [(f"{base}/fall-{i:02d}-acc.csv", f"fall-{i:02d}-acc.csv") for i in range(1, 31)]
    atividades = [(f"{base}/adl-{i:02d}-acc.csv", f"adl-{i:02d}-acc.csv") for i in range(1, 41)]
    return quedas + atividades


def _voiced() -> list[tuple[str, str]]:
    base = "https://physionet.org/files/voiced/1.0.0"
    arquivos = []
    for i in range(1, 209):
        nome = f"voice{i:03d}"
        arquivos += [(f"{base}/{nome}.hea", f"{nome}.hea"), (f"{base}/{nome}.dat", f"{nome}.dat"),
                     (f"{base}/{nome}-info.txt", f"{nome}-info.txt")]
    return arquivos


DATASETS = {
    "bidmc": Dataset(
        nome="bidmc",
        descricao="BIDMC PPG and Respiration Dataset: 53 registros de 8 min de pacientes "
                  "criticos, com FC, pulso, frequencia respiratoria e SpO2 a 1 Hz.",
        origem="https://physionet.org/content/bidmc/1.0.0/",
        licenca="Open Data Commons Attribution License v1.0 (ODC-By)",
        citacao="Pimentel MAF et al. Towards a Robust Estimation of Respiratory Rate from "
                "Pulse Oximeters. IEEE Trans Biomed Eng. 2017;64(8):1914-1923. "
                "Goldberger A et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. "
                "2000;101(23):e215-e220.",
        arquivos=_bidmc,
    ),
    "urfall": Dataset(
        nome="urfall",
        descricao="UR Fall Detection Dataset, acelerometro: 30 quedas e 40 atividades da vida "
                  "diaria; colunas tempo (ms), magnitude total (g), Ax, Ay, Az.",
        origem="https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
        licenca="Creative Commons BY-NC-SA 4.0 (uso academico nao comercial)",
        citacao="Kwolek B, Kepski M. Human fall detection on embedded platform using depth "
                "maps and wireless accelerometer. Comput Methods Programs Biomed. "
                "2014;117(3):489-501.",
        arquivos=_urfall_acelerometro,
    ),
    "voiced": Dataset(
        nome="voiced",
        descricao="VOICED: 208 gravacoes da vogal /a/ sustentada (8 kHz), 57 vozes saudaveis "
                  "e 151 com patologia vocal diagnosticada.",
        origem="https://physionet.org/content/voiced/1.0.0/",
        licenca="Open Data Commons Attribution License v1.0 (ODC-By)",
        citacao="Cesari U, De Pietro G, Marciano E, Niri C, Sannino G, Verde L. A new database "
                "of healthy and pathological voices. Computers & Electrical Engineering. "
                "2018;68:310-321.",
        arquivos=_voiced,
    ),
}


def baixar(url: str, destino: Path, tentativas: int = 3) -> bool:
    if destino.exists() and destino.stat().st_size > 0:
        return False
    requisicao = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    for tentativa in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(requisicao, timeout=60) as resposta:
                conteudo = resposta.read()
            temporario = destino.with_suffix(destino.suffix + ".parcial")
            temporario.write_bytes(conteudo)
            temporario.replace(destino)
            return True
        except (urllib.error.URLError, TimeoutError) as erro:
            if tentativa == tentativas:
                raise RuntimeError(f"Falha ao baixar {url}: {erro}") from erro
            time.sleep(2 * tentativa)
    return False


def preparar(dataset: Dataset, raiz: Path) -> Path:
    pasta = raiz / dataset.nome
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = dataset.arquivos()
    novos = 0
    for indice, (url, nome) in enumerate(arquivos, start=1):
        if baixar(url, pasta / nome):
            novos += 1
            time.sleep(0.2)
        print(f"\r  {dataset.nome}: {indice}/{len(arquivos)}", end="", flush=True)
    print(f"  ({novos} novos)")
    (pasta / "ORIGEM.md").write_text(
        f"# {dataset.nome}\n\n{dataset.descricao}\n\n"
        f"- Origem: {dataset.origem}\n- Licenca: {dataset.licenca}\n"
        f"- Citacao: {dataset.citacao}\n",
        encoding="utf-8",
    )
    return pasta


def main(argumentos: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help=f"Um ou mais de: {', '.join(DATASETS)}")
    parser.add_argument("--todos", action="store_true")
    args = parser.parse_args(argumentos)
    desconhecidos = set(args.datasets) - set(DATASETS)
    if desconhecidos:
        parser.error(f"dataset desconhecido: {', '.join(sorted(desconhecidos))}")
    escolhidos = list(DATASETS) if args.todos or not args.datasets else args.datasets
    for nome in escolhidos:
        preparar(DATASETS[nome], CAMINHOS.multimodal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
