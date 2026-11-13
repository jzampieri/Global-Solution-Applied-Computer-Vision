"""Etapa 1: baixa os datasets públicos usados no treino (Hugging Face, sem login).

Fontes (todas com licença aberta, ver README seção "Dados"):
  - lipi17/building-cracks          -> rachaduras em paredes e estruturas (CC BY 4.0, COCO)
  - bbrenes/objectRecognition_Landslides -> cicatrizes de deslizamento (MIT, parquet)
  - Francesco/wall-damage           -> dano em parede / Roboflow-100 (CC BY 4.0, parquet)

Uso:
    python scripts/01_baixar_dados.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "raw"
HF = "https://huggingface.co/datasets"

ARQUIVOS = {
    "cracks_train.zip": f"{HF}/lipi17/building-cracks/resolve/main/data/train.zip",
    "cracks_valid.zip": f"{HF}/lipi17/building-cracks/resolve/main/data/valid.zip",
    "cracks_test.zip": f"{HF}/lipi17/building-cracks/resolve/main/data/test.zip",
    "landslides.parquet": f"{HF}/bbrenes/objectRecognition_Landslides/resolve/main/data/train-00000-of-00001.parquet",
    "wall_train.parquet": f"{HF}/Francesco/wall-damage/resolve/main/data/train-00000-of-00001-99984d1ffc205e04.parquet",
    "wall_val.parquet": f"{HF}/Francesco/wall-damage/resolve/main/data/validation-00000-of-00001-b45a39c75e8c66dc.parquet",
    "wall_test.parquet": f"{HF}/Francesco/wall-damage/resolve/main/data/test-00000-of-00001-1962293878345fe1.parquet",
}


def baixar(nome: str, url: str) -> None:
    alvo = DESTINO / nome
    if alvo.exists() and alvo.stat().st_size > 0:
        print(f"[ok] {nome} já existe ({alvo.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[baixando] {nome} ...", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = alvo.with_suffix(alvo.suffix + ".part")
        with tmp.open("wb") as f:
            for bloco in r.iter_content(chunk_size=1 << 20):
                f.write(bloco)
        tmp.replace(alvo)
    print(f"[ok] {nome} ({alvo.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nome, url in ARQUIVOS.items():
        baixar(nome, url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
