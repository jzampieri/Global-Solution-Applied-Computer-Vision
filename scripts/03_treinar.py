"""Etapa 3: treino customizado (transfer learning) do YOLO11 no dataset Encosta Viva.

Parte de pesos pré-treinados no COCO (yolo11n.pt / yolo11s.pt) e ajusta para as 3 classes.

Augmentation on-line do Ultralytics, escolhida para o cenário "foto de celular em encosta":
  hsv_v=0.5   brilho (sol forte / dia nublado)
  degrees=10  rotação leve (celular torto)
  fliplr=0.5  espelhamento horizontal
  mosaic=1.0  combina 4 fotos (contexto variado)
  scale=0.5   aproximação/afastamento
  + offline (etapa 2): chuva, neblina, pouca luz, borrão de movimento

Uso:
    python scripts/03_treinar.py --modelo yolo11n.pt --epocas 80
    python scripts/03_treinar.py --modelo yolo11s.pt --epocas 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--epocas", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' para GPU, 'cpu' para CPU")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    nome = Path(args.modelo).stem.replace("yolo", "encosta_")
    modelo = YOLO(args.modelo)
    modelo.train(
        data=str(RAIZ / "data" / "encosta" / "encosta.yaml"),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache="ram",  # ~1,5 GB: evita decodificar JPEG a cada época (no Windows o loader era o gargalo)
        project=str(RAIZ / "runs"),
        name=nome,
        exist_ok=True,
        seed=42,
        deterministic=True,
        patience=25,
        cos_lr=True,
        close_mosaic=10,
        hsv_v=0.5,
        degrees=10.0,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        plots=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
