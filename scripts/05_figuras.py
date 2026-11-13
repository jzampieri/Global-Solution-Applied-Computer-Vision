"""Etapa 5: figuras de documentação (pipeline, inspeção do dataset, augmentation, quadro de demo).

Uso:
    python scripts/05_figuras.py
"""
from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
IMG = RAIZ / "docs" / "img"
DADOS = RAIZ / "data" / "encosta"
sys.path.insert(0, str(RAIZ / "scripts"))


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    caixas = [
        (0.2, "Foto do morador\nou câmera", "app PWA · celular\nIP Webcam · ESP32-CAM", "#e8f1ee"),
        (2.75, "LGPD primeiro", "EXIF/GPS descartado\nrostos borrados", "#fdf1dc"),
        (5.3, "YOLO11 treinado", "rachadura\ncicatriz_deslizamento\ndano_estrutural", "#e3ecfa"),
        (7.85, "Regra explicável", "Observação · Atenção\nAlerta (+ humano)", "#fbe5e3"),
        (10.4, "Fila da Defesa Civil", "POST /visao/detectar\n→ API · Automação", "#ececec"),
    ]
    for x, tit, sub, cor in caixas:
        ax.add_patch(FancyBboxPatch((x, 1.0), 2.3, 2.4, boxstyle="round,pad=0.04,rounding_size=0.15", fc=cor, ec="#333333", lw=1.2))
        ax.text(x + 1.15, 2.85, tit, ha="center", va="center", fontsize=11.5, weight="bold")
        ax.text(x + 1.15, 1.75, sub, ha="center", va="center", fontsize=9.2, color="#333333")
    for x in (2.5, 5.05, 7.6, 10.15):
        ax.add_patch(FancyArrowPatch((x, 2.2), (x + 0.25, 2.2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333333"))
    ax.text(6.5, 0.45, "Treino: 3 datasets públicos → conversão COCO→YOLO → divisão 70/20/10 por foto original → augmentation (chuva, neblina, pouca luz, borrão) → yolo11n × yolo11s",
            ha="center", fontsize=9, color="#555555")
    ax.text(6.5, 3.85, "A IA sugere, a Defesa Civil decide", ha="center", fontsize=10.5, style="italic", color="#0b5d4b")
    fig.savefig(IMG / "pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mosaico_deslizamentos() -> None:
    t = pq.read_table(RAIZ / "data" / "raw" / "landslides.parquet").to_pylist()
    nomes_reais = {2: "pessoa", 4: "montanha/mata", 5: "veículo", 6: "casa", 7: "desliz. antigo", 8: "cidade", 9: "rio", 10: "desliz. ativo", 11: "floresta"}
    crops = defaultdict(list)
    for r in t:
        im = None
        for b, k in zip(r["objects"]["bbox"], r["objects"]["category"]):
            if len(crops[k]) >= 6 or b[2] < 25 or b[3] < 25:
                continue
            im = im or Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            x, y, w, h = b
            c = im.crop((x, y, x + w, y + h))
            c.thumbnail((100, 100))
            crops[k].append(c)
    ks = sorted(crops)
    W = Image.new("RGB", (170 + 6 * 104, len(ks) * 104), "white")
    d = ImageDraw.Draw(W)
    for i, k in enumerate(ks):
        d.text((6, i * 104 + 40), f"id {k}\n{nomes_reais.get(k, '?')}", fill=(200, 0, 0) if k in (7, 10) else "black")
        for j, c in enumerate(crops[k]):
            W.paste(c, (170 + j * 104, i * 104))
    W.save(IMG / "mosaico_landslides.jpg", quality=90)


def augmentation() -> None:
    import importlib

    prep = importlib.import_module("02_preparar_dataset")
    rng = np.random.default_rng(3)
    base = sorted((DADOS / "images" / "test").glob("building-cracks_*.jpg"))[3]
    img = np.asarray(Image.open(base).convert("RGB"))
    H, W = img.shape[:2]
    rot = [list(map(float, l.split())) for l in (DADOS / "labels" / "test" / (base.stem + ".txt")).read_text().splitlines()]
    tiles = []
    for nome, f in [("original", None), *prep.EFEITOS.items()]:
        a = img.copy() if f is None else f(img.copy(), rng)
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        for c, x, y, w, h in rot:
            cv2.rectangle(a, (int((x - w / 2) * W), int((y - h / 2) * H)), (int((x + w / 2) * W), int((y + h / 2) * H)), (0, 0, 230), 3)
        a = cv2.resize(a, (300, int(300 * H / W)))
        faixa = np.full((30, a.shape[1], 3), 25, np.uint8)
        cv2.putText(faixa, nome, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([a, faixa]))
    cv2.imwrite(str(IMG / "augmentation.jpg"), np.hstack(tiles))


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    pipeline()
    mosaico_deslizamentos()
    augmentation()
    print("figuras em", IMG)


if __name__ == "__main__":
    main()
