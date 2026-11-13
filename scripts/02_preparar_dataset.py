"""Etapa 2: une as três fontes em um único dataset YOLO com as classes do Encosta Viva.

Classes finais (id -> nome):
  0 rachadura              <- building-cracks "crack" + wall-damage leve(1) e moderada(2)
  1 cicatriz_deslizamento  <- objectRecognition_Landslides "old landslides" + "active landslides"
  2 dano_estrutural        <- wall-damage grave(3): desplacamento, ferragem exposta, colapso parcial

Correção importante (documentada no relatório): no dataset de deslizamentos, os IDs de
categoria estão deslocados +1 em relação aos nomes do cartão. Conferido visualmente
(mosaico em docs/img/mosaico_landslides.jpg): id 7 = deslizamento antigo (vegetado),
id 10 = deslizamento ativo (solo exposto), id 11 = floresta, id 6 = casa.
Imagens de deslizamento sem cicatriz entram como fundo (rótulo vazio): ensinam o modelo
a NÃO marcar casa, mata ou rio como cicatriz.

Divisão 70/20/10 por fonte, agrupando cópias aumentadas da mesma foto original
(Roboflow gera "crack-123-_jpg.rf.<hash>.jpg"): todas as cópias caem no mesmo split,
evitando vazamento de treino para teste.

Augmentation offline (só no treino, ~30% das imagens): chuva sintética, neblina,
pouca luz e desfoque de movimento, simulando foto de celular ruim em dia de chuva.
O restante (brilho/HSV, rotação leve, flip, mosaic) é feito on-line pelo Ultralytics.

Uso:
    python scripts/02_preparar_dataset.py [--max-rachaduras 650] [--seed 42]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
SAIDA = RAIZ / "data" / "encosta"
CLASSES = ["rachadura", "cicatriz_deslizamento", "dano_estrutural"]
TAM = 640  # lado máximo salvo


# ---------------------------------------------------------------- utilidades
def caixa_yolo(x: float, y: float, w: float, h: float, W: int, H: int) -> tuple[float, float, float, float] | None:
    """COCO (x,y,w,h em px) -> YOLO normalizado, cortando para dentro da imagem."""
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(W), x + w), min(float(H), y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)


def redimensionar(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    if max(img.size) > TAM:
        img.thumbnail((TAM, TAM), Image.LANCZOS)  # normalização da caixa é relativa: não muda
    return img


class Amostra:
    __slots__ = ("fonte", "grupo", "img", "rotulos")

    def __init__(self, fonte: str, grupo: str, img: Image.Image, rotulos: list[tuple[int, tuple]]):
        self.fonte, self.grupo, self.img, self.rotulos = fonte, grupo, img, rotulos


# ---------------------------------------------------------------- leitores
def ler_rachaduras(max_imgs: int, rng: random.Random) -> list[Amostra]:
    grupos: dict[str, list[tuple[zipfile.ZipFile, dict, list]]] = defaultdict(list)
    zips = []
    for nome in ("cracks_train.zip", "cracks_valid.zip", "cracks_test.zip"):
        z = zipfile.ZipFile(RAW / nome)
        zips.append(z)
        coco = json.loads(z.read("_annotations.coco.json"))
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        for im in coco["images"]:
            base = re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", im["file_name"])
            grupos[base].append((z, im, anns[im["id"]]))
    # Uma cópia por foto original (evita o modelo ver a mesma rachadura 3x) e limita o total
    chaves = sorted(grupos)
    rng.shuffle(chaves)
    saida = []
    for base in chaves:
        if len(saida) >= max_imgs:
            break
        z, im, anns = rng.choice(grupos[base])
        rot = [(0, b) for a in anns if (b := caixa_yolo(*a["bbox"], im["width"], im["height"]))]
        if not rot:
            continue
        img = redimensionar(Image.open(io.BytesIO(z.read(im["file_name"]))))
        saida.append(Amostra("building-cracks", f"crack:{base}", img, rot))
    for z in zips:
        z.close()
    return saida


def ler_deslizamentos() -> list[Amostra]:
    # IDs reais (verificados visualmente): 7 = old landslides, 10 = active landslides
    CICATRIZ = {7, 10}
    saida = []
    for i, r in enumerate(pq.read_table(RAW / "landslides.parquet").to_pylist()):
        img = Image.open(io.BytesIO(r["image"]["bytes"]))
        W, H = img.size
        rot = [
            (1, b)
            for bb, c in zip(r["objects"]["bbox"], r["objects"]["category"])
            if c in CICATRIZ and (b := caixa_yolo(*bb, W, H))
        ]
        saida.append(Amostra("landslides", f"land:{r.get('image_id', i)}", redimensionar(img), rot))
    return saida


def ler_paredes() -> list[Amostra]:
    MAPA = {1: 0, 2: 0, 3: 2}  # leve/moderada -> rachadura; grave -> dano_estrutural
    saida = []
    for arq in ("wall_train.parquet", "wall_val.parquet", "wall_test.parquet"):
        for r in pq.read_table(RAW / arq).to_pylist():
            img = Image.open(io.BytesIO(r["image"]["bytes"]))
            W, H = img.size
            rot = [
                (MAPA[c], b)
                for bb, c in zip(r["objects"]["bbox"], r["objects"]["category"])
                if c in MAPA and (b := caixa_yolo(*bb, W, H))
            ]
            if rot:
                saida.append(Amostra("wall-damage", f"wall:{arq}:{r['image_id']}", redimensionar(img), rot))
    return saida


# ---------------------------------------------------------------- augmentation "foto ruim de chuva"
def chuva(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    camada = np.zeros((h, w), np.uint8)
    n = int(h * w / 600)
    comp = rng.integers(8, 20)
    ang = rng.uniform(-0.35, 0.35)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    for x, y in zip(xs, ys):
        x2, y2 = int(x + comp * np.sin(ang)), int(y + comp * np.cos(ang))
        cv2.line(camada, (int(x), int(y)), (x2, y2), 200, 1)
    camada = cv2.blur(camada, (3, 3))
    out = img.astype(np.float32) * 0.82  # céu de chuva escurece
    out += camada[..., None].astype(np.float32) * 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def neblina(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0.30, 0.55)
    return cv2.addWeighted(img, 1 - a, np.full_like(img, 215), a, 0)


def pouca_luz(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = rng.uniform(1.6, 2.4)
    out = 255.0 * (img / 255.0) ** g
    ruido = rng.normal(0, 6, img.shape)
    return np.clip(out + ruido, 0, 255).astype(np.uint8)


def borrao_movimento(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(5, 11))
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2, k / 2), float(rng.uniform(0, 180)), 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img, -1, ker)


EFEITOS = {"chuva": chuva, "neblina": neblina, "pouca_luz": pouca_luz, "borrao": borrao_movimento}


# ---------------------------------------------------------------- escrita
def gravar(a: Amostra, split: str, nome: str, img: Image.Image | None = None) -> None:
    (SAIDA / "images" / split).mkdir(parents=True, exist_ok=True)
    (SAIDA / "labels" / split).mkdir(parents=True, exist_ok=True)
    (img or a.img).save(SAIDA / "images" / split / f"{nome}.jpg", quality=92)  # PIL não copia EXIF
    linhas = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, (x, y, w, h) in a.rotulos]
    (SAIDA / "labels" / split / f"{nome}.txt").write_text("\n".join(linhas), encoding="utf-8")


def dividir(amostras: list[Amostra], rng: random.Random) -> dict[str, list[Amostra]]:
    por_grupo: dict[str, list[Amostra]] = defaultdict(list)
    for a in amostras:
        por_grupo[a.grupo].append(a)
    grupos = sorted(por_grupo)
    rng.shuffle(grupos)
    n = len(grupos)
    cortes = {"train": grupos[: int(n * 0.7)], "val": grupos[int(n * 0.7) : int(n * 0.9)], "test": grupos[int(n * 0.9) :]}
    return {s: [a for g in gs for a in por_grupo[g]] for s, gs in cortes.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rachaduras", type=int, default=650)
    ap.add_argument("--frac-aug", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)

    if SAIDA.exists():
        shutil.rmtree(SAIDA)

    fontes = {
        "building-cracks": ler_rachaduras(args.max_rachaduras, rng),
        "landslides": ler_deslizamentos(),
        "wall-damage": ler_paredes(),
    }
    relatorio = {"classes": CLASSES, "seed": args.seed, "fontes": {}, "splits": {}}
    splits: dict[str, list[Amostra]] = {"train": [], "val": [], "test": []}
    for fonte, ams in fontes.items():
        partes = dividir(ams, rng)  # 70/20/10 dentro de cada fonte = estratificado por fonte
        relatorio["fontes"][fonte] = {s: len(v) for s, v in partes.items()}
        for s, v in partes.items():
            splits[s].extend(v)

    aug_contagem: Counter = Counter()
    for split, ams in splits.items():
        caixas: Counter = Counter()
        fundo = 0
        for i, a in enumerate(ams):
            nome = f"{a.fonte}_{i:05d}"
            gravar(a, split, nome)
            caixas.update(c for c, _ in a.rotulos)
            fundo += not a.rotulos
            if split == "train" and rng.random() < args.frac_aug:
                efeito = rng.choice(list(EFEITOS))
                arr = EFEITOS[efeito](np.asarray(a.img), nrng)
                gravar(a, split, f"{nome}_aug_{efeito}", Image.fromarray(arr))
                aug_contagem[efeito] += 1
                caixas.update(c for c, _ in a.rotulos)
        n_aug = sum(aug_contagem.values()) if split == "train" else 0
        relatorio["splits"][split] = {
            "imagens_originais": len(ams),
            "imagens_aumentadas": n_aug,
            "imagens_fundo": fundo,
            "caixas_por_classe": {CLASSES[k]: v for k, v in sorted(caixas.items())},
        }
    relatorio["augmentation_offline"] = dict(aug_contagem)

    yaml = (
        f"# Gerado por scripts/02_preparar_dataset.py\npath: {SAIDA.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    (SAIDA / "encosta.yaml").write_text(yaml, encoding="utf-8")
    (RAIZ / "configs").mkdir(exist_ok=True)
    (RAIZ / "configs" / "dataset_resumo.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
