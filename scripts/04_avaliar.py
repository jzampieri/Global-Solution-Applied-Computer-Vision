"""Etapa 4: avalia os modelos treinados no split de TESTE (nunca visto no treino nem na seleção).

Gera:
  avaliacao/metricas.json         mAP@50, mAP@50-95, precisão, recall (geral e por classe), FPS
  avaliacao/comparacao.md         tabela yolo11n x yolo11s
  docs/img/acertos.jpg            exemplos de acerto (verde = gabarito, cor = predição)
  docs/img/erros.jpg              exemplos de erro (falso positivo / falso negativo)
  docs/img/<modelo>_*.png         matriz de confusão, curva PR (cópia do Ultralytics)
  models/encosta_visao.pt + .json o modelo escolhido para a API

Uso:
    python scripts/04_avaliar.py [--escolher auto|n|s]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "data" / "encosta"
YAML = DADOS / "encosta.yaml"
AVAL = RAIZ / "avaliacao"
IMG = RAIZ / "docs" / "img"
CLASSES = ["rachadura", "cicatriz_deslizamento", "dano_estrutural"]
CORES = [(0, 0, 230), (0, 140, 255), (230, 160, 0)]
MODELOS = {"yolo11n": RAIZ / "runs" / "encosta_11n" / "weights" / "best.pt", "yolo11s": RAIZ / "runs" / "encosta_11s" / "weights" / "best.pt"}


def medir_fps(modelo: YOLO, imagens: list[Path], device: str, n: int = 60) -> float:
    frames = [cv2.imread(str(p)) for p in imagens[:20]]
    for f in frames[:5]:
        modelo.predict(f, device=device, verbose=False, imgsz=640)
    t0 = time.perf_counter()
    for i in range(n):
        modelo.predict(frames[i % len(frames)], device=device, verbose=False, imgsz=640)
    if device != "cpu":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gabarito(p: Path, W: int, H: int) -> list[tuple[int, np.ndarray]]:
    txt = DADOS / "labels" / "test" / (p.stem + ".txt")
    out = []
    for linha in txt.read_text().splitlines() if txt.exists() else []:
        c, x, y, w, h = map(float, linha.split())
        out.append((int(c), np.array([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])))
    return out


def analisar_imagem(modelo: YOLO, p: Path, conf: float = 0.35) -> dict:
    img = cv2.imread(str(p))
    H, W = img.shape[:2]
    r = modelo.predict(img, conf=conf, verbose=False, imgsz=640)[0]
    preds = [(int(c), b, float(s)) for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    gts = gabarito(p, W, H)
    usados, tp, fp = set(), 0, []
    for c, b, s in sorted(preds, key=lambda t: -t[2]):
        melhor, j_m = 0.0, -1
        for j, (cg, bg) in enumerate(gts):
            if j not in usados and cg == c and (v := iou(b, bg)) > melhor:
                melhor, j_m = v, j
        if melhor >= 0.5:
            usados.add(j_m)
            tp += 1
        else:
            fp.append((c, b, s))
    fn = [g for j, g in enumerate(gts) if j not in usados]
    return {"img": img, "preds": preds, "gts": gts, "tp": tp, "fp": fp, "fn": fn, "path": p}


def desenhar_caso(caso: dict, legenda: str) -> np.ndarray:
    img = caso["img"].copy()
    for c, b in caso["gts"]:  # gabarito em verde tracejado
        x0, y0, x1, y1 = map(int, b)
        for k in range(x0, x1, 12):
            cv2.line(img, (k, y0), (min(k + 6, x1), y0), (60, 200, 60), 2)
            cv2.line(img, (k, y1), (min(k + 6, x1), y1), (60, 200, 60), 2)
        for k in range(y0, y1, 12):
            cv2.line(img, (x0, k), (x0, min(k + 6, y1)), (60, 200, 60), 2)
            cv2.line(img, (x1, k), (x1, min(k + 6, y1)), (60, 200, 60), 2)
    for c, b, s in caso["preds"]:
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(img, (x0, y0), (x1, y1), CORES[c], 3)
        t = f"{CLASSES[c]} {s:.0%}"
        yb = y0 if y0 >= 22 else y0 + 22  # caixa colada no topo: rótulo vai para dentro
        cv2.rectangle(img, (x0, yb - 22), (x0 + 11 * len(t), yb), CORES[c], -1)
        cv2.putText(img, t, (x0 + 3, yb - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.resize(img, (420, int(420 * img.shape[0] / img.shape[1])))
    faixa = np.full((34, img.shape[1], 3), 25, np.uint8)
    cv2.putText(faixa, legenda, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, faixa])


def grade(tiles: list[np.ndarray], colunas: int = 3) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    while len(tiles) % colunas:
        tiles.append(np.full_like(tiles[0], 255))
    linhas = [np.hstack(tiles[i : i + colunas]) for i in range(0, len(tiles), colunas)]
    return np.vstack(linhas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--escolher", default="auto", choices=["auto", "n", "s"])
    args = ap.parse_args()
    AVAL.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    testes = sorted((DADOS / "images" / "test").glob("*.jpg"))

    resultados = {}
    for nome, peso in MODELOS.items():
        if not peso.exists():
            print(f"[pulando] {nome}: {peso} não existe")
            continue
        m = YOLO(str(peso))
        v = m.val(data=str(YAML), split="test", imgsz=640, batch=16, conf=0.001, iou=0.6, plots=True,
                  project=str(RAIZ / "runs" / "teste"), name=nome, exist_ok=True, verbose=False)
        por_classe = {}
        for i, c in enumerate(CLASSES):
            p, r, ap50, ap = v.box.class_result(i)
            por_classe[c] = {"precisao": round(float(p), 3), "recall": round(float(r), 3), "mAP50": round(float(ap50), 3), "mAP50_95": round(float(ap), 3)}
        fps_gpu = medir_fps(m, testes, "0") if torch.cuda.is_available() else None
        fps_cpu = medir_fps(m, testes, "cpu", n=30)
        resultados[nome] = {
            "pesos_mb": round(peso.stat().st_size / 1e6, 1),
            "parametros_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2),
            "geral": {"precisao": round(float(v.box.mp), 3), "recall": round(float(v.box.mr), 3), "mAP50": round(float(v.box.map50), 3), "mAP50_95": round(float(v.box.map), 3)},
            "por_classe": por_classe,
            "fps_gpu_rtx3050": round(fps_gpu, 1) if fps_gpu else None,
            "fps_cpu": round(fps_cpu, 1),
        }
        pasta = RAIZ / "runs" / "teste" / nome
        for arq in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"):
            if (pasta / arq).exists():
                shutil.copy(pasta / arq, IMG / f"{nome}_{arq}")
        treino = RAIZ / "runs" / ("encosta_11n" if nome == "yolo11n" else "encosta_11s")
        if (treino / "results.png").exists():
            shutil.copy(treino / "results.png", IMG / f"{nome}_curvas_treino.png")
        print(nome, json.dumps(resultados[nome], ensure_ascii=False))

    # Escolha: o "s" só vale a pena se ganhar >= 0,03 de mAP50-95; senão fica o "n" (mais leve, roda em CPU)
    if args.escolher == "auto":
        n, s = resultados.get("yolo11n"), resultados.get("yolo11s")
        escolhido = "yolo11s" if (s and n and s["geral"]["mAP50_95"] - n["geral"]["mAP50_95"] >= 0.03) or (s and not n) else "yolo11n"
    else:
        escolhido = f"yolo11{args.escolher}"

    (AVAL / "metricas.json").write_text(json.dumps({"split": "test", "imagens_teste": len(testes), "modelos": resultados, "escolhido": escolhido}, indent=2, ensure_ascii=False), encoding="utf-8")

    linhas = ["| Métrica (teste) | " + " | ".join(resultados) + " |", "|---|" + "---|" * len(resultados)]
    for k, rot in [("precisao", "Precisão"), ("recall", "Recall"), ("mAP50", "mAP@50"), ("mAP50_95", "mAP@50-95")]:
        linhas.append(f"| {rot} | " + " | ".join(f"{r['geral'][k]:.3f}" for r in resultados.values()) + " |")
    linhas.append("| FPS GPU (RTX 3050) | " + " | ".join(str(r["fps_gpu_rtx3050"]) for r in resultados.values()) + " |")
    linhas.append("| FPS CPU | " + " | ".join(str(r["fps_cpu"]) for r in resultados.values()) + " |")
    linhas.append("| Parâmetros (M) | " + " | ".join(str(r["parametros_M"]) for r in resultados.values()) + " |")
    linhas.append("| Peso (MB) | " + " | ".join(str(r["pesos_mb"]) for r in resultados.values()) + " |")
    linhas += ["", f"**Modelo escolhido para a API:** `{escolhido}`", "", "### Por classe (modelo escolhido)", "", "| Classe | Precisão | Recall | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|"]
    for c, r in resultados[escolhido]["por_classe"].items():
        linhas.append(f"| {c} | {r['precisao']:.3f} | {r['recall']:.3f} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} |")
    (AVAL / "comparacao.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))

    # Exporta o modelo escolhido para a API
    (RAIZ / "models").mkdir(exist_ok=True)
    origem = MODELOS[escolhido]
    shutil.copy(origem, RAIZ / "models" / "encosta_visao.pt")
    meta = {
        "versao": f"encosta-visao-{escolhido}-v1",
        "arquitetura": escolhido,
        "classes": CLASSES,
        "treinado_em": date.today().isoformat(),
        "metricas_teste": resultados[escolhido]["geral"],
        "dataset": "data/encosta (ver configs/dataset_resumo.json)",
    }
    (RAIZ / "models" / "encosta_visao.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Exemplos de acerto e de erro com o modelo escolhido
    m = YOLO(str(origem))
    casos = [analisar_imagem(m, p) for p in testes]
    acertos = [c for c in casos if c["gts"] and not c["fp"] and not c["fn"]]
    erros = [c for c in casos if c["fp"] or c["fn"]]
    rng = np.random.default_rng(7)

    def por_classe(lista: list[dict], k: int) -> list[dict]:
        sel, vistos = [], set()
        for cls in range(len(CLASSES)):
            cand = [c for c in lista if any(g[0] == cls for g in c["gts"]) or any(f[0] == cls for f in c["fp"])]
            rng.shuffle(cand)
            novos = [c for c in cand if id(c) not in vistos][:k]
            vistos.update(id(c) for c in novos)
            sel += novos
        return sel

    tiles_ok = [desenhar_caso(c, f"ACERTO | caixas corretas: {c['tp']}") for c in por_classe(acertos, 2)]
    tiles_err = []
    for c in por_classe(erros, 2):
        partes = []
        if c["fp"]:
            partes.append(f"falso positivo: {len(c['fp'])}")
        if c["fn"]:
            partes.append(f"perdido: {len(c['fn'])}")
        tiles_err.append(desenhar_caso(c, "ERRO | " + " | ".join(partes)))
    if tiles_ok:
        cv2.imwrite(str(IMG / "acertos.jpg"), grade(tiles_ok))
    if tiles_err:
        cv2.imwrite(str(IMG / "erros.jpg"), grade(tiles_err))
    resumo_erros = {
        "imagens_sem_erro": len([c for c in casos if not c["fp"] and not c["fn"]]),
        "imagens_com_falso_positivo": len([c for c in casos if c["fp"]]),
        "imagens_com_falso_negativo": len([c for c in casos if c["fn"]]),
        "falsos_positivos_por_classe": {CLASSES[k]: sum(1 for c in casos for f in c["fp"] if f[0] == k) for k in range(len(CLASSES))},
        "falsos_negativos_por_classe": {CLASSES[k]: sum(1 for c in casos for f in c["fn"] if f[0] == k) for k in range(len(CLASSES))},
        "limiar_confianca": 0.35,
        "criterio": "IoU >= 0.5 com caixa do gabarito da mesma classe",
    }
    (AVAL / "analise_erros.json").write_text(json.dumps(resumo_erros, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo_erros, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
