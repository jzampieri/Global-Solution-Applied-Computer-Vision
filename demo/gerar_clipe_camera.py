"""Gera um clipe de demonstração com uma câmera simulada sobre fotos reais do teste.

Cada quadro é um recorte que se move e aproxima sobre a foto (como um celular percorrendo
a cena). O detector real roda em CADA quadro, com borrão de rostos e regra de risco, e o
FPS exibido é o medido no processamento. Usado no vídeo de demonstração (video/).

Uso:
    python demo/gerar_clipe_camera.py --saida video/assets/tempo_real.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "demo"))

from encosta_visao.detector import Detector, desenhar  # noqa: E402
from encosta_visao.privacidade import borrar_rostos  # noqa: E402
from encosta_visao.regras import sugerir_nivel  # noqa: E402
from tempo_real import painel  # noqa: E402

# (foto, zoom inicial, zoom final, centro inicial, centro final) em coordenadas normalizadas
PERCURSO = [
    ("05_landslides_00080.jpg", 1.00, 1.30, (0.50, 0.55), (0.52, 0.35)),
    ("01_building-cracks_00008.jpg", 1.05, 1.35, (0.40, 0.40), (0.55, 0.55)),
    ("06_landslides_00089.jpg", 1.00, 1.25, (0.50, 0.60), (0.50, 0.40)),
    ("08_wall-damage_00120.jpg", 1.30, 1.00, (0.45, 0.30), (0.50, 0.45)),
    ("03_building-cracks_00003.jpg", 1.00, 1.30, (0.60, 0.50), (0.45, 0.45)),
]


def suave(t: float) -> float:
    return t * t * (3 - 2 * t)


def quadro(img: np.ndarray, zoom: float, cx: float, cy: float, lado: int) -> np.ndarray:
    h, w = img.shape[:2]
    janela = min(h, w) / zoom
    x0 = np.clip(cx * w - janela / 2, 0, w - janela)
    y0 = np.clip(cy * h - janela / 2, 0, h - janela)
    M = np.float32([[lado / janela, 0, -x0 * lado / janela], [0, lado / janela, -y0 * lado / janela]])
    return cv2.warpAffine(img, M, (lado, lado), flags=cv2.INTER_CUBIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(RAIZ / "video" / "assets" / "tempo_real.mp4"))
    ap.add_argument("--segundos", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lado", type=int, default=1080)
    args = ap.parse_args()

    det = Detector()
    n = int(args.segundos * args.fps)
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    gravador = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.lado, args.lado))
    tempos: deque[float] = deque(maxlen=30)
    registro = []
    for nome, z0, z1, c0, c1 in PERCURSO:
        img = cv2.imread(str(RAIZ / "docs" / "demo_fotos" / nome))
        for i in range(n):
            t = suave(i / (n - 1))
            f = quadro(img, z0 + (z1 - z0) * t, c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, args.lado)
            t0 = time.perf_counter()
            f, _ = borrar_rostos(f)
            dets, ms = det.detectar(f, conf=0.35)
            sug = sugerir_nivel(dets).como_dict()
            tempos.append(time.perf_counter() - t0)
            fps = len(tempos) / sum(tempos)
            gravador.write(painel(desenhar(f, dets), sug, fps, ms, det.versao))
            registro.append({"foto": nome, "fps": round(fps, 1), "nivel": sug["nivel_sugerido"],
                             "deteccoes": [(d["classe"], d["confianca"]) for d in dets]})
    gravador.release()
    fps_medio = float(np.mean([r["fps"] for r in registro[30:]]))
    Path(args.saida).with_suffix(".json").write_text(
        json.dumps({"quadros": len(registro), "fps_medio": round(fps_medio, 1), "registro": registro[::15]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"{len(registro)} quadros, FPS médio medido {fps_medio:.1f} -> {args.saida}")


if __name__ == "__main__":
    main()
