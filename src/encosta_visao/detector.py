"""Carrega o modelo YOLO treinado e converte a saída para o contrato do Encosta Viva."""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
MODELO_PADRAO = RAIZ / "models" / "encosta_visao.pt"
META_PADRAO = RAIZ / "models" / "encosta_visao.json"

# Cores BGR fixas por classe (alto contraste, legíveis para daltonismo comum)
CORES = {
    "rachadura": (0, 0, 230),  # vermelho
    "cicatriz_deslizamento": (0, 140, 255),  # laranja
    "dano_estrutural": (230, 160, 0),  # azul
}
ROTULOS_PT = {
    "rachadura": "Rachadura",
    "cicatriz_deslizamento": "Cicatriz de deslizamento",
    "dano_estrutural": "Dano estrutural grave",
}


class Detector:
    def __init__(self, caminho_modelo: str | os.PathLike | None = None, dispositivo: str | None = None):
        from ultralytics import YOLO  # import tardio: os testes de regra não precisam do torch

        # O Ultralytics faz cv2.setNumThreads(0) ao ser importado; isso deixava o borrão de
        # rostos (Haar) 4-5x mais lento. Devolvemos até 8 threads ao OpenCV.
        cv2.setNumThreads(min(8, os.cpu_count() or 1))

        caminho = Path(caminho_modelo or os.getenv("ENCOSTA_MODELO", MODELO_PADRAO))
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {caminho}. Rode scripts/03_treinar.py ou baixe o peso da release."
            )
        self.caminho = caminho
        self.modelo = YOLO(str(caminho))
        self.nomes: dict[int, str] = dict(self.modelo.names)
        self.dispositivo = dispositivo or os.getenv("ENCOSTA_DISPOSITIVO") or None
        meta_path = caminho.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.versao = self.meta.get("versao", caminho.stem)

    def detectar(self, imagem_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5, imgsz: int = 640) -> tuple[list[dict], float]:
        """Roda o modelo e devolve (detecções, ms de inferência)."""
        h, w = imagem_bgr.shape[:2]
        t0 = time.perf_counter()
        res = self.modelo.predict(
            imagem_bgr, conf=conf, iou=iou, imgsz=imgsz, device=self.dispositivo, verbose=False
        )[0]
        ms = (time.perf_counter() - t0) * 1000
        saida = []
        for caixa, c, cls in zip(
            res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)
        ):
            x1, y1, x2, y2 = [float(v) for v in caixa]
            classe = self.nomes[int(cls)]
            saida.append(
                {
                    "classe": classe,
                    "rotulo": ROTULOS_PT.get(classe, classe),
                    "confianca": round(float(c), 4),
                    "caixa": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
                    "caixa_normalizada": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                }
            )
        saida.sort(key=lambda d: -d["confianca"])
        return saida, ms


def desenhar(imagem_bgr: np.ndarray, deteccoes: list[dict], rodape: str | None = None) -> np.ndarray:
    """Desenha caixas e rótulos em português sobre uma cópia da imagem."""
    out = imagem_bgr.copy()
    esp = max(2, round(min(out.shape[:2]) / 300))
    escala = max(0.5, min(out.shape[:2]) / 900)
    for d in deteccoes:
        c = d["caixa"]
        p1, p2 = (int(c["x1"]), int(c["y1"])), (int(c["x2"]), int(c["y2"]))
        cor = CORES.get(d["classe"], (255, 255, 255))
        cv2.rectangle(out, p1, p2, cor, esp)
        txt = f"{d['rotulo']} {d['confianca']:.0%}"
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, max(1, esp - 1))
        y_txt = max(th + base + 2, p1[1])
        cv2.rectangle(out, (p1[0], y_txt - th - base - 4), (p1[0] + tw + 6, y_txt), cor, -1)
        cv2.putText(out, txt, (p1[0] + 3, y_txt - base - 2), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), max(1, esp - 1), cv2.LINE_AA)
    if rodape:
        (tw, th), base = cv2.getTextSize(rodape, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
        cv2.rectangle(out, (0, out.shape[0] - th - base - 10), (tw + 12, out.shape[0]), (20, 20, 20), -1)
        cv2.putText(out, rodape, (6, out.shape[0] - base - 4), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@lru_cache(maxsize=1)
def detector_padrao() -> Detector:
    return Detector()
