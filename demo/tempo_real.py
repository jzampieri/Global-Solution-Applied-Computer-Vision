"""Detecção em tempo real: webcam, câmera do celular (IP Webcam / DroidCam) ou arquivo de vídeo.

Mostra as caixas, o FPS medido e o nível sugerido pela regra do Encosta Viva.
Rostos que aparecem na câmera são borrados antes de desenhar (LGPD também no vídeo).

Exemplos:
    python demo/tempo_real.py                                   # webcam 0
    python demo/tempo_real.py --fonte http://192.168.0.10:8080/video   # celular com IP Webcam
    python demo/tempo_real.py --fonte maquete.mp4 --salvar saida.mp4
    python demo/tempo_real.py --fonte docs/img/demo_fotos --salvar demo.mp4   # pasta de fotos -> vídeo

Teclas: Q ou ESC sai · ESPAÇO pausa · S salva o quadro atual em capturas/
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from encosta_visao.detector import Detector, desenhar  # noqa: E402
from encosta_visao.privacidade import borrar_rostos  # noqa: E402
from encosta_visao.regras import sugerir_nivel  # noqa: E402

COR_NIVEL = {"Observação": (90, 160, 40), "Atenção": (0, 170, 230), "Alerta": (40, 40, 220)}


def fonte_quadros(fonte: str, segundos_por_foto: float, fps_saida: float):
    """Gera quadros de webcam, URL, vídeo ou pasta de imagens (slideshow)."""
    p = Path(fonte)
    if p.is_dir():
        fotos = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png")])
        if not fotos:
            raise SystemExit(f"Nenhuma imagem em {p}")
        repet = max(1, int(segundos_por_foto * fps_saida))
        for f in fotos:
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w = img.shape[:2]
            esc = 960 / max(h, w)
            img = cv2.resize(img, (int(w * esc), int(h * esc)))
            for _ in range(repet):
                yield img
        return
    cap = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte de vídeo: {fonte}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def painel(frame: np.ndarray, sug: dict, fps: float, ms: float, versao: str) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    cor = COR_NIVEL.get(sug["nivel_sugerido"], (200, 200, 200))
    cv2.rectangle(out, (0, 0), (w, 44), (25, 25, 25), -1)
    cv2.rectangle(out, (0, 0), (12, 44), cor, -1)
    txt = f"Nivel sugerido: {sug['nivel_sugerido'].replace('ç', 'c').replace('ã', 'a')}"
    cv2.putText(out, txt, (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2, cv2.LINE_AA)
    info = f"{fps:5.1f} FPS | {ms:5.1f} ms | {versao}"
    (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(out, info, (w - tw - 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    rod = "IA sugere, Defesa Civil decide | rostos borrados"
    cv2.rectangle(out, (0, h - 28), (w, h), (25, 25, 25), -1)
    cv2.putText(out, rod, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default="0", help="índice da webcam, URL, arquivo de vídeo ou pasta de fotos")
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--salvar", default=None, help="grava o resultado em .mp4")
    ap.add_argument("--sem-janela", action="store_true", help="não abre janela (útil para gerar vídeo)")
    ap.add_argument("--segundos-por-foto", type=float, default=2.5)
    ap.add_argument("--sem-borrar", action="store_true", help="desliga o borrão de rosto (só para medir FPS puro)")
    args = ap.parse_args()

    det = Detector(args.modelo)
    gravador = None
    tempos: deque[float] = deque(maxlen=30)
    pausado = False
    frame_saida = None
    Path(RAIZ / "capturas").mkdir(exist_ok=True)
    fps_video = 15.0

    for frame in fonte_quadros(args.fonte, args.segundos_por_foto, fps_video):
        if not pausado:
            t0 = time.perf_counter()
            if not args.sem_borrar:
                frame, _ = borrar_rostos(frame)
            dets, ms = det.detectar(frame, conf=args.conf)
            sug = sugerir_nivel(dets).como_dict()
            tempos.append(time.perf_counter() - t0)
            fps = len(tempos) / sum(tempos)
            frame_saida = painel(desenhar(frame, dets), sug, fps, ms, det.versao)
            if args.salvar:
                if gravador is None:
                    h, w = frame_saida.shape[:2]
                    gravador = cv2.VideoWriter(args.salvar, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))
                gravador.write(frame_saida)
        if not args.sem_janela:
            cv2.imshow("Encosta Viva - Visao (Q para sair)", frame_saida)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                pausado = not pausado
            if k == ord("s"):
                nome = RAIZ / "capturas" / f"captura_{int(time.time())}.jpg"
                cv2.imwrite(str(nome), frame_saida)
                print("salvo:", nome)

    if gravador:
        gravador.release()
        print("vídeo salvo em", args.salvar)
    if tempos:
        print(f"FPS médio (últimos {len(tempos)} quadros, incluindo borrão de rosto): {len(tempos) / sum(tempos):.1f}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
