"""LGPD na Visão: a foto do morador nunca é salva como chegou.

Antes de qualquer outra etapa:
  1. a imagem é redecodificada a partir dos pixels, o que descarta EXIF (GPS, modelo do celular, data);
  2. rostos detectados são pixelizados com o YuNet (detector DNN do OpenCV Zoo, 230 KB, MIT,
     roda offline em CPU). Se o arquivo ONNX faltar, cai para o Haar cascade do OpenCV;
  3. só o setor de risco informado pelo morador acompanha a imagem.

Por que YuNet e não Haar (medido em 422 fotos de val/teste sem pessoas + 2 fotos com pessoas):
  Haar (minNeighbors=6): 66 fotos com "rosto" falso borrado (16%), 2 de 4 rostos reais achados
  YuNet (score 0.8):      0 fotos com rosto falso,                 4 de 4 rostos reais achados
Um falso rosto borrado pode apagar justamente a rachadura que o modelo precisa ver.

Limitação documentada: placas de veículo não são detectadas por este módulo
(não há detector aberto e embarcado no OpenCV). O próximo passo é um detector de placas
treinado ou o desfoque manual pelo agente antes de arquivar a evidência.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

YUNET = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"
_CASCADE = None
_YUNET = None


def _cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        caminho = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(caminho)
    return _CASCADE


def _yunet():
    global _YUNET
    if _YUNET is None and YUNET.exists():
        _YUNET = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8, 0.3, 50)
    return _YUNET


def detector_de_rosto() -> str:
    return "yunet" if _yunet() is not None else "haar"


def _achar_rostos(pequena: np.ndarray) -> list[tuple[int, int, int, int]]:
    yn = _yunet()
    if yn is not None:
        yn.setInputSize((pequena.shape[1], pequena.shape[0]))
        _, r = yn.detect(pequena)
        return [] if r is None else [tuple(int(v) for v in f[:4]) for f in r]
    cinza = cv2.equalizeHist(cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY))
    lado_min = max(20, min(pequena.shape[:2]) // 25)
    return [tuple(map(int, f)) for f in _cascade().detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=6, minSize=(lado_min, lado_min))]


@dataclass
class ResultadoPrivacidade:
    imagem_bgr: np.ndarray
    exif_removido: bool
    tinha_gps: bool
    rostos_borrados: int


def carregar_sem_exif(conteudo: bytes) -> tuple[np.ndarray, bool, bool]:
    """Decodifica bytes de imagem, aplica a orientação do EXIF e descarta todos os metadados.

    Retorna (imagem BGR, exif_existia, gps_existia).
    """
    with Image.open(io.BytesIO(conteudo)) as img:
        exif = img.getexif()
        tinha_exif = len(exif) > 0
        tinha_gps = 0x8825 in exif  # tag GPSInfo
        img = ImageOps.exif_transpose(img)  # mantém a foto "em pé" antes de jogar o EXIF fora
        rgb = img.convert("RGB")
        # Copiar só os pixels: o novo array não carrega nenhum metadado.
        arr = np.asarray(rgb, dtype=np.uint8).copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), tinha_exif, tinha_gps


def borrar_rostos(imagem_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Borra rostos frontais. Devolve uma cópia e a quantidade de rostos borrados."""
    saida = imagem_bgr.copy()
    # Detecta numa cópia reduzida (lado maior 640 px): rápido e ainda acha rostos pequenos
    esc = min(1.0, 640 / max(saida.shape[:2]))
    pequena = cv2.resize(saida, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) if esc < 1 else saida
    rostos = [(int(x / esc), int(y / esc), int(w / esc), int(h / esc)) for (x, y, w, h) in _achar_rostos(pequena)]
    for (x, y, w, h) in rostos:
        # margem de 15% para cobrir cabelo/queixo
        mx, my = int(w * 0.15), int(h * 0.15)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(saida.shape[1], x + w + mx), min(saida.shape[0], y + h + my)
        regiao = saida[y0:y1, x0:x1]
        rh, rw = regiao.shape[:2]
        # Pixeliza (8x8 blocos) e suaviza: irreversível e barato mesmo em rostos grandes
        mini = cv2.resize(regiao, (max(1, rw // max(1, rw // 8)), max(1, rh // max(1, rh // 8))), interpolation=cv2.INTER_AREA)
        saida[y0:y1, x0:x1] = cv2.blur(cv2.resize(mini, (rw, rh), interpolation=cv2.INTER_LINEAR), (5, 5))
    return saida, len(rostos)


def anonimizar(conteudo: bytes) -> ResultadoPrivacidade:
    """Pipeline completo de LGPD aplicado a uma foto recebida."""
    img, tinha_exif, tinha_gps = carregar_sem_exif(conteudo)
    img, n = borrar_rostos(img)
    return ResultadoPrivacidade(
        imagem_bgr=img, exif_removido=tinha_exif, tinha_gps=tinha_gps, rostos_borrados=n
    )


def codificar_jpeg(imagem_bgr: np.ndarray, qualidade: int = 85) -> bytes:
    """JPEG sem metadados (cv2.imencode não grava EXIF)."""
    ok, buf = cv2.imencode(".jpg", imagem_bgr, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    if not ok:
        raise ValueError("falha ao codificar JPEG")
    return buf.tobytes()
