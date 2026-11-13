"""Testes que não dependem do modelo treinado (rodam em qualquer máquina)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encosta_visao.privacidade import anonimizar, carregar_sem_exif, codificar_jpeg  # noqa: E402
from encosta_visao.regras import sugerir_nivel  # noqa: E402


def _jpeg_com_gps() -> bytes:
    img = Image.fromarray(np.full((120, 160, 3), 128, np.uint8))
    exif = Image.Exif()
    exif[0x010F] = "MarcaCelular"  # Make
    exif[0x8825] = {1: "S", 2: (23.0, 33.0, 0.0), 3: "W", 4: (46.0, 38.0, 0.0)}  # GPSInfo
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_exif_e_gps_sao_removidos():
    original = _jpeg_com_gps()
    assert len(Image.open(io.BytesIO(original)).getexif()) > 0
    img, tinha_exif, tinha_gps = carregar_sem_exif(original)
    assert tinha_exif and tinha_gps
    limpo = codificar_jpeg(img)
    assert len(Image.open(io.BytesIO(limpo)).getexif()) == 0


def test_orientacao_exif_aplicada_antes_de_descartar():
    img = Image.fromarray(np.zeros((100, 200, 3), np.uint8))  # paisagem
    exif = Image.Exif()
    exif[0x0112] = 6  # girar 90°
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    arr, _, _ = carregar_sem_exif(buf.getvalue())
    assert arr.shape[:2] == (200, 100)  # ficou em pé


def test_anonimizar_sem_rosto():
    res = anonimizar(_jpeg_com_gps())
    assert res.rostos_borrados == 0 and res.tinha_gps


def test_rostos_reais_sao_borrados():
    """Usa a foto de exemplo com pessoas que acompanha o pacote ultralytics."""
    import cv2
    import pytest

    ultralytics = pytest.importorskip("ultralytics")
    foto = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
    if not foto.exists():
        pytest.skip("asset zidane.jpg ausente")
    from encosta_visao.privacidade import borrar_rostos

    img = cv2.imread(str(foto))
    out, n = borrar_rostos(img)
    assert n >= 2
    assert not np.array_equal(out, img)  # pixels alterados nas regiões de rosto


def test_regra_sem_deteccao_observacao():
    s = sugerir_nivel([])
    assert s.nivel_sugerido == "Observação" and not s.abrir_vistoria


def test_regra_rachadura_alta_vira_alerta_com_humano():
    s = sugerir_nivel([{"classe": "rachadura", "confianca": 0.82}])
    assert s.nivel_sugerido == "Alerta"
    assert s.requer_confirmacao_humana and s.abrir_vistoria


def test_regra_confianca_moderada_vira_atencao():
    s = sugerir_nivel([{"classe": "cicatriz_deslizamento", "confianca": 0.45}])
    assert s.nivel_sugerido == "Atenção"


def test_regra_ignora_ruido():
    s = sugerir_nivel([{"classe": "rachadura", "confianca": 0.20}])
    assert s.nivel_sugerido == "Observação"


def test_visao_nunca_sugere_alerta_maximo():
    dets = [{"classe": c, "confianca": 0.99} for c in ("rachadura", "cicatriz_deslizamento", "dano_estrutural")]
    assert sugerir_nivel(dets).nivel_sugerido != "Alerta máximo"
