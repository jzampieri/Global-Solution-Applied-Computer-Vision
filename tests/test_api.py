"""Testes de integração da API (precisam de models/encosta_visao.pt; pulam se não existir)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

pytestmark = pytest.mark.skipif(not (RAIZ / "models" / "encosta_visao.pt").exists(), reason="modelo não treinado")


@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient

    from encosta_visao.api import app

    return TestClient(app)


def _foto_teste() -> bytes:
    pasta = RAIZ / "data" / "encosta" / "images" / "test"
    fotos = sorted(pasta.glob("building-cracks_*.jpg")) if pasta.exists() else []
    if fotos:
        return fotos[0].read_bytes()
    buf = io.BytesIO()
    Image.fromarray(np.full((320, 320, 3), 180, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_saude(cliente):
    r = cliente.get("/saude")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_contrato_detectar(cliente):
    r = cliente.post(
        "/visao/detectar",
        files={"foto": ("parede.jpg", _foto_teste(), "image/jpeg")},
        data={"setor_id": "SSB-R4-012", "retornar_imagem": "true"},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    # contrato combinado com API e Automação: classes, caixas, confiança e versão do modelo
    assert j["modelo"]["versao"]
    assert j["setor_id"] == "SSB-R4-012"
    for d in j["deteccoes"]:
        assert {"classe", "confianca", "caixa"} <= d.keys()
        assert 0 <= d["confianca"] <= 1
    assert j["sugestao_risco"]["nivel_sugerido"] in {"Observação", "Atenção", "Alerta"}
    assert j["privacidade"]["foto_original_armazenada"] is False
    assert j["imagem_anotada_jpeg_base64"]


def test_rejeita_arquivo_que_nao_e_imagem(cliente):
    r = cliente.post("/visao/detectar", files={"foto": ("x.jpg", b"isto nao e imagem", "image/jpeg")})
    assert r.status_code == 422


def test_rejeita_setor_invalido(cliente):
    r = cliente.post(
        "/visao/detectar",
        files={"foto": ("p.jpg", _foto_teste(), "image/jpeg")},
        data={"setor_id": "../../etc/passwd"},
    )
    assert r.status_code == 422
