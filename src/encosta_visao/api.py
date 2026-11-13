"""Serviço de Visão do Encosta Viva (FastAPI).

Contrato com o restante do produto (documento de concepção da equipe):
    POST /visao/detectar  -> classes, caixas, confiança e versão do modelo
Consumidores: API principal (POST /relatos) e Automação (fila de vistorias).

Rodar:
    uvicorn encosta_visao.api:app --app-dir src --port 8001
"""
from __future__ import annotations

import base64
import re
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import UnidentifiedImageError

from . import __version__
from .detector import desenhar, detector_padrao
from .privacidade import anonimizar, codificar_jpeg, detector_de_rosto
from .regras import sugerir_nivel

TAMANHO_MAX = 10 * 1024 * 1024  # 10 MB: foto de celular comum
TIPOS_ACEITOS = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
SETOR_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")

@asynccontextmanager
async def aquecer(_app: FastAPI):
    """Carrega o modelo e faz uma inferência de aquecimento (evita ~5 s na 1ª foto do cidadão)."""
    try:
        detector_padrao().detectar(np.zeros((640, 640, 3), np.uint8))
    except FileNotFoundError:
        pass  # /saude informa "sem_modelo"
    yield


app = FastAPI(
    lifespan=aquecer,
    title="Encosta Viva · Serviço de Visão",
    version=__version__,
    description=(
        "Detector de sinais de instabilidade de encosta em fotos de moradores. "
        "A IA sugere, a Defesa Civil decide: toda sugestão de Alerta exige confirmação humana."
    ),
)


@app.get("/saude")
def saude() -> dict:
    try:
        det = detector_padrao()
        return {"status": "ok", "modelo": det.versao}
    except FileNotFoundError as e:
        return {"status": "sem_modelo", "erro": str(e)}


@app.get("/visao/modelo")
def info_modelo() -> dict:
    det = detector_padrao()
    return {"versao": det.versao, "classes": list(det.nomes.values()), **det.meta}


@app.post("/visao/detectar")
async def detectar(
    foto: UploadFile = File(..., description="Foto JPEG/PNG/WebP de até 10 MB"),
    setor_id: str = Form("desconhecido", description="Setor de risco (ex.: SSB-R4-012). Único dado de local guardado."),
    confianca_minima: float = Form(0.25, ge=0.05, le=0.95),
    retornar_imagem: bool = Form(False, description="Devolve a imagem anonimizada e anotada em base64"),
) -> dict:
    if foto.content_type not in TIPOS_ACEITOS:
        raise HTTPException(415, f"Tipo não suportado: {foto.content_type}. Envie JPEG, PNG ou WebP.")
    if not SETOR_RE.match(setor_id):
        raise HTTPException(422, "setor_id inválido: use letras, números, '-' ou '_' (até 40).")
    conteudo = await foto.read(TAMANHO_MAX + 1)
    if len(conteudo) > TAMANHO_MAX:
        raise HTTPException(413, "Foto maior que 10 MB.")
    if not conteudo:
        raise HTTPException(422, "Arquivo vazio.")

    t0 = time.perf_counter()
    try:
        priv = anonimizar(conteudo)  # LGPD primeiro: EXIF fora e rostos borrados
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "Não consegui abrir a imagem. Tente outra foto.")

    det = detector_padrao()
    deteccoes, ms_inf = det.detectar(priv.imagem_bgr, conf=confianca_minima)
    sugestao = sugerir_nivel(deteccoes)

    por_classe: dict[str, int] = {}
    for d in deteccoes:
        por_classe[d["classe"]] = por_classe.get(d["classe"], 0) + 1

    resposta = {
        "id_analise": str(uuid.uuid4()),
        "setor_id": setor_id,
        "modelo": {"versao": det.versao, "arquitetura": det.meta.get("arquitetura")},
        "imagem": {"largura": priv.imagem_bgr.shape[1], "altura": priv.imagem_bgr.shape[0]},
        "deteccoes": deteccoes,
        "resumo": {"total": len(deteccoes), "por_classe": por_classe},
        "sugestao_risco": sugestao.como_dict(),
        "privacidade": {
            "exif_removido": True,
            "exif_existia": priv.exif_removido,
            "gps_descartado": priv.tinha_gps,
            "rostos_borrados": priv.rostos_borrados,
            "detector_rosto": detector_de_rosto(),
            "foto_original_armazenada": False,
        },
        "tempo_ms": {"inferencia": round(ms_inf, 1), "total": round((time.perf_counter() - t0) * 1000, 1)},
    }
    if retornar_imagem:
        anotada = desenhar(priv.imagem_bgr, deteccoes)
        resposta["imagem_anotada_jpeg_base64"] = base64.b64encode(codificar_jpeg(anotada)).decode()
    return resposta


PAGINA = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Encosta Viva · Visão</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mut:#5b5b57;--card:#fff;--line:#d9d7d0;--acc:#0b5d4b;--on-acc:#fff;--alert:#b3261e;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eceae4;--mut:#a5a39c;--card:#1f1f1d;--line:#34332f;--acc:#5ec4a8;--on-acc:#0d1f1a;--alert:#ff8a80;--warn:#f2c14e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:880px;margin:0 auto;padding:24px 16px}h1{font-size:1.6rem;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 20px}
form,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
label{display:block;font-weight:600;margin:10px 0 4px}input[type=text],input[type=file]{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit}
button{margin-top:14px;padding:12px 18px;border:0;border-radius:8px;background:var(--acc);color:var(--on-acc);font:600 1rem system-ui;cursor:pointer;min-height:44px}
button:focus-visible,input:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
img{max-width:100%;border-radius:8px;display:block}.nivel{font-size:1.3rem;font-weight:700}
.Alerta{color:var(--alert)}.Atenção{color:var(--warn)}.Observação{color:var(--acc)}
ul{padding-left:20px}small{color:var(--mut)}
</style></head><body><main>
<h1>Encosta Viva · Visão</h1>
<p class="sub">Envie a foto de uma parede, muro ou encosta. O sistema aponta sinais de instabilidade e sugere um nível. Quem decide é a Defesa Civil.</p>
<form id="f"><label for="foto">Foto</label><input id="foto" name="foto" type="file" accept="image/*" capture="environment" required>
<label for="setor">Setor de risco</label><input id="setor" name="setor_id" type="text" value="SSB-R4-012" autocomplete="off">
<button type="submit">Analisar foto</button>
<p><small>Antes da análise, a localização GPS da foto é apagada e rostos são borrados. A foto original não é guardada.</small></p></form>
<div id="out" aria-live="polite"></div>
</main><script>
const f=document.getElementById('f'),out=document.getElementById('out');
f.addEventListener('submit',async e=>{e.preventDefault();out.innerHTML='<div class="card">Analisando…</div>';
const fd=new FormData(f);fd.append('retornar_imagem','true');
try{const r=await fetch('/visao/detectar',{method:'POST',body:fd});const j=await r.json();
if(!r.ok){out.innerHTML='<div class="card">Erro: '+(j.detail||r.status)+'</div>';return}
const s=j.sugestao_risco;const esc=t=>String(t).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
out.innerHTML='<div class="card"><div class="nivel '+esc(s.nivel_sugerido)+'">Nível sugerido: '+esc(s.nivel_sugerido)+'</div><p>'+esc(s.motivo)+'</p>'+
(j.deteccoes.length?'<ul>'+j.deteccoes.map(d=>'<li>'+esc(d.rotulo)+': '+Math.round(d.confianca*100)+'%</li>').join('')+'</ul>':'')+
'<small>Modelo '+esc(j.modelo.versao)+' · '+j.tempo_ms.inferencia+' ms · rostos borrados: '+j.privacidade.rostos_borrados+'</small></div>'+
'<div class="card"><img alt="Foto anonimizada com as detecções marcadas" src="data:image/jpeg;base64,'+j.imagem_anotada_jpeg_base64+'"></div>';
}catch(err){out.innerHTML='<div class="card">Sem conexão com o serviço. Tente de novo.</div>'}});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def pagina() -> str:
    return PAGINA
