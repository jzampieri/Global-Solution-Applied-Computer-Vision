"""Gera a trilha do vídeo localmente com MusicGen (facebook/musicgen-medium) na GPU.

O modelo gera ~30 s por vez; para cobrir o filme, cada novo trecho continua a partir dos
últimos 10 s do anterior (continuação com áudio de referência), sem emendas em loop.

Licença: os pesos do MusicGen são CC BY-NC 4.0 (uso não comercial, ok para o trabalho acadêmico).

Uso (no .venv do projeto applied_computer_vision):
    python video/scripts/gerar_trilha.py --segundos 92 --saida video/assets/bgm/trilha.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

PROMPT = (
    "calm documentary underscore, soft felt piano and warm analog synth pads, steady gentle pulse, "
    "subtle low percussion, hopeful and serious, 95 bpm, instrumental, no vocals, clean modern mix"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segundos", type=float, default=92)
    ap.add_argument("--saida", default="video/assets/bgm/trilha.wav")
    ap.add_argument("--modelo", default="facebook/musicgen-medium")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.modelo)
    modelo = MusicgenForConditionalGeneration.from_pretrained(
        args.modelo, torch_dtype=torch.float16 if dev == "cuda" else torch.float32
    ).to(dev)
    sr = modelo.config.audio_encoder.sampling_rate
    fps_tokens = modelo.config.audio_encoder.frame_rate  # ~50 tokens por segundo

    def gerar(segundos: float, contexto: np.ndarray | None) -> np.ndarray:
        kw = dict(text=[args.prompt], padding=True, return_tensors="pt")
        if contexto is not None:
            kw.update(audio=contexto, sampling_rate=sr)
        entradas = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in proc(**kw).items()}
        if "input_values" in entradas and dev == "cuda":
            entradas["input_values"] = entradas["input_values"].half()
        out = modelo.generate(**entradas, do_sample=True, guidance_scale=3.0,
                              max_new_tokens=int(segundos * fps_tokens))
        return out[0, 0].float().cpu().numpy()

    print("trecho 1 (30 s)...", flush=True)
    faixa = gerar(30, None)
    while len(faixa) / sr < args.segundos:
        contexto = faixa[-10 * sr:]
        print(f"continuando a partir de {len(faixa) / sr:.1f} s...", flush=True)
        novo = gerar(25, contexto)
        # a saída inclui o áudio de contexto: fica só com a parte nova
        faixa = np.concatenate([faixa, novo[len(contexto):]])

    faixa = faixa[: int(args.segundos * sr)]
    faixa = faixa / max(1e-6, float(np.abs(faixa).max())) * 0.89  # pico em -1 dBFS
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.saida, faixa, sr)
    print(f"ok: {args.saida} ({len(faixa) / sr:.1f} s, {sr} Hz)")


if __name__ == "__main__":
    main()
