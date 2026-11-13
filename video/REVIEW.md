# REVIEW · Encosta Viva · Visão (vídeo de demonstração, 86 s)

## Decisões criativas
- Roteiro A ("uma foto, da encosta até a Defesa Civil"): a foto real 05 do teste atravessa o filme. O vídeo foi pedido direto, então segui a recomendação sem parar para escolha. B (antes e depois) e C (zoom sem fim) estão em STORYLINES.md.
- Tudo o que aparece vem do projeto: coordenadas da caixa e JSON (assets/resposta_api.json), clipe com o detector rodando em cada quadro (demo/gerar_clipe_camera.py, 21,5 FPS medidos), tela real da API, métricas de avaliacao/metricas.json e um erro real do teste.
- Sem narração (não foi pedida). Os títulos carregam o texto.

## O que foi medido e o que só foi visto
- `hyperframes check`: ok, sem erro nem aviso (lint, runtime, layout, movimento, contraste, craft, áudio).
- Visto em folhas de contato: os 12 quadros de QA da SPEC e as 6 emendas entre cenas.
- Som: os efeitos (Pixabay, embutidos) foram posicionados pelo pico medido com `hyperframes audio`. **Ninguém ouviu o resultado.**
- Trilha: o catálogo HeyGen exige a CLI `heygen`, que não roda no Windows. A trilha foi gerada localmente com MusicGen medium (scripts/gerar_trilha.py, seed 7; pesos CC BY-NC 4.0, uso acadêmico). São 4 trechos encadeados por continuação; nas emendas medimos diferenças de 0,7 a 3 dB e nenhuma descontinuidade. Normalizada a −17 LUFS, mixada em volume 0,7.
- Mix final medido: −16 LUFS integrado, pico real de −1,0 dBTP, zero clipping. Pops de 3,8 a 4,8 dB acima da trilha; whooshes e impacto de 4 a 9 dB acima.

## Ponto mais fraco
- As fotos públicas têm de 480 a 640 px e aparecem ampliadas 2,25x (palco 1080 px). Ficam macias. Com mais tempo: fotos da maquete em alta resolução.
- A régua de risco (cena exemplo, 31–38 s) é pequena em relação ao quadro. Daria para ocupar mais a coluna.

## Avisos mantidos de propósito
- Nenhum.
