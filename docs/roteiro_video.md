# Roteiro do vídeo · Applied Computer Vision (até 5 pontos)

**Duração-alvo:** 4 a 5 min. **Formato:** YouTube não listado (testar o link em janela anônima).
**Regra do enunciado:** o vídeo mostra o detector rodando **em tempo real** na maquete e em fotos reais.

## Antes de gravar (checklist)

- [ ] `.venv` ativo e `models/encosta_visao.pt` presente (`python -m pytest -q` passando)
- [ ] Maquete da encosta (a mesma do Maker Lab) com: muro de isopor ou papelão com rachadura desenhada ou riscada, uma "cicatriz" de terra exposta e um pedaço de "parede" quebrada
- [ ] Celular com **IP Webcam** (Android) ou **DroidCam** na mesma rede Wi-Fi, ou a webcam do notebook
- [ ] Fotos impressas ou abertas em outro celular: 2 rachaduras, 1 deslizamento e 1 parede muito danificada (pastas `data/encosta/images/test/`)
- [ ] Servidor da API no ar: `uvicorn encosta_visao.api:app --app-dir src --port 8001`
- [ ] OBS ou Xbox Game Bar gravando a tela, com o rosto de quem apresenta em um canto

## Sequência

| Tempo | O que aparece | Fala (resumo) |
|---|---|---|
| 0:00–0:30 | Integrantes na câmera e o título | "Somos a equipe do Encosta Viva. Quem mora em encosta percebe os sinais antes de todo mundo: uma rachada no muro, uma árvore que tombou. Só que esse aviso não chega à Defesa Civil a tempo." |
| 0:30–1:00 | Diagrama do pipeline (README, seção 4) | "O componente Visão é um YOLO11 treinado por nós para três sinais: rachadura, cicatriz de deslizamento e dano estrutural grave." |
| 1:00–2:15 | **Tempo real na maquete:** `python demo/tempo_real.py --fonte http://IP-DO-CELULAR:8080/video` | Passar a câmera pelo muro rachado e pela cicatriz. Mostrar o **FPS** e o **nível sugerido** mudando no topo. Aproximar a mão ou o rosto para mostrar o **borrão automático** (LGPD). |
| 2:15–3:00 | **Fotos reais:** apontar a câmera para as fotos impressas, ou rodar `--fonte pasta_de_fotos` | "Em foto real de parede o modelo marca a rachadura com X % de confiança; aqui, a cicatriz de um deslizamento visto de longe." |
| 3:00–3:45 | **Integração:** navegador em `http://localhost:8001/`, upload de uma foto de celular, e depois `/docs` com o JSON | Mostrar no JSON `deteccoes`, `versao` do modelo, `sugestao_risco.requer_confirmacao_humana = true` e `privacidade.gps_descartado = true`. "Esse é o contrato `POST /visao/detectar` que a API e a automação consomem." |
| 3:45–4:30 | `avaliacao/comparacao.md` e `docs/img/erros.jpg` | Métricas no teste (mAP@50, recall), yolo11n vs yolo11s e **um erro real**: "aqui o modelo confundiu a junta do reboco com rachadura. Por isso a regra exige confirmação humana." |
| 4:30–5:00 | Integrantes | "A IA sugere, a Defesa Civil decide. Repositório e relatório no link da descrição." |

## Comandos prontos

```bash
# webcam do notebook
python demo/tempo_real.py
# celular com IP Webcam
python demo/tempo_real.py --fonte http://192.168.0.10:8080/video
# gravar um MP4 de apoio a partir de fotos (sem janela)
python demo/tempo_real.py --fonte docs/demo_fotos --salvar docs/video/demo_fotos.mp4 --sem-janela
```

## Dicas

- Luz boa e câmera parada por 2 s em cada sinal: o nível sugerido usa o quadro atual.
- Se o FPS cair no notebook sem GPU, use `--conf 0.4` e feche o navegador durante a parte em tempo real.
- Não mostre rosto de morador real nem endereço; na maquete, use nomes de setor fictícios (ex.: `SSB-R4-012`).
