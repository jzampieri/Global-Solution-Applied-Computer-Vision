# SPEC · Encosta Viva · Visão (vídeo de demonstração)

GOAL: o avaliador termina sabendo que o detector foi treinado sob medida, funciona de verdade
      (foto, câmera em tempo real e API) e que a decisão final é da Defesa Civil.
OUTPUT: HyperFrames, 1920 x 1080, 30 fps, pt-BR, sem narração (títulos carregam o texto).
TIME: 86 s = 2580 quadros. Música entra depois do login HeyGen; até lá, só efeitos.

LOOK:
- Fundo das cenas de texto: concreto #CFCAC2 com ruído fino 3% (SVG feTurbulence, estático).
- Traço #16130F para todo texto lido. Acento único: laranja de cicatriz #FF8C00 (a cor que o
  detector usa para essa classe), só na caixa, na etiqueta e na palavra "Alerta".
  Cores de classe do próprio detector: rachadura #E60000, dano estrutural #00A0E6.
- Display: Archivo Black, 104 px (9,6% da altura), tracking -1%. Linhas menores: Inter 700, 44 px.
  Texto do produto (JSON): JetBrains Mono 34 px.
- Meio: "foto sendo anotada". Fotos reais em tela cheia, sem moldura nem cartão flutuante.
  Etiqueta = retângulo cheio na cor da classe, texto branco Inter 700, igual à saída do detector.
- Luz e profundidade: nenhuma sombra, brilho ou vinheta. Escurecimento de foto só com camada
  #16130F a 55%.

OBJECTS:
- A foto da encosta (assets/encosta.jpg, real, 480 px): cicatriz laranja acima da casa de telhado
  vermelho com fita amarela. Recortada em 16:9 alinhada ao topo para manter a cicatriz.
- A caixa do detector: retângulo de 6 px nas coordenadas REAIS [0.3239, 0.0474, 0.7592, 0.4093].
- A régua de risco: trilho horizontal com três faixas (Observação, Atenção, Alerta), marca de
  limite em 0,35 e em 0,60 e um cursor que pousa em 0,78.
- O JSON real de assets/resposta_api.json (inferência de 12,8 ms).
- O clipe real assets/tempo_real.mp4 (1080 x 1080, detector em cada quadro, 21,5 FPS medidos).
- A tela real da API (assets/api.png) e o erro real (recorte de assets/erros.jpg).

MOTION:
- Chegada: expo.out 0,5 s. Câmera e cor: power2.inOut. Saída e zoom-through: power3.in 0,3 s.
- Títulos: blur 8→0 px e opacidade 0→1 em 0,18 s, por palavra a cada 0,07 s, sem deslocamento.
  Parados depois de pousar.
- Caixa: desenhada traço a traço (clip-path inset) em 0,45 s, power3.out. Etiqueta 0,1 s depois.
- Trocas: corte seco, inundação de cor, zoom pelo objeto, whip. Nunca crossfade entre mundos.

SHOTS (s):
1   0,0–6,0   foto em tela cheia, push 1,00→1,06; "Quem mora embaixo da encosta / vê o sinal primeiro".
2   6,0–12,0  foto escurece 55%; "Mas o alerta chega por cidade" → "e ninguém vê o terreno". Whip para fora.
3   12,0–19,0 concreto: "Encosta Viva · Visão" + "YOLO11 treinado para três sinais"; montagem de 3 recortes reais (0,7 s cada) com etiqueta de classe.
4   19,0–25,0 foto volta; etiquetas de metadado (GPS, data, modelo do celular) riscadas → "GPS apagado, rostos borrados".
5   25,0–31,0 caixa desenhada nas coordenadas reais; etiqueta "Cicatriz de deslizamento 78%" (pop); linha "O modelo marca o sinal".
6   31,0–38,0 a caixa encolhe e vira o cursor da régua: pousa em 0,78, passa a marca de 0,60, "Alerta" acende (riser e impacto). "Quem confirma é a Defesa Civil".
7   38,0–45,0 JSON real, linha a linha (tique de tecla): nivel_sugerido, requer_confirmacao_humana, foto_original_armazenada, inferencia 12,8 ms.
8   45,0–59,0 clipe real de tempo real em 1080x1080 à esquerda; à direita "Câmera em movimento" / "detector em cada quadro" / "21 FPS medidos".
9   59,0–64,0 tela real da API, pan vertical; "A mesma análise na página do produto".
10  64,0–72,0 número que pousa "1.393 fotos públicas"; barras mAP@50 por classe: 0,88 · 0,75 · 0,47, com "ainda o ponto fraco" em cicatriz.
11  72,0–78,0 erro real (grades marcadas como dano, rachadura perdida); "Ela erra" → "por isso ela só sugere".
12  78,0–86,0 "A IA sugere. / A Defesa Civil decide." Depois, nome, integrantes e RMs. Segura 3 s.

BANNED: fundo escuro vazio com coisas pequenas no meio; brilho, glow, partículas; gradientes roxos
ou azuis; cartões de vidro; papel creme com serifa; rótulos "01/02"; tudo entrando com fade e
subida; títulos que continuam se mexendo depois de pousar; logo no canto; três cartões lado a
lado; molduras e cantoneiras decorativas; crossfade entre mundos; números que não estão nos JSONs.

QA: 1,0 (gancho legível), 9,0 (tensão), 15,0 (título), 22,0 (LGPD), 28,5 (caixa sobre a cicatriz),
35,0 (Alerta), 42,0 (JSON), 52,0 (tempo real + FPS), 61,0 (API), 69,0 (barras), 75,0 (erro), 84,0 (fim).
