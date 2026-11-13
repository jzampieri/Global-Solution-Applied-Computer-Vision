"""Etapa 6: gera docs/Relatorio_Applied_Computer_Vision.pdf a partir dos resultados reais.

Estrutura exigida pela disciplina: título, integrantes, resumo (~250 palavras), introdução,
objetivo (~50 palavras), explicação técnica das ferramentas, metodologia passo a passo,
exemplos de resultados (prints com caixas) e conclusão.

Todos os números vêm de configs/dataset_resumo.json, avaliacao/metricas.json e
avaliacao/analise_erros.json: rodar de novo após um novo treino atualiza o relatório.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
IMG = RAIZ / "docs" / "img"
SAIDA = RAIZ / "docs" / "Relatorio_Applied_Computer_Vision.pdf"
TEXTOS = RAIZ / "docs" / "relatorio_textos.json"

INTEGRANTES = [
    ("Julio Cesar Zampieri", "RM98772"),
    ("Ricardo Augusto de Matos Filho", "RM95906"),
    ("João Gabriel Dias Nascimento", "RM99092"),
]
REPO = "(preencher: python scripts/06_gerar_relatorio.py --repo URL)"
VIDEO = "(preencher: python scripts/06_gerar_relatorio.py --video URL)"


def fontes() -> tuple[str, str]:
    """Usa Segoe UI/Arial do Windows (acentos e símbolos) com fallback para Helvetica."""
    for reg, bold in [("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"), ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]:
        if Path(reg).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("Corpo", reg))
            pdfmetrics.registerFont(TTFont("Corpo-B", bold))
            return "Corpo", "Corpo-B"
    return "Helvetica", "Helvetica-Bold"


def carregar(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    global REPO, VIDEO
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--video", default=VIDEO)
    a = ap.parse_args()
    REPO, VIDEO = a.repo, a.video
    reg, neg = fontes()
    ds = carregar(RAIZ / "configs" / "dataset_resumo.json")
    met = carregar(RAIZ / "avaliacao" / "metricas.json")
    err = carregar(RAIZ / "avaliacao" / "analise_erros.json")
    tx = carregar(TEXTOS)
    esc = met["escolhido"]
    m = met["modelos"][esc]

    ss = getSampleStyleSheet()
    VERDE = colors.HexColor("#0b5d4b")
    corpo = ParagraphStyle("corpo", parent=ss["BodyText"], fontName=reg, fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName=neg, fontSize=15, textColor=VERDE, spaceBefore=10, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=neg, fontSize=12, spaceBefore=8, spaceAfter=4)
    legenda = ParagraphStyle("leg", parent=corpo, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
    titulo = ParagraphStyle("tit", parent=h1, fontSize=22, leading=28, alignment=TA_CENTER, textColor=VERDE)
    sub = ParagraphStyle("sub", parent=corpo, alignment=TA_CENTER, fontSize=11)

    def P(t: str, s=corpo) -> Paragraph:
        return Paragraph(t, s)

    def tabela(dados, larg, cab=True):
        t = Table(dados, colWidths=larg, repeatRows=1 if cab else 0)
        estilo = [("FONTNAME", (0, 0), (-1, -1), reg), ("FONTSIZE", (0, 0), (-1, -1), 9),
                  ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if cab:
            estilo += [("BACKGROUND", (0, 0), (-1, 0), VERDE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), neg)]
        t.setStyle(TableStyle(estilo))
        return t

    def figura(arq: str, larg_cm: float, txt: str):
        p = IMG / arq
        if not p.exists():
            return P(f"[figura ausente: {arq}]", legenda)
        from PIL import Image as PI
        w, h = PI.open(p).size
        return KeepTogether([Image(str(p), width=larg_cm * cm, height=larg_cm * cm * h / w), P(txt, legenda), Spacer(1, 6)])

    def lista(itens):
        return ListFlowable([ListItem(P(i), leftIndent=12) for i in itens], bulletType="bullet", start="•", leftIndent=12)

    st = []
    # ---------------- capa
    st += [Spacer(1, 3 * cm), P("FIAP · Global Solution 2026.2 · CitizenTech · Engenharia de Software 4º ano", sub), Spacer(1, 0.6 * cm),
           P("Encosta Viva · Visão", titulo), P("Detector de sinais de instabilidade de encosta com YOLO11 treinado sob medida", sub),
           Spacer(1, 0.8 * cm), P("Disciplina: Applied Computer Vision · Atividade: Enxergando o mundo real", sub),
           P("Professores: Arnaldo Viana e Paulo Rodrigues", sub), Spacer(1, 1.2 * cm)]
    st.append(tabela([["Integrante", "RM"]] + [list(x) for x in INTEGRANTES], [9 * cm, 4 * cm]))
    def link(u: str) -> str:
        return f"<link href='{u}' color='blue'>{u}</link>" if u.startswith("http") else u

    st += [Spacer(1, 1 * cm), P(f"Repositório: {link(REPO)}", sub), P(f"Vídeo: {link(VIDEO)}", sub), PageBreak()]

    # ---------------- resumo / introdução / objetivo
    st += [P("Resumo", h1)] + [P(x) for x in tx["resumo"]]
    st += [P("1. Introdução", h1)] + [P(x) for x in tx["introducao"]]
    st += [P("2. Objetivo", h1), P(tx["objetivo"])]

    # ---------------- ferramentas
    st += [P("3. Explicação técnica das ferramentas", h1)]
    for nome, desc in tx["ferramentas"]:
        st += [P(nome, h2), P(desc)]

    # ---------------- metodologia
    st += [P("4. Metodologia passo a passo", h1), figura("pipeline.png", 16, "Figura 1: pipeline do componente Visão, da foto do morador à fila de vistoria da Defesa Civil.")]
    for titulo_passo, paragrafos in tx["metodologia"]:
        st += [P(titulo_passo, h2)] + [P(x) for x in paragrafos]
        if titulo_passo.startswith("Passo 2"):
            linhas = [["Fonte (licença)", "Uso no Encosta Viva", "Treino", "Val.", "Teste"]]
            desc = {"building-cracks": ("lipi17/building-cracks via Roboflow (CC BY 4.0)", "rachadura"),
                    "landslides": ("bbrenes/objectRecognition_Landslides (MIT)", "cicatriz_deslizamento + fundo"),
                    "wall-damage": ("Francesco/wall-damage, Roboflow-100 (CC BY 4.0)", "rachadura + dano_estrutural")}
            for f, s in ds["fontes"].items():
                linhas.append([P(desc[f][0], legenda), desc[f][1], s["train"], s["val"], s["test"]])
            tot = {k: sum(v[k] for v in ds["fontes"].values()) for k in ("train", "val", "test")}
            linhas.append(["Total (imagens originais)", "", tot["train"], tot["val"], tot["test"]])
            st += [tabela(linhas, [6.2 * cm, 4.8 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm]), Spacer(1, 4)]
            cx = [["Classe", "Caixas treino", "Caixas val.", "Caixas teste"]]
            for c in ds["classes"]:
                cx.append([c] + [ds["splits"][s]["caixas_por_classe"].get(c, 0) for s in ("train", "val", "test")])
            st += [tabela(cx, [5.5 * cm, 3 * cm, 3 * cm, 3 * cm]), P(f"Treino ampliado com {ds['splits']['train']['imagens_aumentadas']} cópias offline "
                   f"({', '.join(f'{k}: {v}' for k, v in ds['augmentation_offline'].items())}).", legenda)]
            st.append(figura("mosaico_landslides.jpg", 13, "Figura 2: recortes por ID de categoria no dataset de deslizamentos. A inspeção revelou o deslocamento de +1 entre IDs e nomes do cartão."))
            st.append(figura("augmentation.jpg", 16, "Figura 3: augmentation offline aplicada no treino (chuva, neblina, pouca luz, borrão de movimento) com as caixas preservadas."))

    # ---------------- resultados
    st += [P("5. Resultados", h1), P(tx["resultados_intro"])]
    modelos = met["modelos"]
    linhas = [["Métrica (split de teste)"] + list(modelos)]
    for k, r in [("precisao", "Precisão"), ("recall", "Recall"), ("mAP50", "mAP@50"), ("mAP50_95", "mAP@50-95")]:
        linhas.append([r] + [f"{v['geral'][k]:.3f}" for v in modelos.values()])
    linhas.append(["FPS GPU (RTX 3050, 640 px)"] + [str(v["fps_gpu_rtx3050"]) for v in modelos.values()])
    linhas.append(["FPS CPU (640 px)"] + [str(v["fps_cpu"]) for v in modelos.values()])
    linhas.append(["Parâmetros (milhões)"] + [str(v["parametros_M"]) for v in modelos.values()])
    st += [tabela(linhas, [6.5 * cm] + [3.5 * cm] * len(modelos)), P(f"Tabela: comparação no teste ({met['imagens_teste']} imagens nunca vistas). Modelo escolhido: <b>{esc}</b>.", legenda)]
    pc = [["Classe", "Precisão", "Recall", "mAP@50", "mAP@50-95"]]
    for c, r in m["por_classe"].items():
        pc.append([c, f"{r['precisao']:.3f}", f"{r['recall']:.3f}", f"{r['mAP50']:.3f}", f"{r['mAP50_95']:.3f}"])
    st += [Spacer(1, 6), tabela(pc, [5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.8 * cm]), P(f"Tabela: métricas por classe do {esc} no teste.", legenda)]
    st += [P(x) for x in tx["resultados_analise"]]
    st.append(figura(f"{esc}_curvas_treino.png", 16, "Figura 4: curvas de perda e métricas de validação ao longo das épocas."))
    st.append(figura(f"{esc}_confusion_matrix_normalized.png", 11, "Figura 5: matriz de confusão normalizada no teste."))
    st.append(figura("acertos.jpg", 16, "Figura 6: exemplos de acerto no teste. Verde tracejado = gabarito; caixa colorida = predição com confiança."))
    st.append(figura("erros.jpg", 16, "Figura 7: exemplos de erro no teste (falsos positivos e caixas não detectadas)."))
    st += [P("Análise de erros", h2), P(
        f"Com limiar de confiança 0,35 e critério IoU ≥ 0,5: {err['imagens_sem_erro']} de {met['imagens_teste']} imagens sem nenhum erro; "
        f"{err['imagens_com_falso_positivo']} com algum falso positivo e {err['imagens_com_falso_negativo']} com alguma caixa não detectada. "
        f"Falsos positivos por classe: {', '.join(f'{k} {v}' for k, v in err['falsos_positivos_por_classe'].items())}. "
        f"Falsos negativos por classe: {', '.join(f'{k} {v}' for k, v in err['falsos_negativos_por_classe'].items())}.")]
    st += [P(x) for x in tx["erros_analise"]]
    st.append(figura("demo_api.png", 15, "Figura 8: integração: página do serviço de Visão com a foto anonimizada, as detecções e o nível sugerido."))
    st.append(figura("demo_tempo_real.jpg", 15, "Figura 9: quadro da demonstração em tempo real, com FPS, nível sugerido e aviso de humano no circuito."))

    # ---------------- integração, LGPD, conclusão
    st += [P("6. Integração com o produto e LGPD", h1)] + [P(x) for x in tx["integracao"]]
    st += [P("7. Conclusão", h1)] + [P(x) for x in tx["conclusao"]]
    st += [P("Referências", h1), lista(tx["referencias"])]

    def rodape(c, d):
        c.saveState()
        c.setFont(reg, 8)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(2 * cm, 1.2 * cm, "Encosta Viva · Applied Computer Vision · GS 2026.2")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Encosta Viva · Visão", author=", ".join(n for n, _ in INTEGRANTES))
    doc.build(st, onFirstPage=rodape, onLaterPages=rodape)
    print("PDF gerado:", SAIDA)


if __name__ == "__main__":
    main()
