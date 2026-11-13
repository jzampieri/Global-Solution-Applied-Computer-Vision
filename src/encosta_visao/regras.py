"""Regra explicável que liga a Visão ao motor de risco do Encosta Viva.

Princípio do produto: a IA NÃO decide evacuação. A Visão apenas sugere um nível
e explica o porquê. Toda sugestão de "Alerta" exige confirmação de um agente da
Defesa Civil antes de mudar o nível oficial do setor.

Níveis (mesma escala do motor de risco compartilhado):
  Observação -> Atenção -> Alerta -> Alerta máximo
A Visão sozinha nunca sugere "Alerta máximo": esse nível depende de sensor de
inclinação + solo saturado + chuva forte (robô sentinela e pipeline de dados).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Classes que, com confiança alta, indicam sinal forte de movimentação do terreno
# ou de estrutura comprometida (desplacamento, ferragem exposta).
CLASSES_FORTES = {"rachadura", "cicatriz_deslizamento", "dano_estrutural"}

MOTIVOS = {
    "rachadura": "Rachadura detectada: sinal clássico de movimentação do solo sob a construção",
    "cicatriz_deslizamento": "Cicatriz de deslizamento detectada: a encosta já cedeu neste ponto",
    "dano_estrutural": "Dano estrutural grave detectado: a construção pode estar comprometida",
}

LIMIAR_ALTO = 0.60  # confiança a partir da qual a detecção forte sugere "Alerta"
LIMIAR_MINIMO = 0.35  # abaixo disso a detecção é descartada da regra (ruído)


@dataclass
class Sugestao:
    nivel_sugerido: str
    motivo: str
    abrir_vistoria: bool
    requer_confirmacao_humana: bool = True
    fatores: list[str] = field(default_factory=list)

    def como_dict(self) -> dict:
        return {
            "nivel_sugerido": self.nivel_sugerido,
            "motivo": self.motivo,
            "abrir_vistoria": self.abrir_vistoria,
            "requer_confirmacao_humana": self.requer_confirmacao_humana,
            "fatores": self.fatores,
        }


def sugerir_nivel(deteccoes: list[dict]) -> Sugestao:
    """Recebe a lista de detecções ({classe, confianca, ...}) e devolve a sugestão."""
    validas = [d for d in deteccoes if d["confianca"] >= LIMIAR_MINIMO]
    if not validas:
        return Sugestao(
            nivel_sugerido="Observação",
            motivo="Nenhum sinal visual de instabilidade com confiança suficiente.",
            abrir_vistoria=False,
            requer_confirmacao_humana=False,
        )

    fatores = [f"{d['classe']} ({d['confianca']:.0%})" for d in sorted(validas, key=lambda d: -d["confianca"])]
    fortes = [d for d in validas if d["classe"] in CLASSES_FORTES and d["confianca"] >= LIMIAR_ALTO]
    if fortes:
        melhor = max(fortes, key=lambda d: d["confianca"])
        return Sugestao(
            nivel_sugerido="Alerta",
            motivo=(
                f"{MOTIVOS[melhor['classe']]} ({melhor['confianca']:.0%} de confiança). "
                "Sugerido chamado de vistoria. "
                "O nível do setor só muda após confirmação da Defesa Civil."
            ),
            abrir_vistoria=True,
            fatores=fatores,
        )

    return Sugestao(
        nivel_sugerido="Atenção",
        motivo=(
            "Sinais visuais possíveis, mas com confiança moderada. "
            "Recomendada vistoria de rotina e novas fotos, de perto e com boa luz."
        ),
        abrir_vistoria=True,
        fatores=fatores,
    )
