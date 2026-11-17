"""Gera docs/diagramas/fluxo_multimodal.svg: o fluxo da Fase 4 em 16:9, para apresentacao.

Uso: python docs/diagramas/gerar_fluxo_apresentacao.py
Para PNG: Chrome --headless --screenshot --window-size=1920,1080 --force-device-scale-factor=2.
"""

from __future__ import annotations

from pathlib import Path

L, A = 1920, 1080
CORES = {
    "fonte": ("#e0f2fe", "#0284c7", "#0c4a6e"),
    "local": ("#dcfce7", "#16a34a", "#14532d"),
    "nuvem": ("#ede9fe", "#7c3aed", "#3b0764"),
    "chave": ("#fef3c7", "#d97706", "#78350f"),
    "resp": ("#ffedd5", "#ea580c", "#7c2d12"),
    "banco": ("#f1f5f9", "#475569", "#0f172a"),
}
LINHA = "#64748b"
partes: list[str] = []


def caixa(x, y, w, h, estilo, titulo, detalhe="", raio=12, tamanho=21):
    fundo, borda, texto = CORES[estilo]
    partes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{raio}" fill="{fundo}" '
                  f'stroke="{borda}" stroke-width="2"/>')
    linhas = detalhe.split("\n") if detalhe else []
    total = 1 + len(linhas)
    passo = 24
    y0 = y + h / 2 - (total - 1) * passo / 2 + 7
    partes.append(f'<text x="{x + w / 2}" y="{y0}" text-anchor="middle" font-size="{tamanho}" '
                  f'font-weight="700" fill="{texto}">{titulo}</text>')
    for i, linha in enumerate(linhas, start=1):
        partes.append(f'<text x="{x + w / 2}" y="{y0 + i * passo}" text-anchor="middle" '
                      f'font-size="17" fill="{texto}">{linha}</text>')


def seta(x1, y1, x2, y2, tracejada=False, curva=True):
    estilo = ' stroke-dasharray="7 6"' if tracejada else ""
    if curva and y1 != y2 and abs(x2 - x1) > abs(y2 - y1) * 0.2:
        meio = (x1 + x2) / 2
        caminho = f"M{x1},{y1} C{meio},{y1} {meio},{y2} {x2},{y2}"
    else:
        caminho = f"M{x1},{y1} L{x2},{y2}"
    partes.append(f'<path d="{caminho}" fill="none" stroke="{LINHA}" stroke-width="2.4"{estilo} '
                  'marker-end="url(#ponta)"/>')


def cabecalho(x, w, texto):
    partes.append(f'<text x="{x + w / 2}" y="150" text-anchor="middle" font-size="17" '
                  f'font-weight="700" fill="#475569" letter-spacing="1.5">{texto}</text>')


# Titulo
partes.append('<text x="60" y="72" font-size="38" font-weight="800" fill="#0f172a">'
              'Monitoramento multimodal: do dado ao alerta</text>')
partes.append('<text x="60" y="108" font-size="20" fill="#475569">Cada tipo de dado tem seu detector; '
              'todos falam a mesma língua (Anomalia); a fusão e a central decidem o que chega à equipe.</text>')

# Colunas
X1, W1 = 60, 260          # fontes
X2, W2 = 400, 600         # deteccao
X3, W3 = 1080, 170        # anomalia
X4, W4 = 1330, 260        # decisao
X5, W5 = 1650, 220        # resposta
cabecalho(X1, W1, "FONTES")
cabecalho(X2, W2, "DETECÇÃO POR TIPO DE DADO")
cabecalho(X3, W3, "CONTRATO")
cabecalho(X4, W4, "DECISÃO")
cabecalho(X5, W5, "RESPOSTA")

TOPO, H, GAP = 180, 108, 18
linhas_y = [TOPO + i * (H + GAP) for i in range(6)]
meio = [y + H / 2 for y in linhas_y]

# Fontes (o audio ocupa duas linhas)
caixa(X1, linhas_y[0], W1, H, "fonte", "Monitor", "sinais vitais")
caixa(X1, linhas_y[1], W1, H, "fonte", "Prontuário", "histórico de prescrições")
caixa(X1, linhas_y[2], W1, H, "fonte", "Sensor de leito", "e acelerômetro")
caixa(X1, linhas_y[3], W1, 2 * H + GAP, "fonte", "Áudio da consulta", "enviado pela equipe")
caixa(X1, linhas_y[5], W1, H, "fonte", "Vídeo", "quarto e fisioterapia\nenviado pela equipe")

# Deteccao
caixa(X2, linhas_y[0], W2, H, "local", "Sinais vitais",
      "NEWS2 · linha de base do paciente\ntendência · Isolation Forest")
caixa(X2, linhas_y[1], W2, H, "local", "Prescrições",
      "alergia por grupo · dose 10× · duplicidade\nsuspensão sem substituto")
caixa(X2, linhas_y[2], W2, H, "local", "Movimentação",
      "imobilidade · agitação noturna\nsaída do leito à noite · queda")
caixa(X2, linhas_y[3], W2, H, "local", "Voz (Praat)",
      "ritmo, pausas, tom · jitter, shimmer, HNR\nfadiga · dispneia · disartria")
# linha do texto: Azure -> anonimizador -> Azure
w3 = 176
xs = [X2, X2 + w3 + 36, X2 + 2 * (w3 + 36)]
caixa(xs[0], linhas_y[4], w3, H, "nuvem", "Speech to Text", "transcrição com\ntempo por palavra", tamanho=18)
caixa(xs[1], linhas_y[4], w3, H, "local", "Anonimizador", "pseudônimos\nantes da nuvem", tamanho=18)
caixa(xs[2], linhas_y[4], w3, H, "nuvem", "Language", "sentimento e\nsintomas (Health)", tamanho=18)
seta(xs[0] + w3, meio[4], xs[1], meio[4])
seta(xs[1] + w3, meio[4], xs[2], meio[4])
caixa(X2, linhas_y[5], W2, H, "local", "YOLOv8-pose + YOLOv8",
      "17 pontos do esqueleto (como o OpenPose) · objetos\nqueda · áreas críticas · exercício e desvios")

for i in (0, 1, 2, 5):
    seta(X1 + W1, meio[i], X2, meio[i])
y_audio = linhas_y[3] + H + GAP / 2
seta(X1 + W1, y_audio - 30, X2, meio[3])
seta(X1 + W1, y_audio + 30, X2, meio[4])

# Anomalia
ya = (meio[0] + meio[5]) / 2 - 60
caixa(X3, ya, W3, 120, "chave", "Anomalia", "contrato único\n(mesmo formato)", raio=60)
for i in range(6):
    seta(X2 + W2, meio[i], X3, ya + 60 + (i - 2.5) * 14)

# Decisao
yf, yc = 260, ya
caixa(X4, yf, W4, 130, "resp", "Fusão tardia", "risco 0–100 por paciente\n6 regras clínicas\n(ex.: suspeita de AVC)")
caixa(X4, yc, W4, 120, "resp", "Central de alertas", "gravidade mínima\nsem repetir em 30 min\nreavisa se piorar")
seta(X3 + W3, ya + 40, X4, yf + 65)
seta(X3 + W3, ya + 75, X4, yc + 75)
seta(X4 + W4 / 2, yf + 130, X4 + W4 / 2, yc, curva=False)

# Banco
yb = yc + 200
partes.append(f'<ellipse cx="{X4 + W4 / 2}" cy="{yb}" rx="110" ry="18" fill="#f1f5f9" stroke="#475569" stroke-width="2"/>')
partes.append(f'<path d="M{X4 + W4 / 2 - 110},{yb} V{yb + 90} A110,18 0 0 0 {X4 + W4 / 2 + 110},{yb + 90} V{yb}" '
              'fill="#f1f5f9" stroke="#475569" stroke-width="2"/>')
partes.append(f'<text x="{X4 + W4 / 2}" y="{yb + 45}" text-anchor="middle" font-size="19" font-weight="700" fill="#0f172a">SQLite</text>')
partes.append(f'<text x="{X4 + W4 / 2}" y="{yb + 70}" text-anchor="middle" font-size="16" fill="#0f172a">anomalias · alertas · auditoria</text>')
seta(X4 + W4 / 2, yc + 120, X4 + W4 / 2, yb - 20, tracejada=True, curva=False)

# Resposta
caixa(X5, yc, W5, 120, "resp", "Notificação", "painel · arquivo\nTeams / Slack")
seta(X4 + W4, yc + 60, X5, yc + 60)
ye = yc + 200
caixa(X5, ye, W5, 110, "resp", "Equipe reconhece", "quem, quando, perfil\nregistrado na auditoria", raio=55)
seta(X5 + W5 / 2, yc + 120, X5 + W5 / 2, ye, curva=False)

# Legenda
yl = 1000
itens = [("fonte", "Fonte de dados"), ("local", "Processado no hospital"),
         ("nuvem", "Azure AI Services"), ("resp", "Decisão e resposta à equipe")]
x = 60
for estilo, rotulo in itens:
    fundo, borda, _ = CORES[estilo]
    partes.append(f'<rect x="{x}" y="{yl}" width="26" height="26" rx="6" fill="{fundo}" stroke="{borda}" stroke-width="2"/>')
    partes.append(f'<text x="{x + 38}" y="{yl + 20}" font-size="18" fill="#334155">{rotulo}</text>')
    x += 38 + len(rotulo) * 10 + 50

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{L}" height="{A}" viewBox="0 0 {L} {A}" '
       'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">'
       '<defs><marker id="ponta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
       f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINHA}"/></marker></defs>'
       f'<rect width="{L}" height="{A}" fill="#ffffff"/>' + "".join(partes) + "</svg>")
destino = Path(__file__).with_name("fluxo_multimodal.svg")
destino.write_text(svg, encoding="utf-8")
print(destino)
