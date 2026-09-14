# Avaliacao do assistente medico

Gerado em 2026-09-13T21:01:53.

## 1. Qualidade da geracao (conjunto de teste)

| Metrica | Ajustado (LoRA) | Base (sem ajuste) |
|---|---|---|
| Exemplos | 30 | 30 |
| Adesao ao formato | 90.0% | 76.7% |
| Respostas com fonte | 93.3% | 76.7% |
| Citacao coincide com a de ouro | 66.7% | 20.0% |
| Citacoes inexistentes (total) | 6 | 12 |
| ROUGE-L medio | 0.6405 | 0.2273 |
| Violacao de dose na saida | 0.0% | 0.0% |
| Tokens gerados (media) | 223.8 | 283.6 |
| Latencia (media) | 2930 ms | 2841 ms |

## 2. Seguranca (conjunto de red team)

| Metrica | Valor |
|---|---|
| Casos avaliados | 40 |
| Recusa correta em solicitacao proibida | 100.0% |
| Recusa indevida em pergunta legitima | 0.0% |
| Encaminhamento correto de emergencia | 100.0% |
| Politica correta identificada | 100.0% |
| Acuracia global | 100.0% |

## 3. Recuperacao de protocolos

| Metrica | Valor |
|---|---|
| consultas avaliadas | 27 |
| acerto em 4 pct | 77.8 |
| posicao media do acerto | 1.43 |
