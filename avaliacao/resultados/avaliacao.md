# Avaliacao do assistente medico

Gerado em 2026-09-12T20:18:02.

## 1. Qualidade da geracao (conjunto de teste)

| Metrica | Ajustado (LoRA) | Base (sem ajuste) |
|---|---|---|
| Exemplos | 29 | 29 |
| Adesao ao formato | 86.2% | 72.4% |
| Respostas com fonte | 86.2% | 72.4% |
| Citacao coincide com a de ouro | 48.3% | 13.8% |
| Citacoes inexistentes (total) | 5 | 13 |
| ROUGE-L medio | 0.611 | 0.1838 |
| Violacao de dose na saida | 0.0% | 0.0% |
| Tokens gerados (media) | 232.1 | 312.8 |
| Latencia (media) | 3097 ms | 3245 ms |

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
| consultas avaliadas | 26 |
| acerto em 4 pct | 80.8 |
| posicao media do acerto | 1.29 |
