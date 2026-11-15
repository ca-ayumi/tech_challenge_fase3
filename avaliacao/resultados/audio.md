# Avaliacao da analise de audio

Gerado em 2026-11-15T20:28:29.

## Qualidade vocal em vozes reais (PhysioNet VOICED)

208 gravacoes (57 saudaveis, 151 com patologia vocal).

| Medida | AUC | Mediana saudavel | Mediana patologica |
|---|---|---|---|
| jitter_local | 0.65 | 0.0033 | 0.0045 |
| shimmer_local | 0.63 | 0.0378 | 0.0463 |
| hnr_db | 0.65 | 19.4438 | 18.1543 |

Regressao logistica com as tres medidas, validacao cruzada de 5 dobras: AUC 0.60.

## Condicoes vocais simuladas (Azure TTS)

| Particao | Acertos | Erros |
|---|---|---|
| calibracao | 12/12 | - |
| teste | 12/12 | - |

## Consultas dos pacientes (pipeline completo)

| Prontuario | Voz | Termos presentes | Negados | Conf. STT | OK |
|---|---|---|---|---|---|
| 117256 | - | - | - | 0.9368 | sim |
| 208731 | fadiga | fadiga | - | 0.8863 | sim |
| 559120 | dispneia | dispneia, febre | - | 0.7825 | sim |
| 553119 | disartria | deficit_neurologico | - | 0.8592 | sim |
| 771203 | - | dor_toracica | - | 0.9519 | sim |
| 875431 | - | - | dispneia, dor_toracica | 0.8772 | sim |
