# Avaliacao dos detectores de anomalia

Gerado em 2026-11-15T19:29:55.

## Sinais vitais: PhysioNet BIDMC com anomalias injetadas

53 registros reais, 7.07 h a 1 Hz. Cada registro e reproduzido limpo e com cada evento injetado; entre parenteses, a latencia mediana de deteccao.

| Configuracao | Sensib. geral | dessaturacao | taquicardia | taquipneia | piora_combinada | Alertas/h sem injecao | Artefato -> alarme |
|---|---|---|---|---|---|---|---|
| limiar (NEWS2) | 61% | 72% (80 s) | 32% (56 s) | 92% (71 s) | 49% (78 s) | 1.42 | 0/53 |
| + linha de base | 94% | 100% (60 s) | 100% (39 s) | 100% (62 s) | 77% (83 s) | 1.56 | 0/53 |
| + tendencia | 100% | 100% (54 s) | 100% (39 s) | 100% (58 s) | 98% (83 s) | 2.83 | 0/53 |
| + Isolation Forest | 100% | 100% (54 s) | 100% (39 s) | 100% (58 s) | 100% (78 s) | 3.54 | 0/53 |

## Quedas: UR Fall Detection (acelerometro)

| Particao | Quedas | Atividades | Sensibilidade | Especificidade | Erros |
|---|---|---|---|---|---|
| calibracao (impares) | 15 | 20 | 100% | 90% | adl-31, adl-33 |
| teste (pares) | 15 | 20 | 93% | 95% | adl-40, fall-10 |

## Internacao sintetica (eventos rotulados)

| Modalidade | Eventos | Detectados | Anomalias fora de rotulo |
|---|---|---|---|
| sinais_vitais | 3 | 3 | 0 |
| prescricao | 8 | 8 | 0 |
| movimentacao | 3 | 3 | 0 |

| Paciente | Evento | Detectado | Tipos |
|---|---|---|---|
| 559120 | insuficiencia_respiratoria | sim | desvio_basal_fc, desvio_basal_fr, desvio_basal_saturacao, news2_elevado, parametro_critico_fr, parametro_critico_saturacao, tendencia_fr, tendencia_saturacao |
| 443902 | deterioracao_septica | sim | desvio_basal_fc, desvio_basal_fr, desvio_basal_pas, desvio_basal_temperatura, news2_elevado, padrao_multivariado, tendencia_fc, tendencia_pas, tendencia_temperatura |
| 310445 | febre_apos_suspensao | sim | desvio_basal_fc, desvio_basal_temperatura, padrao_multivariado, tendencia_fc, tendencia_temperatura |
| 559120 | alergia_penicilina | sim | conflito_alergia |
| 443902 | dose_10x | sim | acima_referencia, salto_dose |
| 208731 | insulina_10x | sim | salto_dose |
| 208731 | imobilidade | sim | imobilidade_prolongada |
| 117256 | duplo_anticoagulante | sim | duplicidade_terapeutica |
| 310445 | suspensao_antimicrobiano | sim | suspensao_sem_substituto |
| 624190 | alergia_morfina | sim | conflito_alergia |
| 662480 | suspensao_insulina | sim | suspensao_sem_substituto |
| 990117 | alergia_dipirona | sim | conflito_alergia |
| 875431 | agitacao | sim | agitacao_noturna |
| 875431 | saida_noturna | sim | saida_leito_noturna |
