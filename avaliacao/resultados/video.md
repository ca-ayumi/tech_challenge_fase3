# Avaliacao da deteccao de queda por video (UR Fall, RGB)

Gerado em 2026-11-15T21:27:50. Modelo: YOLOv8s-pose; limiares calibrados nas sequencias impares.

| Particao | Quedas | Atividades | Sensibilidade | Especificidade | Atividades que deitam sem alarme | Latencia mediana | Erros |
|---|---|---|---|---|---|---|---|
| calibracao (impares) | 15 | 20 | 87% | 90% | 6/8 | 0.67 s | adl-33, adl-37, fall-03, fall-05 |
| teste (pares) | 15 | 20 | 100% | 90% | 6/8 | 0.37 s | adl-22, adl-32 |
