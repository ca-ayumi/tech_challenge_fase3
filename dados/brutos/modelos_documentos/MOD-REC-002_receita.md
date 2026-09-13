---
id: MOD-REC-002
tipo: receita
titulo: Modelo Institucional de Receituário
versao: "1.6"
---

# Modelo Institucional de Receituário

## Estrutura obrigatória

1. Identificação do estabelecimento (nome, endereço, CNES);
2. Identificação do paciente (nome completo, registro, data de nascimento);
3. Prescrição: fármaco, concentração, forma farmacêutica, via, posologia,
   duração e quantidade total;
4. Orientações ao paciente em linguagem acessível;
5. Data, assinatura e carimbo do prescritor com CRM;
6. Para medicamentos controlados, uso do receituário específico previsto na
   legislação vigente.

## Exemplo de estrutura (sem conteúdo terapêutico)

```
HOSPITAL EXEMPLO — CNES 0000000
Paciente: [NOME COMPLETO]   Registro: [PRONTUÁRIO]   Nascimento: [DD/MM/AAAA]

1) [FÁRMACO] [CONCENTRAÇÃO] — [FORMA FARMACÊUTICA]
   Uso [VIA]. [POSOLOGIA] por [DURAÇÃO].   Quantidade: [TOTAL]

Orientações: [TEXTO EM LINGUAGEM ACESSÍVEL]

[CIDADE], [DATA]
______________________________
[NOME DO MÉDICO] — CRM [UF] [NÚMERO]
```

## Regra de uso por sistemas de apoio — restrição máxima

Assistentes virtuais **não emitem receitas**. Não preenchem fármaco,
concentração, posologia, via ou quantidade. A única atuação permitida é:

- Explicar a estrutura formal exigida no receituário;
- Verificar se a prescrição já registrada pelo médico conflita com alergias
  documentadas do paciente e emitir alerta;
- Apontar necessidade de ajuste por função renal como **alerta para o médico**,
  sem indicar o valor do ajuste.

Ver PROT-GOV-010 §3 e §4.
