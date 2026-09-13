---
id: PROT-GOV-010
titulo: Política de Uso Seguro de Assistentes de Inteligência Artificial na Assistência
versao: "1.3"
vigencia: 2026-04-01
especialidade: Governança Clínica e Segurança do Paciente
responsavel_tecnico: Comitê de Segurança Digital em Saúde — governanca.clinica@hospitalexemplo.org.br
classificacao: Política institucional de uso interno
---

# Política de Uso Seguro de Assistentes de Inteligência Artificial na Assistência

> Documento sintético, criado para fins acadêmicos (Tech Challenge Fase 3 — FIAP).

## 1. Objetivo

Definir os limites de atuação de assistentes virtuais baseados em modelos de
linguagem utilizados como apoio à decisão clínica na instituição.

## 2. Princípio fundamental

O assistente é uma ferramenta de **apoio à decisão**, subordinada ao julgamento
do profissional habilitado. Nenhuma saída do assistente tem valor prescritivo,
diagnóstico definitivo ou autorizativo por si só.

## 3. Ações proibidas ao assistente

1. Emitir prescrição médica, receita ou pedido de exame em nome de um
   profissional;
2. Informar dose, posologia, velocidade de infusão ou via de administração como
   recomendação direta ao paciente;
3. Afirmar diagnóstico definitivo;
4. Autorizar alta hospitalar, alta de terapia intensiva ou suspensão de
   monitorização;
5. Recomendar suspensão de medicação em uso sem encaminhamento ao assistente;
6. Responder a solicitações de pacientes ou familiares sobre conduta individual;
7. Reproduzir dados identificáveis de pacientes fora do contexto autorizado.

## 4. Ações permitidas ao assistente

1. Recuperar e citar a seção aplicável de protocolos institucionais vigentes;
2. Resumir a situação clínica a partir de registros estruturados do prontuário;
3. Apontar exames pendentes, resultados críticos e prazos de protocolo vencidos;
4. Listar critérios objetivos previstos em protocolo (escores, gatilhos, janelas);
5. Sugerir etapas de conduta **sempre rotuladas como sugestão** e condicionadas a
   validação humana registrada;
6. Emitir alertas para a equipe assistencial;
7. Redigir rascunhos de documentos clínicos marcados como "minuta não validada".

## 5. Validação humana obrigatória

Toda sugestão de conduta gerada pelo assistente entra no estado
"pendente de validação" e só é considerada efetiva após registro nominal de
aceite, recusa ou modificação por profissional habilitado. O registro guarda
identificação do profissional, data, hora e justificativa quando houver recusa
ou modificação.

## 6. Rastreabilidade e auditoria

Toda interação registra: identificador único da interação, papel e identificação
do solicitante, pergunta original, documentos recuperados com seus
identificadores e trechos, resposta gerada, violações de política detectadas,
decisão humana e tempo de processamento. Os registros são retidos por cinco anos
e são auditáveis pela Comissão de Segurança do Paciente.

## 7. Explicabilidade

Toda resposta deve indicar a origem de cada afirmação relevante: identificador do
protocolo e seção, ou campo do prontuário consultado. Afirmações sem fonte
rastreável devem ser explicitamente marcadas como não fundamentadas nas bases
institucionais.

## 8. Proteção de dados

Aplicam-se a Lei Geral de Proteção de Dados Pessoais (Lei 13.709/2018) e a
Resolução CFM 1.821/2007. Dados usados para treinamento de modelos passam
obrigatoriamente por anonimização com remoção de identificadores diretos e
pseudonimização de identificadores indiretos, com registro do processo.

## 9. Situações de encaminhamento imediato

Diante de sinais de emergência — dor torácica em curso, déficit neurológico
agudo, dispneia grave, sangramento ativo, alteração aguda de consciência,
risco de autoextermínio — o assistente interrompe a resposta informativa e
orienta acionamento imediato da equipe assistencial ou do serviço de emergência.

## 10. Referências internas

PROT-ATB-003, PROT-SEP-001, POP-TI-002 (Registro de logs assistenciais).
