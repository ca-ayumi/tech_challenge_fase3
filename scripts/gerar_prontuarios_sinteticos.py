"""Gera prontuarios sinteticos (com PII ficticia) usados como base do hospital.

Os dados sao totalmente ficticios e deterministicos: nenhum paciente real e
representado. A PII e incluida de proposito, para que o pipeline de
anonimizacao tenha o que remover e para que a demonstracao seja honesta.

Uso:
    python scripts/gerar_prontuarios_sinteticos.py [--saida dados/brutos/prontuarios.jsonl]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA_PADRAO = RAIZ / "dados" / "brutos" / "prontuarios.jsonl"

AGORA = datetime.now().replace(second=0, microsecond=0)


def h(minutos: int) -> str:
    """Instante deslocado em minutos para tras a partir de agora."""
    return (AGORA - timedelta(minutes=minutos)).isoformat(timespec="minutes")


def exame(codigo, nome, status, minutos_solicitado, resultado=None, unidade=None,
          referencia=None, critico=False):
    return {
        "codigo": codigo,
        "nome": nome,
        "status": status,
        "solicitado_em": h(minutos_solicitado),
        "liberado_em": h(max(minutos_solicitado - 30, 0)) if status == "liberado" else None,
        "resultado": resultado,
        "unidade": unidade,
        "referencia": referencia,
        "critico": critico,
    }


def sinais(minutos, pas, pad, fc, fr, temp, sato2, glasgow=15):
    return {
        "aferido_em": h(minutos),
        "pas": pas, "pad": pad, "fc": fc, "fr": fr,
        "temperatura": temp, "saturacao": sato2, "glasgow": glasgow,
    }


PACIENTES = [
    {
        "prontuario": "884521",
        "nome": "Maria Aparecida da Silva",
        "cpf": "321.654.987-00",
        "cns": "700 5012 3344 8877",
        "rg": "28.445.120-7",
        "data_nascimento": "1958-03-12",
        "sexo": "F",
        "telefone": "(11) 97744-2210",
        "email": "maria.aparecida58@provedorexemplo.com.br",
        "endereco": "Rua das Acacias, 421, apto 52 - Sao Paulo/SP - CEP 04532-010",
        "unidade": "Pronto-Socorro Adulto",
        "leito": "PS-07",
        "data_admissao": h(95),
        "motivo_internacao": "Febre, disuria e queda do estado geral ha 2 dias",
        "medico_responsavel": "Dr. Ricardo Almeida Sobral (CRM-SP 118.442)",
        "comorbidades": ["Diabetes mellitus tipo 2", "Hipertensao arterial"],
        "alergias": [],
        "medicacoes_em_uso": ["Metformina", "Losartana"],
        "protocolos_ativos": [{"protocolo": "PROT-SEP-001", "aberto_em": h(70)}],
        "sinais_vitais": [
            sinais(90, 102, 60, 118, 26, 38.7, 93),
            sinais(30, 94, 55, 124, 28, 38.9, 92),
        ],
        "exames": [
            exame("HMG", "Hemograma completo", "liberado", 65, "Leucocitos 19.400/mm3", "/mm3", "4.000 a 11.000"),
            exame("LAC", "Lactato arterial", "pendente", 65, critico=True),
            exame("HMC1", "Hemocultura - amostra 1", "pendente", 65, critico=True),
            exame("HMC2", "Hemocultura - amostra 2", "pendente", 65, critico=True),
            exame("URO", "Urocultura", "pendente", 60),
            exame("CREA", "Creatinina", "liberado", 65, "1,8", "mg/dL", "0,6 a 1,2"),
        ],
        "eventos": [{"tipo": "antimicrobiano_administrado", "ocorreu": False, "registrado_em": None}],
        "evolucoes": [
            {"data_hora": h(70), "autor": "Dr. Ricardo Almeida Sobral (CRM-SP 118.442)",
             "texto": "Paciente Maria Aparecida da Silva, 67 anos, CPF 321.654.987-00, admitida com quadro de febre e disuria. Filha Joana da Silva, telefone (11) 98123-4455, acompanha. Aberto protocolo de sepse com foco urinario presumido."},
        ],
    },
    {
        "prontuario": "771203",
        "nome": "Joao Batista Ferreira",
        "cpf": "145.987.320-11",
        "cns": "701 2233 4455 6677",
        "rg": "19.220.884-3",
        "data_nascimento": "1961-07-30",
        "sexo": "M",
        "telefone": "(11) 99560-7781",
        "email": "jbferreira61@provedorexemplo.com.br",
        "endereco": "Avenida Guarulhos, 1820, casa 3 - Guarulhos/SP - CEP 07020-330",
        "unidade": "Pronto-Socorro Adulto",
        "leito": "PS-02",
        "data_admissao": h(22),
        "motivo_internacao": "Dor toracica opressiva ha 1 hora com sudorese",
        "medico_responsavel": "Dra. Helena Martins Prado (CRM-SP 92.117)",
        "comorbidades": ["Tabagismo", "Dislipidemia"],
        "alergias": [],
        "medicacoes_em_uso": ["Sinvastatina"],
        "protocolos_ativos": [{"protocolo": "PROT-CAR-002", "aberto_em": h(20)}],
        "sinais_vitais": [sinais(18, 148, 92, 96, 22, 36.4, 95)],
        "exames": [
            exame("ECG", "Eletrocardiograma 12 derivacoes", "liberado", 18,
                  "Supradesnivelamento de ST em parede anterior", None, None, critico=True),
            exame("TROP0", "Troponina ultrassensivel - admissao", "pendente", 16, critico=True),
            exame("RXT", "Radiografia de torax", "pendente", 15),
        ],
        "eventos": [{"tipo": "hemodinamica_ativada", "ocorreu": False, "registrado_em": None}],
        "evolucoes": [
            {"data_hora": h(20), "autor": "Dra. Helena Martins Prado (CRM-SP 92.117)",
             "texto": "Joao Batista Ferreira, 64 anos, chega com dor toracica tipica. ECG com supra de ST anterior. Esposa Cleide Ferreira contactada no (11) 99560-7781."},
        ],
    },
    {
        "prontuario": "553119",
        "nome": "Terezinha Lopes de Oliveira",
        "cpf": "402.118.556-42",
        "cns": "702 8877 6655 4433",
        "rg": "33.190.774-1",
        "data_nascimento": "1949-11-02",
        "sexo": "F",
        "telefone": "(11) 96633-1102",
        "email": "terezinha.lopes@provedorexemplo.com.br",
        "endereco": "Rua Sao Jorge, 77 - Osasco/SP - CEP 06010-120",
        "unidade": "Pronto-Socorro Adulto",
        "leito": "PS-11",
        "data_admissao": h(45),
        "motivo_internacao": "Hemiparesia a direita e disartria de inicio subito",
        "medico_responsavel": "Dr. Fabio Kenji Nakamura (CRM-SP 134.209)",
        "comorbidades": ["Fibrilacao atrial", "Hipertensao arterial"],
        "alergias": ["Contraste iodado"],
        "medicacoes_em_uso": ["Varfarina", "Atenolol"],
        "protocolos_ativos": [{"protocolo": "PROT-NEU-003", "aberto_em": h(43)}],
        "sinais_vitais": [sinais(40, 176, 98, 88, 18, 36.6, 96, glasgow=14)],
        "exames": [
            exame("GLIC", "Glicemia capilar", "liberado", 42, "132", "mg/dL", "70 a 99"),
            exame("NIHSS", "Escala NIHSS", "liberado", 40, "9 pontos", "pontos", None, critico=True),
            exame("TCC", "Tomografia de cranio sem contraste", "pendente", 40, critico=True),
            exame("INR", "Coagulograma com INR", "pendente", 40, critico=True),
        ],
        "eventos": [{"tipo": "ultimo_momento_visto_bem", "ocorreu": True, "registrado_em": h(130)}],
        "evolucoes": [
            {"data_hora": h(43), "autor": "Dr. Fabio Kenji Nakamura (CRM-SP 134.209)",
             "texto": "Terezinha Lopes de Oliveira, 76 anos, prontuario 553119, trazida pelo SAMU. Ultimo momento vista bem ha aproximadamente 2 horas, segundo o vizinho Sr. Osvaldo, telefone (11) 94422-0087. Em uso de varfarina."},
        ],
    },
    {
        "prontuario": "662480",
        "nome": "Lucas Martins Andrade",
        "cpf": "088.774.221-09",
        "cns": "703 1122 3344 5566",
        "rg": "41.775.302-9",
        "data_nascimento": "1997-05-18",
        "sexo": "M",
        "telefone": "(11) 98844-6620",
        "email": "lucas.andrade97@provedorexemplo.com.br",
        "endereco": "Rua Tupi, 300, bloco B - Sao Paulo/SP - CEP 01233-000",
        "unidade": "Unidade de Terapia Intensiva",
        "leito": "UTI-04",
        "data_admissao": h(180),
        "motivo_internacao": "Hiperglicemia, vomitos e dispneia - cetoacidose diabetica",
        "medico_responsavel": "Dr. Marcelo Tavares Lima (CRM-SP 87.560)",
        "comorbidades": ["Diabetes mellitus tipo 1"],
        "alergias": [],
        "medicacoes_em_uso": ["Insulina regular em infusao continua", "Cristaloide isotonico"],
        "protocolos_ativos": [{"protocolo": "PROT-END-005", "aberto_em": h(175)}],
        "sinais_vitais": [sinais(60, 108, 64, 112, 28, 36.8, 97)],
        "exames": [
            exame("GLIC", "Glicemia", "liberado", 170, "418", "mg/dL", "70 a 99", critico=True),
            exame("GASO", "Gasometria arterial", "liberado", 170, "pH 7,08 / HCO3 9", None, "pH 7,35 a 7,45", critico=True),
            exame("K", "Potassio serico", "liberado", 170, "3,1", "mEq/L", "3,5 a 5,0", critico=True),
            exame("CET", "Cetonemia", "liberado", 168, "Positiva", None, None),
            exame("K2", "Potassio serico - controle", "pendente", 60, critico=True),
        ],
        "eventos": [{"tipo": "insulina_iniciada", "ocorreu": True, "registrado_em": h(165)}],
        "evolucoes": [
            {"data_hora": h(165), "autor": "Dr. Marcelo Tavares Lima (CRM-SP 87.560)",
             "texto": "Lucas Martins Andrade, 28 anos, CPF 088.774.221-09, em cetoacidose diabetica grave. Iniciada insulina em infusao continua. Mae, Sra. Marlene Andrade, informada pelo telefone (11) 98844-6620."},
        ],
    },
    {
        "prontuario": "559120",
        "nome": "Antonio Carlos Moreira",
        "cpf": "255.410.889-33",
        "cns": "700 5012 3344 8877",
        "rg": "12.330.998-4",
        "data_nascimento": "1952-01-25",
        "sexo": "M",
        "telefone": "(11) 95511-8899",
        "email": "acmoreira52@provedorexemplo.com.br",
        "endereco": "Rua Bela Vista, 90 - Santo Andre/SP - CEP 09010-540",
        "unidade": "Enfermaria Clinica",
        "leito": "ENF-210",
        "data_admissao": h(60 * 26),
        "motivo_internacao": "Pneumonia adquirida na comunidade",
        "medico_responsavel": "Dra. Patricia Gomes Vieira (CRM-SP 96.733)",
        "comorbidades": ["DPOC", "Insuficiencia cardiaca"],
        "alergias": ["Penicilina"],
        "medicacoes_em_uso": ["Ceftriaxona", "Azitromicina", "Furosemida"],
        "protocolos_ativos": [{"protocolo": "PROT-PNM-006", "aberto_em": h(60 * 26)}],
        "sinais_vitais": [
            sinais(60 * 25, 92, 58, 104, 32, 38.4, 89),
            sinais(120, 108, 66, 96, 24, 37.6, 93),
        ],
        "exames": [
            exame("RXT", "Radiografia de torax", "liberado", 60 * 25, "Consolidacao em base direita", None, None),
            exame("UR", "Ureia", "liberado", 60 * 25, "62", "mg/dL", "15 a 45", critico=True),
            exame("PCR", "Proteina C reativa", "liberado", 60 * 4, "118", "mg/L", "ate 5"),
            exame("HMC1", "Hemocultura - amostra 1", "liberado", 60 * 25, "Negativa ate o momento", None, None),
        ],
        "eventos": [{"tipo": "reavaliacao_antimicrobiano", "ocorreu": False, "registrado_em": None}],
        "evolucoes": [
            {"data_hora": h(60 * 26), "autor": "Dra. Patricia Gomes Vieira (CRM-SP 96.733)",
             "texto": "Antonio Carlos Moreira, 74 anos, CNS 700 5012 3344 8877, internado com pneumonia. CURB-65 de 4 pontos na admissao: confusao, ureia 62, FR 32 e idade acima de 65 anos."},
        ],
    },
    {
        "prontuario": "990117",
        "nome": "Beatriz Nunes Carvalho",
        "cpf": "617.203.994-55",
        "cns": "704 5566 7788 9900",
        "rg": "52.001.337-6",
        "data_nascimento": "1990-09-08",
        "sexo": "F",
        "telefone": "(11) 97001-3322",
        "email": "bia.carvalho@provedorexemplo.com.br",
        "endereco": "Alameda dos Ipes, 15 - Sao Paulo/SP - CEP 05422-080",
        "unidade": "Pronto-Socorro Adulto",
        "leito": "PS-05",
        "data_admissao": h(200),
        "motivo_internacao": "Urticaria generalizada e dispneia apos uso de analgesico",
        "medico_responsavel": "Dr. Andre Salgado Ferreira (CRM-SP 122.318)",
        "comorbidades": ["Asma"],
        "alergias": ["Dipirona", "Anti-inflamatorio nao esteroidal"],
        "medicacoes_em_uso": ["Dipirona"],
        "protocolos_ativos": [{"protocolo": "PROT-ALE-007", "aberto_em": h(198)}],
        "sinais_vitais": [sinais(190, 86, 52, 126, 30, 36.9, 91), sinais(45, 112, 70, 92, 18, 36.7, 97)],
        "exames": [exame("ECG", "Eletrocardiograma 12 derivacoes", "liberado", 180, "Ritmo sinusal", None, None)],
        "eventos": [{"tipo": "adrenalina_administrada", "ocorreu": True, "registrado_em": h(196)},
                    {"tipo": "registro_alergia_atualizado", "ocorreu": False, "registrado_em": None}],
        "evolucoes": [
            {"data_hora": h(196), "autor": "Dr. Andre Salgado Ferreira (CRM-SP 122.318)",
             "texto": "Beatriz Nunes Carvalho, 35 anos, quadro compativel com anafilaxia apos dipirona. Adrenalina intramuscular administrada. Contato de emergencia: irmao Rafael Carvalho, (11) 97001-3322."},
        ],
    },
]

PACIENTES += [
    {
        "prontuario": "443902",
        "nome": "Sebastiao Rodrigues Pinto",
        "cpf": "739.118.402-77",
        "cns": "705 2211 4433 6655",
        "rg": "08.774.215-2",
        "data_nascimento": "1954-06-14",
        "sexo": "M",
        "telefone": "(11) 94411-7788",
        "email": "sebastiao.pinto@provedorexemplo.com.br",
        "endereco": "Rua do Comercio, 512 - Diadema/SP - CEP 09910-220",
        "unidade": "Enfermaria Clinica",
        "leito": "ENF-118",
        "data_admissao": h(60 * 74),
        "motivo_internacao": "Celulite extensa em membro inferior direito",
        "medico_responsavel": "Dra. Juliana Esteves Rocha (CRM-SP 101.884)",
        "comorbidades": ["Obesidade grau II", "Insuficiencia venosa cronica"],
        "alergias": [],
        "medicacoes_em_uso": ["Cefazolina"],
        "protocolos_ativos": [],
        "sinais_vitais": [sinais(240, 128, 78, 84, 18, 36.9, 96)],
        "exames": [
            exame("HMG", "Hemograma completo", "liberado", 60 * 72, "Leucocitos 13.200/mm3", "/mm3", "4.000 a 11.000"),
            exame("CREA", "Creatinina", "liberado", 60 * 72, "1,0", "mg/dL", "0,6 a 1,2"),
        ],
        "eventos": [{"tipo": "avaliacao_risco_tev", "ocorreu": False, "registrado_em": None}],
        "evolucoes": [
            {"data_hora": h(60 * 74), "autor": "Dra. Juliana Esteves Rocha (CRM-SP 101.884)",
             "texto": "Sebastiao Rodrigues Pinto, 72 anos, 131 kg, internado por celulite. Mobilidade reduzida no leito. RG 08.774.215-2."},
        ],
    },
    {
        "prontuario": "310445",
        "nome": "Vanessa Ribeiro Duarte",
        "cpf": "512.667.330-18",
        "cns": "706 9988 7766 5544",
        "rg": "36.550.101-8",
        "data_nascimento": "1983-12-21",
        "sexo": "F",
        "telefone": "(11) 98120-4477",
        "email": "vanessa.duarte83@provedorexemplo.com.br",
        "endereco": "Rua Piaui, 240, apto 14 - Sao Paulo/SP - CEP 01243-001",
        "unidade": "Enfermaria Clinica",
        "leito": "ENF-204",
        "data_admissao": h(60 * 122),
        "motivo_internacao": "Pielonefrite aguda",
        "medico_responsavel": "Dra. Renata Lopes Camargo (CRM-SP 78.902)",
        "comorbidades": [],
        "alergias": ["Sulfametoxazol"],
        "medicacoes_em_uso": ["Ceftriaxona"],
        "protocolos_ativos": [],
        "sinais_vitais": [sinais(180, 118, 74, 78, 17, 36.5, 98)],
        "exames": [
            exame("URO", "Urocultura", "liberado", 60 * 120, "Escherichia coli sensivel a cefalosporina de 1a geracao", None, None, critico=True),
            exame("CREA", "Creatinina", "liberado", 60 * 24, "0,9", "mg/dL", "0,6 a 1,2"),
        ],
        "eventos": [{"tipo": "reavaliacao_antimicrobiano", "ocorreu": False, "registrado_em": None}],
        "evolucoes": [
            {"data_hora": h(60 * 24), "autor": "Dra. Renata Lopes Camargo (CRM-SP 78.902)",
             "texto": "Vanessa Ribeiro Duarte, 42 anos, afebril ha 48 horas, em ceftriaxona desde a admissao. Urocultura com E. coli sensivel a espectro mais estreito."},
        ],
    },
    {
        "prontuario": "208731",
        "nome": "Nelson Aparecido Ramos",
        "cpf": "934.500.117-62",
        "cns": "707 3344 2211 8899",
        "rg": "22.114.909-5",
        "data_nascimento": "1946-02-03",
        "sexo": "M",
        "telefone": "(11) 93377-2244",
        "email": "nelson.ramos46@provedorexemplo.com.br",
        "endereco": "Travessa das Rosas, 8 - Maua/SP - CEP 09370-410",
        "unidade": "Enfermaria Clinica",
        "leito": "ENF-133",
        "data_admissao": h(60 * 50),
        "motivo_internacao": "Descompensacao de insuficiencia renal cronica",
        "medico_responsavel": "Dr. Ricardo Almeida Sobral (CRM-SP 118.442)",
        "comorbidades": ["Doenca renal cronica estagio 4", "Hipertensao arterial", "Diabetes mellitus tipo 2"],
        "alergias": [],
        "medicacoes_em_uso": ["Furosemida", "Insulina NPH"],
        "protocolos_ativos": [],
        "sinais_vitais": [sinais(200, 152, 88, 76, 18, 36.3, 95)],
        "exames": [
            exame("CREA", "Creatinina", "liberado", 60 * 48, "3,4", "mg/dL", "0,6 a 1,2", critico=True),
            exame("CLCR", "Clearance de creatinina estimado", "liberado", 60 * 48, "22", "mL/min", "acima de 60", critico=True),
            exame("K", "Potassio serico", "liberado", 60 * 6, "5,4", "mEq/L", "3,5 a 5,0", critico=True),
        ],
        "eventos": [{"tipo": "avaliacao_risco_tev", "ocorreu": True, "registrado_em": h(60 * 49)}],
        "evolucoes": [
            {"data_hora": h(60 * 48), "autor": "Dr. Ricardo Almeida Sobral (CRM-SP 118.442)",
             "texto": "Nelson Aparecido Ramos, 80 anos, clearance estimado de 22 mL/min. Padua 5 na admissao, profilaxia farmacologica indicada com necessidade de ajuste por funcao renal."},
        ],
    },
    {
        "prontuario": "117256",
        "nome": "Celia Maria Tavares",
        "cpf": "180.442.775-90",
        "cns": "708 6677 5544 3322",
        "rg": "44.902.118-0",
        "data_nascimento": "1965-04-09",
        "sexo": "F",
        "telefone": "(11) 96688-0011",
        "email": "celia.tavares@provedorexemplo.com.br",
        "endereco": "Rua Ipiranga, 1100 - Sao Bernardo do Campo/SP - CEP 09720-330",
        "unidade": "Enfermaria Cirurgica",
        "leito": "CIR-309",
        "data_admissao": h(60 * 30),
        "motivo_internacao": "Pos-operatorio de colecistectomia videolaparoscopica",
        "medico_responsavel": "Dra. Juliana Esteves Rocha (CRM-SP 101.884)",
        "comorbidades": ["Hipotireoidismo"],
        "alergias": [],
        "medicacoes_em_uso": ["Levotiroxina"],
        "protocolos_ativos": [],
        "sinais_vitais": [sinais(90, 122, 76, 72, 16, 36.4, 98)],
        "exames": [exame("HMG", "Hemograma completo", "liberado", 60 * 28, "Leucocitos 8.100/mm3", "/mm3", "4.000 a 11.000")],
        "eventos": [{"tipo": "avaliacao_risco_tev", "ocorreu": True, "registrado_em": h(60 * 29)}],
        "evolucoes": [
            {"data_hora": h(60 * 12), "autor": "Dra. Juliana Esteves Rocha (CRM-SP 101.884)",
             "texto": "Celia Maria Tavares evolui bem no pos-operatorio, deambulando, aceitando dieta. Sem queixas."},
        ],
    },
    {
        "prontuario": "875431",
        "nome": "Osvaldo Pereira Gomes",
        "cpf": "664.031.228-14",
        "cns": "709 1122 9988 7766",
        "rg": "17.885.443-9",
        "data_nascimento": "1948-08-27",
        "sexo": "M",
        "telefone": "(11) 92244-6677",
        "email": "osvaldo.gomes48@provedorexemplo.com.br",
        "endereco": "Rua Sete de Setembro, 44 - Sao Caetano do Sul/SP - CEP 09520-120",
        "unidade": "Enfermaria Clinica",
        "leito": "ENF-221",
        "data_admissao": h(60 * 150),
        "motivo_internacao": "Pneumonia adquirida na comunidade em resolucao",
        "medico_responsavel": "Dra. Patricia Gomes Vieira (CRM-SP 96.733)",
        "comorbidades": ["Hipertensao arterial"],
        "alergias": [],
        "medicacoes_em_uso": ["Amoxicilina com clavulanato", "Losartana"],
        "protocolos_ativos": [{"protocolo": "PROT-PNM-006", "aberto_em": h(60 * 150)}],
        "sinais_vitais": [
            sinais(60 * 30, 118, 72, 88, 20, 37.2, 94),
            sinais(120, 124, 78, 82, 18, 36.5, 96),
        ],
        "exames": [
            exame("RXT", "Radiografia de torax", "liberado", 60 * 148, "Consolidacao em lobo inferior esquerdo", None, None),
            exame("PCR", "Proteina C reativa", "liberado", 60 * 12, "24", "mg/L", "ate 5"),
        ],
        "eventos": [{"tipo": "reavaliacao_antimicrobiano", "ocorreu": True, "registrado_em": h(60 * 100)}],
        "evolucoes": [
            {"data_hora": h(120), "autor": "Dra. Patricia Gomes Vieira (CRM-SP 96.733)",
             "texto": "Osvaldo Pereira Gomes, 77 anos, afebril ha mais de 48 horas, saturando 96 por cento em ar ambiente, aceitando dieta por via oral, orientado."},
        ],
    },
    {
        "prontuario": "624190",
        "nome": "Priscila Amaral Bastos",
        "cpf": "473.882.610-05",
        "cns": "710 4455 6677 8899",
        "rg": "39.114.500-7",
        "data_nascimento": "1978-10-15",
        "sexo": "F",
        "telefone": "(11) 91177-5533",
        "email": "priscila.bastos@provedorexemplo.com.br",
        "endereco": "Rua Alvorada, 62, apto 91 - Sao Paulo/SP - CEP 04578-020",
        "unidade": "Unidade de Terapia Intensiva",
        "leito": "UTI-02",
        "data_admissao": h(60 * 8),
        "motivo_internacao": "Choque septico de foco abdominal apos apendicectomia",
        "medico_responsavel": "Dr. Ricardo Almeida Sobral (CRM-SP 118.442)",
        "comorbidades": [],
        "alergias": ["Morfina"],
        "medicacoes_em_uso": ["Noradrenalina", "Piperacilina com tazobactam"],
        "protocolos_ativos": [{"protocolo": "PROT-SEP-001", "aberto_em": h(60 * 8)},
                              {"protocolo": "PROT-UTI-002", "aberto_em": h(60 * 7)}],
        "sinais_vitais": [sinais(60 * 7, 78, 44, 132, 30, 38.2, 90), sinais(30, 104, 62, 104, 22, 37.4, 95)],
        "exames": [
            exame("LAC", "Lactato arterial", "liberado", 60 * 8, "5,2", "mmol/L", "abaixo de 2", critico=True),
            exame("LAC2", "Lactato arterial - controle", "liberado", 60 * 4, "3,1", "mmol/L", "abaixo de 2", critico=True),
            exame("LAC3", "Lactato arterial - controle 2", "pendente", 90, critico=True),
            exame("HMC1", "Hemocultura - amostra 1", "liberado", 60 * 8, "Em processamento", None, None),
        ],
        "eventos": [{"tipo": "antimicrobiano_administrado", "ocorreu": True, "registrado_em": h(60 * 8 - 35)}],
        "evolucoes": [
            {"data_hora": h(60 * 7), "autor": "Dr. Ricardo Almeida Sobral (CRM-SP 118.442)",
             "texto": "Priscila Amaral Bastos, 47 anos, prontuario 624190, em choque septico de foco abdominal. Em noradrenalina. Marido, Sr. Diego Bastos, telefone (11) 91177-5533, ciente da gravidade."},
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", encoding="utf-8") as f:
        for paciente in PACIENTES:
            f.write(json.dumps(paciente, ensure_ascii=False) + "\n")

    print(f"{len(PACIENTES)} prontuarios sinteticos gravados em {args.saida}")


if __name__ == "__main__":
    main()
