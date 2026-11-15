.PHONY: ajuda instalar preparar treinar avaliar testar cobertura lint app perguntar limpar \
	baixar-dados azure monitorar avaliar-anomalias

PY := .venv/bin/python
PIP := .venv/bin/pip

ajuda:
	@echo "Alvos disponiveis:"
	@echo "  instalar   Cria o ambiente virtual e instala as dependencias de treino"
	@echo "  preparar   Gera prontuarios sinteticos, banco e dataset de fine-tuning"
	@echo "  treinar    Executa o fine-tuning por LoRA (backend MLX)"
	@echo "  avaliar    Avalia o modelo ajustado e compara com o modelo base"
	@echo "  testar     Roda a suite de testes"
	@echo "  cobertura  Roda a suite com relatorio de cobertura"
	@echo "  lint       Roda o ruff"
	@echo "  app        Sobe a interface Streamlit"
	@echo "  perguntar  Ex.: make perguntar P='Ha pendencia no leito PS-07?'"
	@echo ""
	@echo "Fase 4 - monitoramento multimodal:"
	@echo "  baixar-dados       Baixa BIDMC e UR Fall (PhysioNet / Univ. de Rzeszow)"
	@echo "  azure              Verifica as credenciais Azure (Speech e Language)"
	@echo "  monitorar          Simula 24 h de monitoramento com alertas (A=3600 acelera)"
	@echo "  avaliar-anomalias  Avalia os detectores e grava avaliacao/resultados/anomalias.md"

instalar:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-treino.txt -r requirements-dev.txt

preparar:
	$(PY) -m assistente_medico.cli preparar --forcar

treinar:
	$(PY) -m assistente_medico.finetuning.treinar

avaliar:
	$(PY) -m assistente_medico.finetuning.avaliar --comparar-base

testar:
	$(PY) -m pytest

cobertura:
	$(PY) -m pytest --cov=assistente_medico --cov-report=term-missing

lint:
	.venv/bin/ruff check assistente_medico tests scripts

app:
	.venv/bin/streamlit run assistente_medico/app/streamlit_app.py \
		--server.fileWatcherType none \
		--browser.gatherUsageStats false

perguntar:
	$(PY) -m assistente_medico.cli perguntar "$(P)" --explicar

limpar:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ logs/*.log logs/*.jsonl

baixar-dados:
	$(PY) scripts/baixar_datasets.py --todos

azure:
	$(PY) -m assistente_medico.cli azure

monitorar:
	$(PY) -m assistente_medico.cli monitorar simular --horas 24 --aceleracao $(or $(A),0)

avaliar-anomalias:
	$(PY) -m assistente_medico.cli monitorar avaliar
