.PHONY: ajuda instalar preparar treinar avaliar testar cobertura lint app perguntar limpar

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
	.venv/bin/streamlit run assistente_medico/app/streamlit_app.py

perguntar:
	$(PY) -m assistente_medico.cli perguntar "$(P)" --explicar

limpar:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ logs/*.log logs/*.jsonl
