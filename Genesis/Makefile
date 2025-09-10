# Fantasy RAG - Makefile
# Commandes d'automatisation

.PHONY: help install test lint format run-web run-streamlit clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances
	pip install -r config/requirements/base.txt

install-dev:  ## Installe les dépendances de développement
	pip install -r config/requirements/development.txt

test:  ## Lance tous les tests
	python -m pytest tests/ -v

test-unit:  ## Lance les tests unitaires
	python -m pytest tests/unit/ -v

test-integration:  ## Lance les tests d'intégration
	python -m pytest tests/integration/ -v

lint:  ## Vérifie le code avec flake8
	flake8 fantasy_rag/ apps/ tools/

format:  ## Formate le code avec black
	black fantasy_rag/ apps/ tools/

run-web:  ## Lance l'application web
	python apps/web/app.py

run-streamlit:  ## Lance l'application Streamlit
	streamlit run apps/streamlit/main.py

run-cli:  ## Lance l'interface CLI
	python apps/cli/fantasy_cli.py --help

setup:  ## Configuration initiale
	python tools/setup/install_dependencies.py

clean:  ## Nettoie les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf data/cache/*

demo:  ## Lance une démonstration
	python demos/basic/demo_search.py

health:  ## Vérifie la santé du système
	python tools/debug/validate_system.py
