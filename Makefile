# Fantasy RAG - Makefile
# Commandes d'automatisation

PYTHON ?= python

.PHONY: help install install-dev install-gen test lint format run clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances de base (recherche)
	$(PYTHON) -m pip install -r config/requirements/base.txt

install-dev:  ## Installe les dépendances de développement
	$(PYTHON) -m pip install -r config/requirements/development.txt

install-gen:  ## Installe les dépendances de génération 3D (GPU NVIDIA)
	$(PYTHON) -m pip install -r config/requirements/generation.txt

test:  ## Lance les tests
	$(PYTHON) -m pytest tests/ -v

lint:  ## Vérifie le code avec flake8
	$(PYTHON) -m flake8 fantasy_rag/ apps/ tools/

format:  ## Formate le code avec black
	$(PYTHON) -m black fantasy_rag/ apps/ tools/

run:  ## Lance l'interface Streamlit (recherche + génération 3D)
	$(PYTHON) -m streamlit run apps/streamlit/app.py

clean:  ## Supprime les caches Python et pytest
	find . -path ./.venv -prune -o -path ./third_party -prune -o -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache/
