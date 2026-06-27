# ════════════════════════════════════════════════════════════════════════
#  Makefile — interface unique du pipeline LLMOps du corpus wolof.
#  Toutes les commandes s'exécutent dans src/ avec le venv uv (.venv).
#
#  Usage :  make help
# ════════════════════════════════════════════════════════════════════════

PY := .venv/bin/python
RUN := cd src && $(PY) -m pipeline.run

.DEFAULT_GOAL := help

.PHONY: help setup lint test ingest centralize merge stats datacard validate \
        publish all clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Installe les dépendances (uv) + outils dev
	cd src && uv sync --extra dev

lint:  ## Lint du code (ruff)
	cd src && uv run ruff check pipeline corpus_stats.py ingest_hf_datasets.py \
	  centralize_ingested.py merge_corpora.py

test:  ## Tests unitaires (pytest)
	cd src && uv run pytest -q

ingest:  ## [1] Ingestion des 17 datasets HF (resumable)
	$(RUN) ingest

centralize:  ## [2] Centralise les jsonl -> corpus ingéré
	$(RUN) centralize

merge:  ## [3] Fusionne corpus HF + ingéré -> unifié (dédup)
	$(RUN) merge

stats:  ## [4] Statistiques (volumétrie + LID GlotLID)
	$(RUN) stats

datacard:  ## [5] Régénère la data card (README)
	$(RUN) datacard

validate:  ## [6] Quality gates (BLOQUANT)
	$(RUN) validate

publish:  ## [7] Publie vers HF (refusé si gates KO)
	$(RUN) publish

all:  ## Pipeline complet (sans publication)
	$(RUN) all

clean:  ## Supprime les caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; \
	rm -rf src/.pytest_cache src/.ruff_cache
