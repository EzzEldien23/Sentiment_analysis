#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sentiment_analysis
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Train a baseline Amazon TF-IDF model
.PHONY: train-amazon
train-amazon:
	$(PYTHON_INTERPRETER) -m src.train --dataset amazon --vectorizer tfidf --reduction lemma

## Train a baseline Sentiment140 TF-IDF model
.PHONY: train-sentiment140
train-sentiment140:
	$(PYTHON_INTERPRETER) -m src.train --dataset sentiment140 --vectorizer tfidf --reduction lemma

## Build the Amazon BM25 search index
.PHONY: search-index
search-index:
	$(PYTHON_INTERPRETER) -m src.search

## Create a word embedding visualization
.PHONY: embeddings
embeddings:
	$(PYTHON_INTERPRETER) -m src.embeddings

## Compare BoW, TF-IDF, BM25, and embedding classifiers
.PHONY: compare
compare:
	$(PYTHON_INTERPRETER) -m src.compare_representations

## Compare no reduction, stemming, and lemmatization
.PHONY: compare-reductions
compare-reductions:
	$(PYTHON_INTERPRETER) -m src.compare_reductions

## Show nearest words from learned distributional embeddings
.PHONY: nearest-words
nearest-words:
	$(PYTHON_INTERPRETER) -m src.nearest_words

## Run the FastAPI app locally
.PHONY: api
api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
