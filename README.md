# Sentiment Analysis Production Lab

End-to-end NLP lab comparing Amazon Fine Food Reviews with Sentiment140. The project covers
normalization, tokenization, stemming, lemmatization, BoW, TF-IDF, BM25, experiment tracking,
retrieval, and API deployment.

## What is included

- Reusable preprocessing: `src.preprocessing.TextPreprocessor`
- Dataset loaders for Amazon and Sentiment140: `src.data`
- BoW, TF-IDF, and BM25 training: `src.train`
- BM25 retrieval with positive/negative filtering: `src.search`
- FastAPI service for `/predict`, `/search`, and `/health`: `src.api`
- DVC pipeline stages in `dvc.yaml`
- Docker packaging in `Dockerfile`

## Data

Place the raw files here:

- `data/raw/Reviews.csv`
- `data/raw/training.1600000.processed.noemoticon.csv`

The current workspace already contains both files. After installing DVC, initialize tracking:

```powershell
dvc init
dvc add data/raw/Reviews.csv data/raw/training.1600000.processed.noemoticon.csv
git add .dvc .dvcignore data/raw/*.dvc
```

## Setup

```powershell
uv sync
```

Or with pip:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Experiments

Run one experiment:

```powershell
python -m src.train --dataset amazon --vectorizer tfidf --reduction lemma --sample 50000
```

Compare vectorizers:

```powershell
python -m src.train --dataset amazon --vectorizer bow --reduction lemma
python -m src.train --dataset amazon --vectorizer tfidf --reduction lemma
python -m src.train --dataset amazon --vectorizer bm25 --reduction lemma
python -m src.train --dataset sentiment140 --vectorizer bow --reduction lemma
python -m src.train --dataset sentiment140 --vectorizer tfidf --reduction lemma
python -m src.train --dataset sentiment140 --vectorizer bm25 --reduction lemma
```

If MLflow is installed, runs are logged to the `sentiment-lab` experiment automatically:

```powershell
mlflow ui
```

## Build Search

```powershell
python -m src.search --sample 500000
```

This writes `models/amazon_bm25_search.joblib`.

## Reproduce With DVC

```powershell
dvc repro
```

The default DVC stages train TF-IDF baselines for both datasets and build the Amazon BM25
search index.

## Serve API

Train an Amazon model and build the search index first, then run:

```powershell
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Example requests:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType application/json -Body '{"text":"This coffee tastes fresh and smells amazing"}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/search -ContentType application/json -Body '{"query":"fresh coffee aroma","sentiment":"positive","top_k":5}'
```

## Docker

```powershell
docker build -t sentiment-lab-api .
docker run --rm -p 8000:8000 sentiment-lab-api
```

The image expects trained artifacts under `models/` before build time.
