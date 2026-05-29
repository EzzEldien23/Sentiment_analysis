# Sentiment Analysis Production Lab

End-to-end NLP lab comparing Amazon Fine Food Reviews with Sentiment140. The project covers
normalization, tokenization, stemming, lemmatization, BoW, TF-IDF, BM25, experiment tracking,
retrieval, and API deployment.

## DagsHub, DVC, and MLflow Setup

The repository is configured for:

- DagsHub repository: `https://dagshub.com/ezzeldiennassar/Sentiment_analysis`
- DVC remote: `s3://dvc`
- DagsHub S3 endpoint: `https://dagshub.com/ezzeldiennassar/Sentiment_analysis.s3`
- MLflow tracking URI: `https://dagshub.com/ezzeldiennassar/Sentiment_analysis.mlflow`

Install dependencies:

```powershell
uv sync
```

Configure DVC credentials locally. Do not commit `.dvc/config.local`.

```powershell
uv run dvc remote add origin s3://dvc
uv run dvc remote modify origin endpointurl https://dagshub.com/ezzeldiennassar/Sentiment_analysis.s3
uv run dvc remote modify origin --local access_key_id <DAGSHUB_TOKEN>
uv run dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>
uv run dvc remote default origin
```

Configure MLflow for the current Windows `cmd.exe` session:

```cmd
set DAGSHUB_TOKEN=<DAGSHUB_TOKEN>
set MLFLOW_TRACKING_URI=https://dagshub.com/ezzeldiennassar/Sentiment_analysis.mlflow
set MLFLOW_TRACKING_USERNAME=ezzeldiennassar
set MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>
```

For PowerShell:

```powershell
$env:DAGSHUB_TOKEN="<DAGSHUB_TOKEN>"
$env:MLFLOW_TRACKING_URI="https://dagshub.com/ezzeldiennassar/Sentiment_analysis.mlflow"
$env:MLFLOW_TRACKING_USERNAME="ezzeldiennassar"
$env:MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"
```

Run the reproducible pipeline and push artifacts:

```powershell
uv run dvc repro
uv run dvc push
git add .dvc/config .dvc/.gitignore .gitignore dvc.yaml params.yaml README.md pyproject.toml requirements.txt src
git commit -m "Configure DagsHub DVC and MLflow tracking"
git push
```

Every training run logs to DagsHub MLflow when `mlflow` is installed. The training scripts log
dataset, vectorizer, reduction, sample size, accuracy, precision, recall, F1 score, vocabulary
size, training time, generated reports, and trained model artifacts.

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
