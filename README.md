# Movie Recommender

Simple movie-recommender demo using the MovieLens `ml-latest-small` dataset.

Core Project: 
- `main.py` — small CLI runner that builds a user profile and prints recommendations.
- `recommender.py` — core logic for genre vectors, user profiles and recommendations.
- `similarity.py` — cosine similarity helper.

Setup

1. Create a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Download the MovieLens `ml-latest-small` dataset from GroupLens:

https://grouplens.org/datasets/movielens/

Unzip and place the folder under `data/ml-latest-small/` so the CSVs are available to the loader.

Run

```powershell
python main.py
```

Notes
- If you want to publish a lightweight public repo, keep only the core files listed above and this `README.md`.
- If you accidentally committed dataset or cache files already, remove them with the commands shown below in this README.

License

This repository uses the MIT license (see `LICENSE`).
