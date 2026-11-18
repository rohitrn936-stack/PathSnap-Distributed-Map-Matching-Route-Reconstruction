# PathSnap
### Distributed Map-Matching and Route Reconstruction

PathSnap reconstructs road routes from noisy GPS observations by matching GPS points to a road network and selecting connected road segments.

## Features
- GPS candidate generation using spatial search
- Dynamic Programming (DP) route reconstruction
- Road-network connectivity analysis and intermediate arc insertion
- Hidden Markov Model (HMM) with Viterbi route reconstruction
- DP vs HMM comparison
- FastAPI backend
- Interactive Leaflet map dashboard

## Technology Stack
- Python
- NumPy, SciPy, Pandas
- FastAPI and Uvicorn
- HTML, JavaScript and Leaflet

## Results
- 80 training trips processed successfully
- 199,077 GPS points matched during DP reconstruction
- 99.9662% network connectivity after intermediate arc insertion
- 98 unresolved transitions remained
- 56 trips achieved 100% connectivity

## Setup

Place the dataset in:

`data/Student folder/train/`

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Backend

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Access
- Dashboard: http://127.0.0.1:8000/dashboard/
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Notes
Raw datasets and large generated route files are excluded from Git using `.gitignore`. The dataset and generated route outputs must be available locally for complete dashboard functionality.

Bayesian optimization is reserved for future work.