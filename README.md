# PathSnap

Route reconstruction dashboard backed by the supplied student-test road networks and preprocessing candidate exports.

## Run the dashboard

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000/dashboard/ . API documentation: http://127.0.0.1:8000/docs .

The dashboard uses a connected route from `results/batch_connected` when available. Otherwise it shows the nearest-candidate preprocessing baseline from `memberOneOutput/student_test`; the API labels it `nearest_candidate_baseline`.

## Other components

- `preprocessDataset.py`: candidate and observation exports from `datasets`.
- `route_reconstruction/`: graph search and Viterbi decoding utilities.
- `member2_ml/`: independent ML workspace with its own README and requirements.
