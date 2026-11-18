# PathSnap: Member 2 ML workspace

**Scope**: 5-model candidate classification, trip-grouped CV, genuine student-test evaluation, model serialization, Optuna Bayesian search. **Not a full PathSnap submission.** Requires candidate and observation-level label exports from Member 1.

## VS Code on Windows (PowerShell)
1. Open the extracted `PathSnap_Member2` folder in VS Code.
2. Install the Microsoft Python extension; choose Python 3.11 or 3.12.
3. Open **Terminal > New Terminal**, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell refuses activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and invoke Python with that full path, or switch VS Code terminal to Command Prompt and run `.venv\Scripts\activate.bat`.

4. **Smoke test only** (synthetic, never report as real accuracy):

```powershell
python make_demo_data.py
python ml_pipeline.py --candidates data/demo_candidates.csv --labels data/demo_labels.csv --output outputs/demo --folds 3
```

5. **When Member 1 sends verified CSVs**:

```powershell
python ml_pipeline.py --candidates data/candidates.csv --labels data/labels.csv --output outputs/real --folds 3 --tune --trials 15
```

## Data contract to share with Member 1

- `candidates.csv`: `split,trip_id,observation_idx,arc_id,distance_m,heading_diff_deg,...` one row **per road candidate**. The extra feature columns must all be numeric. `split` values `train` and `student_test` only. GPS observation index is zero-based, arc ID is **trip-scoped**. Distances must be meters, not raw longitude/latitude degrees.
- `labels.csv`: `split,trip_id,observation_idx,reference_arc_id` one row per **labeled GPS observation**, including observations with no candidate. Labels must come from a **verified alignment process**, not naive row matching between `.track` and `.route`. `arc_id` and `reference_arc_id` must use the same road-arc ID convention.
- The candidate CSV must not contain `target`, `label`, `reference_arc_id`, or other ground-truth information. The classifier must never see reference information as features.
- At least two training trips needed to exercise grouped cross-validation; actual competition expects 80 train and 10 student-test trips.
- Every candidate observation must have a label. Labels may also include observations with zero candidates, which count as misses in Top-1 evaluation.

## Outputs
- `model_comparison.csv`: 5 real model comparisons after real data is supplied.
- `cv_folds.csv`: trip-grouped fold metrics.
- `feature_order.json`, `data_summary.json`: versioned schema and counts.
- `*.joblib`: fitted model+imputation+scaling pipeline.
- `selected_model.txt`: model selected by **train CV macro F1**.
- Optional `optuna_trials.csv`, `tuning_summary.json`, `best_tuned_model.joblib`.

### Metric caveats
Candidate-level macro F1, precision, recall, log loss are computed per candidate row. Top-1 accuracy is computed over *all labeled observations* in the split, including observations for which no candidate exists. Candidate miss rate uses the same denominator. CV selects models using training folds; student-test metrics are reported independently. Route continuity and HMM/Viterbi improvement require Member 3's decoder and are not fabricated here. Grouped CV keeps all observations from the same trip in the same fold.

**Academic integrity**: Treat demo results as code testing only. Run actual experiments independently, review the baseline/search choices, preserve logs, disclose AI assistance if required, and be able to defend methods and results.
