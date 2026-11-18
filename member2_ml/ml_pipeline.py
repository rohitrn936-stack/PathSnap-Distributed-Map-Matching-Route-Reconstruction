"""PathSnap Member 2: leak-aware candidate-level ML experiments.

Inputs (CSV):
  candidates.csv: split,trip_id,observation_idx,arc_id,<numeric features>
  labels.csv:     split,trip_id,observation_idx,reference_arc_id

The label file is deliberately separate from feature input. A missing reference arc
in candidates is counted as a missed observation rather than quietly discarded.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

KEY = ['split', 'trip_id', 'observation_idx']
REQUIRED_C = KEY + ['arc_id']
REQUIRED_L = KEY + ['reference_arc_id']
SEED = 42


def load_tables(candidate_path, label_path):
    c = pd.read_csv(candidate_path, dtype={'split':'string','trip_id':'string','observation_idx':'int64','arc_id':'string'})
    l = pd.read_csv(label_path, dtype={'split':'string','trip_id':'string','observation_idx':'int64','reference_arc_id':'string'})
    for name, df, required in [('candidate', c, REQUIRED_C), ('label', l, REQUIRED_L)]:
        missing = set(required) - set(df.columns)
        if missing: raise ValueError(f'{name} CSV missing columns: {sorted(missing)}')
        if df[required].isna().any().any(): raise ValueError(f'{name} contains null keys / IDs')
    if c.duplicated(REQUIRED_C).any(): raise ValueError('Duplicate candidate (split,trip,observation,arc)')
    if l.duplicated(KEY).any(): raise ValueError('Duplicate observation label')
    if not set(c['split']).issubset({'train','student_test'}): raise ValueError('split must be train/student_test; evaluator data forbidden')
    if not set(l['split']).issubset({'train','student_test'}): raise ValueError('split must be train/student_test; evaluator data forbidden')
    if set(map(tuple,c[KEY].itertuples(index=False,name=None))) - set(map(tuple,l[KEY].itertuples(index=False,name=None))):
        raise ValueError('Candidate observations missing labels; fix alignment or record exclusions explicitly')
    train_ids = set(c.loc[c.split=='train', 'trip_id'])
    test_ids = set(c.loc[c.split=='student_test', 'trip_id'])
    if train_ids & test_ids: raise ValueError('Trip IDs overlap between splits; namespace them first')
    blocked = [x for x in c.columns if any(w in x.lower() for w in ['label','target','reference','ground_truth','is_correct'])]
    if blocked: raise ValueError(f'Possible target-leakage feature columns: {blocked}')
    features = [f for f in c.columns if f not in REQUIRED_C]
    if not features: raise ValueError('No candidate features supplied')
    if any(not pd.api.types.is_numeric_dtype(c[f]) for f in features):
        raise ValueError('All features must be numeric: ' + str([f for f in features if not pd.api.types.is_numeric_dtype(c[f])]))
    merged = c.merge(l, on=KEY, how='left', validate='many_to_one')
    merged['target'] = (merged['arc_id'] == merged['reference_arc_id']).astype(int)
    if not (merged['target'] == 1).any(): raise ValueError('No positive candidate labels. Member 1 must validate alignment.')
    return merged, l, features


def models():
    # These are deliberately initial baselines; the team must justify actual selection.
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
        'GaussianNB': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, class_weight='balanced', random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31, learning_rate=0.1, random_state=SEED),
    }


def pipeline(estimator, features):
    prep = ColumnTransformer([('numerical', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), features)], remainder='drop')
    return Pipeline([('preprocess', prep), ('model', clone(estimator))])


def evaluate(rows, labels, scores):
    rows = rows.copy()
    rows['score'] = scores
    y = rows.target.to_numpy(dtype=int)
    # Candidate-level binary probability metrics include all candidate rows.
    candidate = {
        'candidate_precision':float(precision_score(y, scores >= 0.5, zero_division=0)),
        'candidate_recall':float(recall_score(y, scores >= 0.5, zero_division=0)),
        'candidate_macro_f1':float(f1_score(y, scores >= 0.5, average='macro', zero_division=0)),
        'candidate_log_loss': float(log_loss(y, np.clip(scores, 1e-12, 1-1e-12), labels=[0,1])),
    }
    # Use all labeled observations including those with zero candidate rows.
    best_idx = rows.groupby(KEY, sort=False)['score'].idxmax()
    best = rows.loc[best_idx, KEY + ['arc_id']].rename(columns={'arc_id':'predicted_arc_id'})
    obs = labels.merge(best, on=KEY, how='left', validate='one_to_one')
    correct = obs['predicted_arc_id'].notna() & (obs['predicted_arc_id'] == obs['reference_arc_id'])
    candidate_hit = rows.groupby(KEY, sort=False)['target'].max().reset_index(name='hit')
    all_hits = labels.merge(candidate_hit, on=KEY, how='left')
    miss_rate = 1.0 - float(all_hits['hit'].fillna(0).mean())
    return {**candidate,'top1_accuracy':float(correct.mean()), 'candidate_miss_rate':miss_rate,
            'n_observations':int(len(obs)), 'n_candidate_rows':int(len(rows))}


def cv_results(train, labels, features, estimator, n_splits=3):
    groups = train['trip_id'].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2: raise ValueError('At least 2 distinct train trips required for grouped CV')
    cv = GroupKFold(n_splits=min(n_splits, n_groups))
    results = []
    for fold,(i,j) in enumerate(cv.split(train[features],train.target,groups), 1):
        tr,va = train.iloc[i],train.iloc[j]
        if tr.target.nunique() < 2: raise ValueError('Training fold lacks both classes; check candidate labels')
        p = pipeline(estimator,features)
        p.fit(tr[features], tr.target)
        pred = p.predict_proba(va[features])[:,1]
        valid_labels = labels[(labels.split=='train') & (labels.trip_id.isin(va.trip_id.unique()))]
        fold_data = evaluate(va, valid_labels, pred)
        fold_data['fold'] = fold
        fold_data['validation_trips'] = int(va.trip_id.nunique())
        results.append(fold_data)
    return results


def run(args):
    data,labels,features = load_tables(args.candidates,args.labels)
    train = data[data.split=='train'].copy().reset_index(drop=True)
    test = data[data.split=='student_test'].copy().reset_index(drop=True)
    if train.empty or test.empty: raise ValueError('Both train and student_test candidates must exist')
    train_labels=labels[labels.split=='train'].copy()
    test_labels=labels[labels.split=='student_test'].copy()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'feature_order.json').write_text(json.dumps(features,indent=2))
    (output/'data_summary.json').write_text(json.dumps({'train_trips':int(train.trip_id.nunique()),'student_test_trips':int(test.trip_id.nunique()),'train_rows':len(train),'test_rows':len(test),'features':features,'seed':SEED},indent=2))
    comparison=[];folds=[];ref_models=models()
    for name, estimator in ref_models.items():
        print(f'Running {name} ...',flush=True)
        start=time.perf_counter()
        fold=cv_results(train,train_labels,features,estimator,args.folds)
        cv_seconds=time.perf_counter()-start
        folds.extend([{'model':name,**r} for r in fold])
        p=pipeline(estimator,features)
        t0=time.perf_counter();p.fit(train[features],train.target);fit_seconds=time.perf_counter()-t0
        t0=time.perf_counter();pred=p.predict_proba(test[features])[:,1];infer_seconds=time.perf_counter()-t0
        score=evaluate(test,test_labels,pred)
        record={'model':name,'cv_macro_f1':float(np.mean([f['candidate_macro_f1'] for f in fold])),
                'cv_top1_accuracy':float(np.mean([f['top1_accuracy'] for f in fold])),
                'cv_seconds':cv_seconds,'fit_seconds':fit_seconds,'inference_seconds':infer_seconds,**score}
        comparison.append(record)
        joblib.dump(p,output/f'{name}.joblib')
        print('  ',name,'CV macro F1:',round(record['cv_macro_f1'],4),'test Top1:',round(record['top1_accuracy'],4),flush=True)
    pd.DataFrame(folds).to_csv(output/'cv_folds.csv',index=False)
    pd.DataFrame(comparison).sort_values('cv_macro_f1',ascending=False).to_csv(output/'model_comparison.csv',index=False)
    # Select based on grouped train CV only; keep test as development report, not selection source.
    winner=max(comparison,key=lambda a:a['cv_macro_f1'])['model']
    (output/'selected_model.txt').write_text(winner+'\n')
    if args.tune:
        import optuna
        # Bayesian optimization using TPE on CV macro F1 (training trips only).
        def trial_model(trial):
            if winner=='RandomForest':
                return RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators',80,200,step=40),max_depth=trial.suggest_int('max_depth',5,22),min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),n_jobs=-1,random_state=SEED,class_weight='balanced_subsample')
            if winner=='HistGradientBoosting':
                return HistGradientBoostingClassifier(max_iter=trial.suggest_int('max_iter',80,200,step=40),learning_rate=trial.suggest_float('learning_rate',0.02,0.2,log=True),max_leaf_nodes=trial.suggest_int('max_leaf_nodes',10,50),random_state=SEED)
            if winner=='LogisticRegression':
                return LogisticRegression(C=trial.suggest_float('C',1e-3,100,log=True),max_iter=1000,class_weight='balanced',random_state=SEED)
            if winner=='DecisionTree':
                return DecisionTreeClassifier(max_depth=trial.suggest_int('max_depth',3,25),min_samples_leaf=trial.suggest_int('min_samples_leaf',1,20),random_state=SEED,class_weight='balanced')
            return GaussianNB(var_smoothing=trial.suggest_float('var_smoothing',1e-12,1e-6,log=True))
        def objective(trial):
            folds=cv_results(train,train_labels,features,trial_model(trial),args.folds)
            return float(np.mean([f['candidate_macro_f1'] for f in folds]))
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
        tick=time.perf_counter();study.optimize(objective,n_trials=args.trials)
        study.trials_dataframe().to_csv(output/'optuna_trials.csv',index=False)
        tuned=trial_model(optuna.trial.FixedTrial(study.best_params))
        chosen=pipeline(tuned,features).fit(train[features],train.target)
        joblib.dump(chosen,output/'best_tuned_model.joblib')
        tuned_results=evaluate(test,test_labels,chosen.predict_proba(test[features])[:,1])
        tuned_cv=cv_results(train,train_labels,features,tuned,args.folds)
        (output/'tuning_summary.json').write_text(json.dumps({'model':winner,'best_cv_macro_f1':study.best_value,'best_params':study.best_params,'tuning_seconds':time.perf_counter()-tick,'untuned_test':next(v for v in comparison if v['model']==winner),'tuned_test':tuned_results,'tuned_cv_macro_f1':np.mean([f['candidate_macro_f1'] for f in tuned_cv])},indent=2,default=float))
    print('DONE. Best baseline by grouped CV:',winner)
    print('Outputs:',output.resolve())


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--candidates',required=True)
    a.add_argument('--labels',required=True)
    a.add_argument('--output',default='outputs')
    a.add_argument('--folds',type=int,default=3)
    a.add_argument('--tune',action='store_true')
    a.add_argument('--trials',type=int,default=15)
    run(a.parse_args())

if __name__=='__main__':main()
