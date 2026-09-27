import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, fbeta_score
from collections import defaultdict

def f05_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5)

def entity_level_f05(val_df, preds, threshold, gt):
    """
    Compute EXACT competition metric: macro-averaged F0.5 per S1 entity.
    This is what the leaderboard actually measures.
    """
    val_df = val_df.copy()
    val_df['pred_prob'] = preds
    matched = val_df[val_df['pred_prob'] > threshold][['source1_entity_id', 'candidate_entity_id']]
    
    pred_dict = defaultdict(set)
    for row in matched.itertuples(index=False):
        pred_dict[row.source1_entity_id].add(row.candidate_entity_id)
    
    # All S1 entities in val set
    all_s1 = val_df['source1_entity_id'].unique()
    
    scores = []
    for s1_id in all_s1:
        true_set = gt.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())
        
        if len(true_set) == 0 and len(pred_set) == 0:
            scores.append(1.0)  # Correct singleton prediction
        elif len(pred_set) == 0:
            scores.append(0.0)  # Missed all true matches
        else:
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec = tp / len(true_set) if true_set else 0.0
            if prec + rec == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                scores.append(f05)
    
    return np.mean(scores)

def train_model():
    base_dir = 'output'
    print("Loading features...")
    df = pd.read_parquet(os.path.join(base_dir, 'train_features.parquet'))
    
    # Define features
    features = ['name_lev', 'name_jaro', 'name_token_sort', 'name_token_set', 'name_partial', 'addr_lev', 'addr_token_set', 'addr_missing']
    
    # We want to use GroupKFold to ensure the same S1 entity isn't in both train and validation
    gkf = GroupKFold(n_splits=5)
    
    groups = df['source1_entity_id'].values
    X = df[features]
    y = df['label']
    
    # We will just do a simple 80/20 train/val split for threshold tuning
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        break
        
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"Positive samples in train: {y_train.sum()}, in val: {y_val.sum()}")
    
    # LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'device': 'gpu'
    }
    
    print("Training LightGBM model...")
    # Use callbacks instead of early_stopping_rounds in lgb.train
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    
    print("Predicting on validation set...")
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # -----------------------------------------------------------------------
    # CRITICAL: Tune threshold on ENTITY-LEVEL macro F0.5
    # This is the EXACT metric used by the leaderboard — not pair-level F0.5!
    # -----------------------------------------------------------------------
    print("Building ground truth dict for entity-level scoring...")
    val_df = df.iloc[val_idx][['source1_entity_id', 'candidate_entity_id', 'label']].copy()
    
    # Build true match sets per S1 entity
    gt_dict = defaultdict(set)
    for row in val_df[val_df['label'] == 1].itertuples(index=False):
        gt_dict[row.source1_entity_id].add(row.candidate_entity_id)
    gt_dict = dict(gt_dict)
    
    best_threshold = 0.5
    best_f05 = 0.0
    
    print("Tuning threshold on ENTITY-LEVEL macro F0.5 (exact competition metric)...")
    for t in np.arange(0.1, 0.95, 0.01):
        f05 = entity_level_f05(val_df, preds, t, gt_dict)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t
            
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Validation Entity-Level Macro F0.5: {best_f05:.4f}")
    
    # Save model and threshold
    model_path = os.path.join(base_dir, 'lgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': best_threshold}, f)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
