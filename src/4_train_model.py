import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle
from sklearn.model_selection import GroupKFold
from collections import defaultdict


def entity_level_f05(val_df, preds, threshold, gt):
    """
    Compute EXACT competition metric: macro-averaged F0.5 per S1 entity.
    This is what the leaderboard measures — NOT pair-level binary F0.5.
    """
    val_df = val_df.copy()
    val_df['pred_prob'] = preds
    matched = val_df[val_df['pred_prob'] > threshold][['source1_entity_id', 'candidate_entity_id']]

    pred_dict = defaultdict(set)
    for row in matched.itertuples(index=False):
        pred_dict[row.source1_entity_id].add(row.candidate_entity_id)

    all_s1 = val_df['source1_entity_id'].unique()
    scores = []
    for s1_id in all_s1:
        true_set = gt.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        if len(true_set) == 0 and len(pred_set) == 0:
            scores.append(1.0)   # Correctly predicted singleton
        elif len(true_set) == 0 and len(pred_set) > 0:
            scores.append(0.0)   # False positive on true singleton
        elif len(pred_set) == 0:
            scores.append(0.0)   # Missed all true matches
        else:
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set)
            rec = tp / len(true_set)
            if prec + rec == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                scores.append(f05)

    return float(np.mean(scores))


def train_model():
    base_dir = 'output'
    print("Loading features...")
    df = pd.read_parquet(os.path.join(base_dir, 'train_features.parquet'))

    # All 15 features — every one contributes to score
    features = [
        'name_lev', 'name_jaro', 'name_token_sort', 'name_token_set', 'name_partial',
        'addr_lev', 'addr_token_set', 'addr_missing',
        'tfidf_score', 'name_len_ratio', 'name_first_token_match',
        'name_first_token_lev', 'addr_num_overlap', 'is_s2', 'name_exact',
    ]
    # Only use features that exist in this run (handles backward compat)
    features = [f for f in features if f in df.columns]
    print(f"Using features: {features}")

    gkf = GroupKFold(n_splits=5)
    groups = df['source1_entity_id'].values
    X = df[features].astype(np.float32)
    y = df['label'].astype(np.int8)

    # Use first fold for val (80% train, 20% val, no S1 entity overlap)
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        break

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    pos_train = int(y_train.sum())
    neg_train = len(y_train) - pos_train
    print(f"Positive samples in train: {pos_train} ({pos_train/len(y_train):.3%})")
    print(f"Class imbalance ratio: {neg_train/pos_train:.1f}:1")

    # Class imbalance: ~25 negatives per positive
    # scale_pos_weight compensates so model doesn't just predict "no match" for everything
    scale_pos_weight = neg_train / pos_train

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        # More leaves = more complex decision boundaries = better fit
        # 127 is proven sweet spot for entity resolution tasks
        'num_leaves': 127,
        'max_depth': 8,
        # Regularization to prevent overfitting on the 25:1 imbalanced data
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        # CRITICAL: tells LightGBM the positive class is rare, prevents predicting all zeros
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'device': 'gpu',
    }

    print("Training LightGBM model...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        callbacks=callbacks,
    )

    print("Predicting on validation set...")
    preds = model.predict(X_val, num_iteration=model.best_iteration)

    # Feature importance — helps understand what's driving predictions
    importances = sorted(zip(features, model.feature_importance(importance_type='gain')),
                         key=lambda x: x[1], reverse=True)
    print("\nFeature Importances (by gain):")
    for feat, imp in importances:
        print(f"  {feat:30s}: {imp:.1f}")

    # -----------------------------------------------------------------------
    # EXACT COMPETITION METRIC: entity-level macro F0.5
    # -----------------------------------------------------------------------
    print("\nBuilding ground truth dict for entity-level scoring...")
    val_df = df.iloc[val_idx][['source1_entity_id', 'candidate_entity_id', 'label']].copy()

    gt_dict = defaultdict(set)
    for row in val_df[val_df['label'] == 1].itertuples(index=False):
        gt_dict[row.source1_entity_id].add(row.candidate_entity_id)
    gt_dict = dict(gt_dict)

    best_threshold = 0.5
    best_f05 = 0.0
    threshold_results = []

    print("Tuning threshold on ENTITY-LEVEL macro F0.5...")
    for t in np.arange(0.05, 0.99, 0.01):
        f05 = entity_level_f05(val_df, preds, t, gt_dict)
        threshold_results.append((t, f05))
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t

    print(f"\nBest Threshold: {best_threshold:.3f}")
    print(f"Validation Entity-Level Macro F0.5: {best_f05:.4f}")

    # Show curve around best threshold for insight
    print("\nThreshold curve (±0.05 around best):")
    for t, f05 in threshold_results:
        if abs(t - best_threshold) <= 0.05:
            marker = " ◄ BEST" if abs(t - best_threshold) < 0.001 else ""
            print(f"  t={t:.2f}  F0.5={f05:.4f}{marker}")

    # Save model and threshold
    model_path = os.path.join(base_dir, 'lgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': best_threshold, 'features': features}, f)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    train_model()
