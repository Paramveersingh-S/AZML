import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, fbeta_score

def f05_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5)

def train_model():
    base_dir = 'output'
    print("Loading features...")
    df = pd.read_csv(os.path.join(base_dir, 'train_features.csv'))
    
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
        'verbose': -1
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
    
    # Tune threshold for F_0.5
    best_threshold = 0.5
    best_f05 = 0.0
    
    print("Tuning threshold for F_0.5...")
    for t in np.arange(0.1, 0.9, 0.02):
        y_pred = (preds > t).astype(int)
        f05 = f05_score(y_val, y_pred)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t
            
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Validation F0.5 Score: {best_f05:.4f}")
    
    # Save model and threshold
    model_path = os.path.join(base_dir, 'lgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': best_threshold}, f)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
