import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle
import glob
from collections import defaultdict

def inference():
    base_dir = 'output'
    test_clean = '../../dataset/student_resource/dataset/test_clean'
    
    # Load model and threshold
    model_path = os.path.join(base_dir, 'lgb_model.pkl')
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        threshold = data['threshold']
        
    print(f"Loaded Model. Best threshold from training: {threshold}")
    
    print("Loading test features...")
    df = pd.read_parquet(os.path.join(base_dir, 'test_features.parquet'))
    
    features = ['name_lev', 'name_jaro', 'name_token_sort', 'name_token_set', 'name_partial', 'addr_lev', 'addr_token_set', 'addr_missing']
    X_test = df[features]
    
    print("Running Inference...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    df['pred_prob'] = preds
    
    # -----------------------------------------------------------------------
    # CRITICAL FIX: The README states S1 may match MANY S2/S3 records.
    # The "global one-to-one" constraint was WRONG and was throwing away
    # true positives. We simply apply the threshold per pair.
    # F0.5 is precision-heavy so we use the model-tuned threshold.
    # -----------------------------------------------------------------------
    print(f"Applying threshold: {threshold:.4f}")
    matched = df[df['pred_prob'] > threshold][['source1_entity_id', 'candidate_entity_id', 'pred_prob']].copy()
    
    # Build S1 -> [matches] dict
    matches_dict = defaultdict(list)
    for row in matched.itertuples(index=False):
        matches_dict[row.source1_entity_id].append(row.candidate_entity_id)
    
    # Deduplicate within each S1's match list
    for k in matches_dict:
        matches_dict[k] = list(set(matches_dict[k]))
    
    # Load ALL test S1 entities — every one MUST appear in output (even singletons)
    s1_test = pd.read_csv(os.path.join(test_clean, 'test_source1.csv'), dtype=str)
    s1_ids = s1_test['entity_id'].values
    
    print(f"Total S1 entities in test: {len(s1_ids)}")
    matched_count = sum(1 for s1_id in s1_ids if s1_id in matches_dict)
    print(f"S1 entities with at least 1 match: {matched_count}")
    print(f"Singletons (no match predicted): {len(s1_ids) - matched_count}")
    
    final_output_dir = 'output/final'
    os.makedirs(final_output_dir, exist_ok=True)
    
    matching_file = os.path.join(final_output_dir, 'matching_results.tsv')
    with open(matching_file, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in s1_ids:
            if s1_id in matches_dict:
                f.write(f"{s1_id}\t{','.join(matches_dict[s1_id])}\n")
            else:
                f.write(f"{s1_id}\t\n")
                
    print(f"Saved matching_results.tsv ({len(s1_ids)} rows)")
    
    # Generate candidate_pairs.tsv from blocking output
    print("Formatting candidate_pairs.tsv...")
    cand_files = glob.glob(os.path.join(base_dir, 'test_*_candidates_*.csv'))
    
    cand_dict = defaultdict(set)
    for cf in cand_files:
        with open(cf, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.read().split('\n')
            for line in lines[1:]:
                parts = line.strip('\n').split(',')
                if len(parts) > 1:
                    s1_id = parts[0]
                    cands = [c for c in parts[1:] if c]
                    cand_dict[s1_id].update(cands)
            
    candidate_file = os.path.join(final_output_dir, 'candidate_pairs.tsv')
    with open(candidate_file, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in s1_ids:
            if s1_id in cand_dict and len(cand_dict[s1_id]) > 0:
                f.write(f"{s1_id}\t{','.join(list(cand_dict[s1_id]))}\n")
            else:
                f.write(f"{s1_id}\t\n")
                
    print("Outputs generated in output/final/")
    print("Now run: python utils/validate_submission.py --matching output/final/matching_results.tsv --candidate output/final/candidate_pairs.tsv --test-dir dataset/test")

if __name__ == '__main__':
    inference()
