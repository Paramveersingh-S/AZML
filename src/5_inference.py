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
    df = pd.read_csv(os.path.join(base_dir, 'test_features.csv'))
    
    features = ['name_lev', 'name_jaro', 'name_token_sort', 'name_token_set', 'name_partial', 'addr_lev', 'addr_token_set', 'addr_missing']
    X_test = df[features]
    
    print("Running Inference...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    
    df['pred_prob'] = preds
    df['is_match'] = (preds > threshold).astype(int)
    
    # Generate matching_results.tsv
    print("Formatting matching_results.tsv...")
    matches = df[df['is_match'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(list).to_dict()
    
    # Load test S1 to ensure ALL S1 entities are in the output (even singletons)
    s1_test = pd.read_csv(os.path.join(test_clean, 'test_source1.csv'), dtype=str)
    s1_ids = s1_test['entity_id'].values
    
    final_output_dir = 'output/final'
    os.makedirs(final_output_dir, exist_ok=True)
    
    matching_file = os.path.join(final_output_dir, 'matching_results.tsv')
    with open(matching_file, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in s1_ids:
            if s1_id in matches:
                # Remove duplicates if any just in case
                unique_matches = list(set(matches[s1_id]))
                f.write(f"{s1_id}\t{','.join(unique_matches)}\n")
            else:
                f.write(f"{s1_id}\t\n")
                
    # Generate candidate_pairs.tsv
    print("Formatting candidate_pairs.tsv...")
    cand_files = glob.glob(os.path.join(base_dir, 'test_*_candidates_*.csv'))
    
    cand_dict = defaultdict(set)
    for cf in cand_files:
        with open(cf, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.read().split('\n')
            if len(lines) > 0:
                # skip header, usually first line
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
                
    print("Outputs generated successfully in d:/Projects/AZML/output/")
    print("You should now run: python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv")

if __name__ == '__main__':
    inference()
