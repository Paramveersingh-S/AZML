import pandas as pd
import numpy as np
import os
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import multiprocessing

# We use rapidfuzz because it's in C++ and insanely fast for string matching

def build_features(stage='train'):
    base_dir = '../../dataset/student_resource/dataset'
    clean_dir = os.path.join(base_dir, f'{stage}_clean')
    out_dir = 'output'
    
    print(f"Loading {stage} Data...")
    s1 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source1.csv'), dtype=str)
    s1 = s1[['entity_id', 'business_name_clean', 'business_address_clean']]
    s1.columns = ['source1_entity_id', 's1_name', 's1_addr']
    
    s2 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source2.csv'), dtype=str)
    s2 = s2[['entity_id', 'business_name_clean', 'business_address_clean']]
    s2.columns = ['candidate_entity_id', 'target_name', 'target_addr']
    
    s3 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source3.csv'), dtype=str)
    s3 = s3[['entity_id', 'business_name_clean', 'business_address_clean']]
    s3.columns = ['candidate_entity_id', 'target_name', 'target_addr']
    
    target_dict = pd.concat([s2, s3])
    
    # Load Candidates
    print("Loading Candidates...")
    import glob
    cand_files = glob.glob(os.path.join(out_dir, f'{stage}_*_candidates_*.csv'))
    
    pairs = []
    for f in cand_files:
        with open(f, 'r', encoding='utf-8') as file:
            next(file) # skip header
            for line in file:
                parts = line.strip('\n').split(',')
                if len(parts) > 1:
                    s1_id = parts[0]
                    # Take all K candidates for maximum recall
                    cands = [c for c in parts[1:] if c]
                    for c in cands:
                        pairs.append((s1_id, c))
                        
    pairs = pd.DataFrame(pairs, columns=['source1_entity_id', 'candidate_entity_id']).drop_duplicates()
    
    print(f"Total Candidate Pairs: {len(pairs)}")
    
    # Merge text
    print("Merging text for feature extraction...")
    pairs = pairs.merge(s1, on='source1_entity_id', how='left')
    pairs = pairs.merge(target_dict, on='candidate_entity_id', how='left')
    
    # Vectorized computation using list comprehensions
    s1_names = pairs['s1_name'].fillna('').astype(str).tolist()
    s1_addrs = pairs['s1_addr'].fillna('').astype(str).tolist()
    t_names = pairs['target_name'].fillna('').astype(str).tolist()
    t_addrs = pairs['target_addr'].fillna('').astype(str).tolist()
    
    print("  Calculating name_lev...")
    pairs['name_lev'] = [fuzz.ratio(a, b) for a, b in zip(s1_names, t_names)]
    print("  Calculating name_jaro...")
    pairs['name_jaro'] = [JaroWinkler.normalized_similarity(a, b) * 100 for a, b in zip(s1_names, t_names)]
    print("  Calculating name_token_sort...")
    pairs['name_token_sort'] = [fuzz.token_sort_ratio(a, b) for a, b in zip(s1_names, t_names)]
    print("  Calculating name_token_set...")
    pairs['name_token_set'] = [fuzz.token_set_ratio(a, b) for a, b in zip(s1_names, t_names)]
    print("  Calculating name_partial...")
    pairs['name_partial'] = [fuzz.partial_ratio(a, b) for a, b in zip(s1_names, t_names)]
    print("  Calculating addr_lev...")
    pairs['addr_lev'] = [fuzz.ratio(a, b) if b else 0.0 for a, b in zip(s1_addrs, t_addrs)]
    print("  Calculating addr_token_set...")
    pairs['addr_token_set'] = [fuzz.token_set_ratio(a, b) if b else 0.0 for a, b in zip(s1_addrs, t_addrs)]
    pairs['addr_missing'] = [1.0 if not b else 0.0 for b in t_addrs]
    
    final_df = pairs
    
    # Labels
    if stage == 'train':
        print("Attaching Ground Truth labels...")
        gt = pd.read_csv(os.path.join(clean_dir, 'train_ground_truth.csv'), dtype=str)
        gt = gt.dropna(subset=['matched_entity_ids'])
        gt = gt.assign(candidate_entity_id=gt['matched_entity_ids'].str.split(',')).explode('candidate_entity_id')
        gt['label'] = 1
        gt = gt[['source1_entity_id', 'candidate_entity_id', 'label']]
        
        final_df = final_df.merge(gt, on=['source1_entity_id', 'candidate_entity_id'], how='left')
        final_df['label'] = final_df['label'].fillna(0).astype(int)
        
    print(f"Saving features for {stage}...")
    final_df.to_csv(os.path.join(out_dir, f'{stage}_features.csv'), index=False)
    print("Done.")

if __name__ == '__main__':
    build_features('train')
    build_features('test')
