import pandas as pd
import numpy as np
import os
import glob
import gc
import re
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


def extract_numbers(text):
    """Extract all numeric tokens from text (for address number matching)."""
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r'\d+', text))


def first_token(text):
    """Get first word of text (usually most important part of business name)."""
    if not text or not isinstance(text, str):
        return ''
    parts = text.strip().split()
    return parts[0] if parts else ''


def build_features(stage='train'):
    base_dir = '../../dataset/student_resource/dataset'
    clean_dir = os.path.join(base_dir, f'{stage}_clean')
    out_dir = 'output'

    print(f"Loading {stage} Data...")
    s1 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source1.csv'), dtype=str)
    s1 = s1[['entity_id', 'business_name_clean', 'business_address_clean', 'country']]
    s1.columns = ['source1_entity_id', 's1_name', 's1_addr', 's1_country']

    s2 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source2.csv'), dtype=str)
    s2 = s2[['entity_id', 'business_name_clean', 'business_address_clean']]
    s2.columns = ['candidate_entity_id', 'target_name', 'target_addr']

    s3 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source3.csv'), dtype=str)
    s3 = s3[['entity_id', 'business_name_clean', 'business_address_clean']]
    s3.columns = ['candidate_entity_id', 'target_name', 'target_addr']

    target_dict = pd.concat([s2, s3], ignore_index=True)

    # Load candidate pairs — now includes TF-IDF cosine scores!
    print("Loading Candidates...")
    cand_files = glob.glob(os.path.join(out_dir, f'{stage}_*_candidates_*.csv'))

    pairs = []
    for f in cand_files:
        with open(f, 'r', encoding='utf-8') as file:
            header = next(file).strip().split(',')
            # Detect new format (has score cols) vs old format
            has_scores = any('score_' in h for h in header)
            K = sum(1 for h in header if h.startswith('cand_'))

            for line in file:
                parts = line.strip('\n').split(',')
                if len(parts) < 2:
                    continue
                s1_id = parts[0]
                if has_scores:
                    cand_vals = parts[1:K+1]
                    score_vals = parts[K+1:K+1+K]
                    for cand, score in zip(cand_vals, score_vals):
                        if cand:
                            try:
                                pairs.append((s1_id, cand, float(score) if score else 0.0))
                            except ValueError:
                                pairs.append((s1_id, cand, 0.0))
                else:
                    # Old format without scores
                    cands = [c for c in parts[1:] if c]
                    for c in cands:
                        pairs.append((s1_id, c, 0.0))

    if not pairs:
        raise RuntimeError("No candidate files found! Run step 2 first.")

    pairs = pd.DataFrame(pairs, columns=['source1_entity_id', 'candidate_entity_id', 'tfidf_score'])
    # Keep highest tfidf_score if same pair appears in multiple blocking files
    pairs = pairs.sort_values('tfidf_score', ascending=False).drop_duplicates(
        subset=['source1_entity_id', 'candidate_entity_id'], keep='first'
    )

    print(f"Total Candidate Pairs: {len(pairs)}")

    # Merge text
    print("Merging text for feature extraction...")
    pairs = pairs.merge(s1, on='source1_entity_id', how='left')
    pairs = pairs.merge(target_dict, on='candidate_entity_id', how='left')

    # FREE RAM immediately after merge
    del s1, target_dict, s2, s3
    gc.collect()

    # -----------------------------------------------------------------------
    # FEATURE ENGINEERING
    # Rule: more features = better model = higher score
    # -----------------------------------------------------------------------
    s1_names = pairs['s1_name'].fillna('').astype(str).tolist()
    s1_addrs = pairs['s1_addr'].fillna('').astype(str).tolist()
    t_names = pairs['target_name'].fillna('').astype(str).tolist()
    t_addrs = pairs['target_addr'].fillna('').astype(str).tolist()

    print("  Calculating name features...")
    # Core string similarity (proven features for entity resolution)
    pairs['name_lev'] = [fuzz.ratio(a, b) for a, b in zip(s1_names, t_names)]
    pairs['name_jaro'] = [JaroWinkler.normalized_similarity(a, b) * 100 for a, b in zip(s1_names, t_names)]
    pairs['name_token_sort'] = [fuzz.token_sort_ratio(a, b) for a, b in zip(s1_names, t_names)]
    pairs['name_token_set'] = [fuzz.token_set_ratio(a, b) for a, b in zip(s1_names, t_names)]
    pairs['name_partial'] = [fuzz.partial_ratio(a, b) for a, b in zip(s1_names, t_names)]

    print("  Calculating address features...")
    pairs['addr_lev'] = [fuzz.ratio(a, b) if b else 0.0 for a, b in zip(s1_addrs, t_addrs)]
    pairs['addr_token_set'] = [fuzz.token_set_ratio(a, b) if b else 0.0 for a, b in zip(s1_addrs, t_addrs)]
    pairs['addr_missing'] = [1.0 if not b else 0.0 for b in t_addrs]

    print("  Calculating advanced features...")

    # TF-IDF cosine score from blocking (single most predictive feature!)
    # Already in pairs['tfidf_score']

    # Name length ratio (identical entities have similar name lengths)
    s1_len = np.array([len(n) for n in s1_names], dtype=np.float32)
    t_len = np.array([len(n) for n in t_names], dtype=np.float32)
    # Avoid division by zero
    max_len = np.maximum(s1_len, t_len)
    min_len = np.minimum(s1_len, t_len)
    pairs['name_len_ratio'] = np.where(max_len > 0, min_len / max_len, 0.0)

    # First-token exact match: "McDonald's" vs "McDonald Restaurant" → first token matches!
    # One of the strongest single features for business name matching
    pairs['name_first_token_match'] = [
        1.0 if first_token(a) == first_token(b) and first_token(a) != '' else 0.0
        for a, b in zip(s1_names, t_names)
    ]

    # First-token fuzzy match (handles "McDnlds" vs "McDonald")
    pairs['name_first_token_lev'] = [
        fuzz.ratio(first_token(a), first_token(b))
        for a, b in zip(s1_names, t_names)
    ]

    # Address numeric overlap: "123 Main St" vs "123 Oak Ave" → both have "123"
    # Critical for address matching since numbers are the most unique address component
    pairs['addr_num_overlap'] = [
        len(extract_numbers(a) & extract_numbers(b)) / max(len(extract_numbers(a) | extract_numbers(b)), 1)
        for a, b in zip(s1_addrs, t_addrs)
    ]

    # Is candidate from S2 or S3? Different sources may have different noise patterns.
    pairs['is_s2'] = pairs['candidate_entity_id'].str.startswith('S2-').astype(np.int8)

    # Name exact match (highest-precision signal — rare but perfect)
    pairs['name_exact'] = (pairs['s1_name'] == pairs['target_name']).astype(np.int8)

    # FREE RAM before saving
    del s1_names, s1_addrs, t_names, t_addrs
    gc.collect()

    feature_cols = [
        'source1_entity_id', 'candidate_entity_id',
        # Original 8
        'name_lev', 'name_jaro', 'name_token_sort', 'name_token_set', 'name_partial',
        'addr_lev', 'addr_token_set', 'addr_missing',
        # New 7 high-value features
        'tfidf_score', 'name_len_ratio', 'name_first_token_match',
        'name_first_token_lev', 'addr_num_overlap', 'is_s2', 'name_exact',
    ]

    if stage == 'train':
        feature_cols.append('label')

    if stage == 'train':
        print("Attaching Ground Truth labels...")
        gt = pd.read_csv(os.path.join(clean_dir, 'train_ground_truth.csv'), dtype=str)
        gt = gt.dropna(subset=['matched_entity_ids'])
        gt = gt.assign(candidate_entity_id=gt['matched_entity_ids'].str.split(',')).explode('candidate_entity_id')
        gt['candidate_entity_id'] = gt['candidate_entity_id'].str.strip()
        gt['label'] = 1
        gt = gt[['source1_entity_id', 'candidate_entity_id', 'label']]

        pairs = pairs.merge(gt, on=['source1_entity_id', 'candidate_entity_id'], how='left')
        pairs['label'] = pairs['label'].fillna(0).astype(np.int8)

    final_df = pairs[feature_cols]

    print(f"Saving features for {stage} (Parquet)...")
    final_df.to_parquet(os.path.join(out_dir, f'{stage}_features.parquet'), index=False)
    print(f"Done. Shape: {final_df.shape}")

    if stage == 'train':
        pos = final_df['label'].sum()
        total = len(final_df)
        print(f"Positive rate: {pos}/{total} = {pos/total:.4%}")


if __name__ == '__main__':
    build_features('train')
    build_features('test')
