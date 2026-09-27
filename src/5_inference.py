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

    # Load model, threshold, and the exact feature list used during training
    model_path = os.path.join(base_dir, 'lgb_model.pkl')
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        threshold = data['threshold']
        # Load features list saved during training for 100% compatibility
        features = data.get('features', [
            'name_lev', 'name_jaro', 'name_token_sort', 'name_token_set', 'name_partial',
            'addr_lev', 'addr_token_set', 'addr_missing',
        ])

    print(f"Loaded Model. Best threshold from training: {threshold:.4f}")
    print(f"Using features: {features}")

    print("Loading test features...")
    df = pd.read_parquet(os.path.join(base_dir, 'test_features.parquet'))

    # Only use features that exist in both model and test data
    features = [f for f in features if f in df.columns]
    X_test = df[features].astype(np.float32)

    print(f"Test pairs to score: {len(df):,}")
    print("Running Inference...")
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    df['pred_prob'] = preds

    # -----------------------------------------------------------------------
    # INFERENCE LOGIC
    # The README explicitly states: "A Source 1 entity may match zero, one,
    # or MANY records from Source 2 and Source 3."
    # We apply the threshold per pair — no global 1:1 constraint.
    # F0.5 is precision-heavy so threshold is already tuned conservatively.
    # -----------------------------------------------------------------------
    print(f"Applying threshold: {threshold:.4f}")
    matched = df[df['pred_prob'] > threshold][['source1_entity_id', 'candidate_entity_id']].copy()
    print(f"Matched pairs above threshold: {len(matched):,}")

    # Build S1 -> [matches] dict with deduplication
    matches_dict = defaultdict(set)
    for row in matched.itertuples(index=False):
        matches_dict[row.source1_entity_id].add(row.candidate_entity_id)

    # Load ALL test S1 entities (every one MUST appear in output, even singletons)
    s1_test = pd.read_csv(os.path.join(test_clean, 'test_source1.csv'), dtype=str)
    s1_ids = s1_test['entity_id'].values

    matched_count = sum(1 for s1_id in s1_ids if s1_id in matches_dict)
    singleton_count = len(s1_ids) - matched_count
    total_match_pairs = sum(len(v) for v in matches_dict.values())

    print(f"\nResults Summary:")
    print(f"  Total S1 entities:        {len(s1_ids):,}")
    print(f"  Entities with ≥1 match:   {matched_count:,} ({matched_count/len(s1_ids):.1%})")
    print(f"  Singletons (no match):    {singleton_count:,} ({singleton_count/len(s1_ids):.1%})")
    print(f"  Total match pairs output: {total_match_pairs:,}")
    print(f"  Avg matches per S1:       {total_match_pairs/max(matched_count,1):.2f}")

    final_output_dir = 'output/final'
    os.makedirs(final_output_dir, exist_ok=True)

    # Write matching_results.tsv (the scored file)
    matching_file = os.path.join(final_output_dir, 'matching_results.tsv')
    with open(matching_file, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in s1_ids:
            if s1_id in matches_dict:
                f.write(f"{s1_id}\t{','.join(sorted(matches_dict[s1_id]))}\n")
            else:
                f.write(f"{s1_id}\t\n")

    print(f"\nSaved matching_results.tsv ({len(s1_ids)} rows)")

    # Write candidate_pairs.tsv (not scored, used for blocking analysis)
    print("Formatting candidate_pairs.tsv...")
    cand_files = glob.glob(os.path.join(base_dir, 'test_*_candidates_*.csv'))

    cand_dict = defaultdict(set)
    for cf in cand_files:
        with open(cf, 'r', encoding='utf-8', errors='replace') as file:
            header = next(file).strip().split(',')
            K = sum(1 for h in header if h.startswith('cand_'))
            if K == 0:
                K = len(header) - 1  # old format

            for line in file:
                parts = line.strip('\n').split(',')
                if len(parts) < 2:
                    continue
                s1_id = parts[0]
                cands = [c for c in parts[1:K+1] if c]
                cand_dict[s1_id].update(cands)

    candidate_file = os.path.join(final_output_dir, 'candidate_pairs.tsv')
    with open(candidate_file, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in s1_ids:
            if s1_id in cand_dict and len(cand_dict[s1_id]) > 0:
                f.write(f"{s1_id}\t{','.join(sorted(cand_dict[s1_id]))}\n")
            else:
                f.write(f"{s1_id}\t\n")

    print(f"Saved candidate_pairs.tsv")
    print(f"\nOutputs ready in {final_output_dir}/")
    print("Validate with:")
    print("  python utils/validate_submission.py \\")
    print("    --matching output/final/matching_results.tsv \\")
    print("    --candidate output/final/candidate_pairs.tsv \\")
    print("    --test-dir dataset/test")


if __name__ == '__main__':
    inference()
