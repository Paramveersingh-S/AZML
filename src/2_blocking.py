import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def run_tfidf_blocking(s1_df, target_df, s1_name, target_name, country, out_file, K=20):
    print(f"[{country}] Blocking {s1_name} vs {target_name}")
    print(f"  S1 Size: {len(s1_df)}, Target Size: {len(target_df)}")

    # Skip if already done (saves time on re-runs)
    if os.path.exists(out_file):
        print(f"  Already exists, skipping: {out_file}")
        return

    print("  Extracting text...")
    # Use the pre-computed combined_text which weights name 2x
    s1_text = s1_df['combined_text'].fillna('').values
    target_text = target_df['combined_text'].fillna('').values

    print("  Fitting TF-IDF Vectorizer...")
    # char_wb: pads words with spaces so boundaries are included in grams
    # ngram_range (2,4): catches all typo patterns (missing chars, swaps, insertions)
    # max_df=0.005: removes ultra-common character sequences that add noise
    # min_df=3: removes ultra-rare tokens that only appear in 1-2 records
    # sublinear_tf: dampens term freq to prevent very long names dominating
    vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        min_df=3,
        max_df=0.005,
        sublinear_tf=True,
    )

    t0 = time.time()
    Target_tfidf = vec.fit_transform(target_text)
    print(f"  Target TF-IDF Shape: {Target_tfidf.shape}, Time: {time.time()-t0:.2f}s")

    S1_tfidf = vec.transform(s1_text)
    print(f"  S1 TF-IDF Shape: {S1_tfidf.shape}")

    # Transpose target for efficient matrix multiply
    Target_tfidf_T = Target_tfidf.T.tocsr()

    print("  Running sparse_dot_topn C++ engine...")
    t0 = time.time()
    # Returns sparse matrix of shape (n_s1, n_target) with only top-K per row
    res_sparse = sp_matmul_topn(S1_tfidf, Target_tfidf_T, top_n=K, n_threads=4)
    print(f"  Search completed in {time.time()-t0:.2f}s!")

    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values

    print("  Writing results to file...")
    indptr = res_sparse.indptr
    indices = res_sparse.indices
    data = res_sparse.data  # TF-IDF cosine similarity scores!

    # Save: s1_id, cand1:score1, cand2:score2, ... (scores are critical features!)
    with open(out_file, 'w', encoding='utf-8') as f:
        header_cols = ['source1_entity_id'] + [f'cand_{i+1}' for i in range(K)] + [f'score_{i+1}' for i in range(K)]
        f.write(','.join(header_cols) + '\n')

        for i in range(S1_tfidf.shape[0]):
            start = indptr[i]
            end = indptr[i+1]
            cols = indices[start:end]
            scores = data[start:end]

            s1_id = s1_ids[i]
            cands = list(target_ids[cols])
            score_vals = list(scores)

            # Pad to K
            while len(cands) < K:
                cands.append('')
                score_vals.append('')

            f.write(f"{s1_id},{','.join(str(c) for c in cands)},{','.join(str(s) for s in score_vals)}\n")


def process_blocking(stage='train'):
    base_dir = '../../dataset/student_resource/dataset'
    clean_dir = os.path.join(base_dir, f'{stage}_clean')

    print(f"Loading {stage} datasets...")
    s1 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source1.csv'), dtype=str)
    s2 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source2.csv'), dtype=str)
    s3 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source3.csv'), dtype=str)

    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    # Get all countries — including France in test (never hardcode!)
    countries = s1['country'].dropna().unique()
    print(f"Found countries: {countries}")

    for c in countries:
        s1_c = s1[s1['country'] == c]
        s2_c = s2[s2['country'] == c]
        s3_c = s3[s3['country'] == c]

        out_s2 = os.path.join(out_dir, f'{stage}_{c}_candidates_s2.csv')
        out_s3 = os.path.join(out_dir, f'{stage}_{c}_candidates_s3.csv')

        run_tfidf_blocking(s1_c, s2_c, 'S1', 'S2', c, out_s2, K=20)
        run_tfidf_blocking(s1_c, s3_c, 'S1', 'S3', c, out_s3, K=20)

    print(f"Blocking completed for {stage}!")


if __name__ == '__main__':
    process_blocking('train')
    process_blocking('test')
