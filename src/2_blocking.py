import pandas as pd
import numpy as np
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from tqdm import tqdm
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

def run_sparse_blocking(s1_df, target_df, s1_prefix, target_prefix, country, out_file, K=15):
    """
    Computes matches using TF-IDF and Sparse Matrix Batch Multiplication to avoid OOM while achieving high recall.
    """
    print(f"[{country}] Blocking {s1_prefix} vs {target_prefix}")
    print(f"  S1 Size: {len(s1_df)}, Target Size: {len(target_df)}")
    
    if len(s1_df) == 0 or len(target_df) == 0:
        return
        
    print("  Extracting combined text features...")
    # Fill NAs and convert to strings
    s1_text = s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')
    target_text = target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')
    
    print("  Fitting TF-IDF Vectorizer (char_wb ngrams)...")
    # Character n-grams are robust to typos and missing spaces
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_df=0.5)
    
    # Fit on target text to build the vocabulary
    vectorizer.fit(pd.concat([s1_text, target_text]))
    
    print("  Transforming to Sparse Matrices...")
    S1_tfidf = vectorizer.transform(s1_text)
    Target_tfidf = vectorizer.transform(target_text)
    
    Target_tfidf_T = Target_tfidf.T.tocsr()
    
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    print("  Performing Batch Sparse Matrix Multiplication...")
    batch_size = 2000
    N = S1_tfidf.shape[0]
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id,candidate_entity_ids\n')
        
        for start_row in tqdm(range(0, N, batch_size)):
            end_row = min(start_row + batch_size, N)
            
            chunk = S1_tfidf[start_row:end_row]
            # Matrix multiplication
            C_chunk = chunk.dot(Target_tfidf_T)
            
            for i in range(C_chunk.shape[0]):
                s1_idx = start_row + i
                s1_id = s1_ids[s1_idx]
                
                row = C_chunk.getrow(i)
                data = row.data
                indices = row.indices
                
                if len(data) == 0:
                    f.write(f"{s1_id},\n")
                    continue
                
                if len(data) > K:
                    # Get top K indices
                    top_k_idx = np.argpartition(data, -K)[-K:]
                    # Sort them by score
                    sort_idx = top_k_idx[np.argsort(data[top_k_idx])[::-1]]
                    best_indices = indices[sort_idx]
                else:
                    sort_idx = np.argsort(data)[::-1]
                    best_indices = indices[sort_idx]
                    
                cands = target_ids[best_indices]
                f.write(f"{s1_id},{','.join(cands)}\n")


def process_blocking(stage='train'):
    base_dir = '../../dataset/student_resource/dataset'
    clean_dir = os.path.join(base_dir, f'{stage}_clean')
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {stage} datasets...")
    s1 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source1.csv'), dtype=str)
    s2 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source2.csv'), dtype=str)
    s3 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source3.csv'), dtype=str)
    
    # We process by country to reduce search space
    countries = s1['country'].unique()
    print(f"Found countries: {countries}")
    
    for c in countries:
        c_safe = c.replace(' ', '_')
        s1_c = s1[s1['country'] == c]
        s2_c = s2[s2['country'] == c]
        s3_c = s3[s3['country'] == c]
        
        # Block S1 vs S2
        out_s2 = os.path.join(out_dir, f'{stage}_s2_candidates_{c_safe}.csv')
        run_sparse_blocking(s1_c, s2_c, 'S1', 'S2', c, out_s2, K=15)
        
        # Block S1 vs S3
        out_s3 = os.path.join(out_dir, f'{stage}_s3_candidates_{c_safe}.csv')
        run_sparse_blocking(s1_c, s3_c, 'S1', 'S3', c, out_s3, K=15)

    print(f"Blocking completed for {stage}!")

if __name__ == '__main__':
    # Process train first to train the model
    process_blocking('train')
    # Later we will process test
    process_blocking('test')
