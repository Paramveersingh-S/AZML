import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

def run_tfidf_blocking(s1_df, target_df, s1_name, target_name, country, out_file, K=15):
    print(f"[{country}] Blocking {s1_name} vs {target_name}")
    print(f"  S1 Size: {len(s1_df)}, Target Size: {len(target_df)}")
    
    print("  Extracting text...")
    s1_text = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).values
    target_text = (target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')).values
    
    print("  Fitting TF-IDF Vectorizer...")
    # Char n-grams catch typos. max_df prevents the matrix from getting 100% dense
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_df=0.05)
    
    t0 = time.time()
    Target_tfidf = vec.fit_transform(target_text)
    print(f"  Target TF-IDF Shape: {Target_tfidf.shape}, Time: {time.time()-t0:.2f}s")
    
    S1_tfidf = vec.transform(s1_text)
    print(f"  S1 TF-IDF Shape: {S1_tfidf.shape}")
    
    Target_tfidf_T = Target_tfidf.T.tocsr()
    
    print("  Running sparse_dot_topn C++ engine... (This will zoom through the math!)")
    t0 = time.time()
    # This automatically runs in C++, keeping memory sparse, and only returning top 15!
    # No chunking required, it manages memory internally.
    res_sparse = sp_matmul_topn(S1_tfidf, Target_tfidf_T, top_n=K, n_threads=4)
    print(f"  Search completed in {time.time()-t0:.2f}s!")
    
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    print("  Writing results to file...")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id," + ",".join([f"candidate_{i+1}" for i in range(K)]) + "\n")
        
        indptr = res_sparse.indptr
        indices = res_sparse.indices
        
        for i in range(S1_tfidf.shape[0]):
            start = indptr[i]
            end = indptr[i+1]
            cols = indices[start:end]
            
            s1_id = s1_ids[i]
            cands = target_ids[cols]
            
            # If less than 15 found, pad with empty string
            cands_list = list(cands)
            while len(cands_list) < K:
                cands_list.append('')
                
            f.write(f"{s1_id},{','.join(cands_list)}\n")
            
def process_blocking(stage='train'):
    base_dir = '../../dataset/student_resource/dataset'
    clean_dir = os.path.join(base_dir, f'{stage}_clean')
    
    print(f"Loading {stage} datasets...")
    s1 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source1.csv'), dtype=str)
    s2 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source2.csv'), dtype=str)
    s3 = pd.read_csv(os.path.join(clean_dir, f'{stage}_source3.csv'), dtype=str)
    
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    
    countries = s1['country'].dropna().unique()
    print(f"Found countries: {countries}")
    
    for c in countries:
        s1_c = s1[s1['country'] == c]
        s2_c = s2[s2['country'] == c]
        s3_c = s3[s3['country'] == c]
        
        out_s2 = os.path.join(out_dir, f'{stage}_{c}_candidates_s2.csv')
        out_s3 = os.path.join(out_dir, f'{stage}_{c}_candidates_s3.csv')
        
        run_tfidf_blocking(s1_c, s2_c, 'S1', 'S2', c, out_s2, K=15)
        run_tfidf_blocking(s1_c, s3_c, 'S1', 'S3', c, out_s3, K=15)
        
    print(f"Blocking completed for {stage}!")

if __name__ == '__main__':
    process_blocking('train')
    process_blocking('test')
