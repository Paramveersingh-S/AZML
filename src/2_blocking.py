import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def run_tfidf_blocking(s1_df, target_df, s1_name, target_name, country, out_file, K=15):
    print(f"[{country}] Blocking {s1_name} vs {target_name}")
    print(f"  S1 Size: {len(s1_df)}, Target Size: {len(target_df)}")
    
    print("  Extracting combined text features...")
    s1_text = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).values
    target_text = (target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')).values
    
    print("  Fitting TF-IDF Vectorizer (this is character-aware and handles typos!)...")
    # max_df=0.05 guarantees we drop hyper-common n-grams which cause 100% density and OOMs.
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_df=0.05)
    
    t0 = time.time()
    Target_tfidf = vec.fit_transform(target_text)
    print(f"  Target TF-IDF Shape: {Target_tfidf.shape}, Time: {time.time()-t0:.2f}s")
    
    t0 = time.time()
    S1_tfidf = vec.transform(s1_text)
    print(f"  S1 TF-IDF Shape: {S1_tfidf.shape}, Time: {time.time()-t0:.2f}s")
    
    print("  Building Sparse NearestNeighbors Index (Multi-threaded C++)...")
    t0 = time.time()
    # brute algorithm natively handles sparse matrices extremely well with cosine metric
    nn = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(Target_tfidf)
    print(f"  Index Build Time: {time.time()-t0:.2f}s")
    
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    # Process in larger chunks to exploit parallel CPU cores but avoid RAM spikes
    batch_size = 50000
    N = S1_tfidf.shape[0]
    
    print("  Searching Nearest Neighbors...")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id," + ",".join([f"candidate_{i+1}" for i in range(K)]) + "\n")
        
        for start_row in range(0, N, batch_size):
            t_chunk = time.time()
            end_row = min(start_row + batch_size, N)
            chunk = S1_tfidf[start_row:end_row]
            
            # This is automatically multi-threaded
            distances, indices = nn.kneighbors(chunk)
            
            # Write to file
            for i in range(end_row - start_row):
                s1_id = s1_ids[start_row + i]
                cands = target_ids[indices[i]]
                f.write(f"{s1_id},{','.join(cands)}\n")
            
            print(f"    Processed {end_row}/{N} | Chunk Time: {time.time()-t_chunk:.2f}s")

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
