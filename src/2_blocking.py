import pandas as pd
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def run_embedding_blocking(s1_df, target_df, s1_prefix, target_prefix, country, out_file, K=15):
    """
    Computes matches using Semantic Embeddings and FAISS GPU for blazing fast, high-recall retrieval.
    """
    print(f"[{country}] Blocking {s1_prefix} vs {target_prefix}")
    print(f"  S1 Size: {len(s1_df)}, Target Size: {len(target_df)}")
    
    if len(s1_df) == 0 or len(target_df) == 0:
        return
        
    print("  Extracting combined text features...")
    # Fill NAs and convert to strings
    s1_text = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).tolist()
    target_text = (target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')).tolist()
    
    print("  Loading Sentence Transformer (GPU)...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    print("  Encoding Target texts...")
    target_embeddings = model.encode(target_text, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
    
    print("  Encoding S1 texts...")
    s1_embeddings = model.encode(s1_text, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
    
    print("  Building FAISS GPU Index...")
    d = target_embeddings.shape[1]
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(d) # Inner product = Cosine similarity for normalized vectors
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    gpu_index.add(target_embeddings)
    
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    print(f"  Searching top {K} matches on GPU...")
    batch_size = 2000
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id,candidate_entity_ids\n')
        
        for i in tqdm(range(0, len(s1_embeddings), batch_size)):
            chunk = s1_embeddings[i:i+batch_size]
            distances, indices = gpu_index.search(chunk, K)
            
            for j in range(len(chunk)):
                s1_id = s1_ids[i+j]
                cands = target_ids[indices[j]]
                f.write(f"{s1_id},{','.join(cands)}\n")


def process_blocking(stage='train'):
    base_dir = '../dataset/student_resource/dataset'
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
        run_embedding_blocking(s1_c, s2_c, 'S1', 'S2', c, out_s2, K=15)
        
        # Block S1 vs S3
        out_s3 = os.path.join(out_dir, f'{stage}_s3_candidates_{c_safe}.csv')
        run_embedding_blocking(s1_c, s3_c, 'S1', 'S3', c, out_s3, K=15)

    print(f"Blocking completed for {stage}!")

if __name__ == '__main__':
    # Process train first to train the model
    process_blocking('train')
    # Later we will process test
    process_blocking('test')
