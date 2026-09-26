import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

def scipy_to_torch_sparse(scipy_csr, device):
    """Converts a Scipy CSR matrix to a PyTorch CSR tensor on the specified device."""
    scipy_csr = scipy_csr.tocsr()
    crow = torch.from_numpy(scipy_csr.indptr).long()
    col = torch.from_numpy(scipy_csr.indices).long()
    val = torch.from_numpy(scipy_csr.data).float()
    return torch.sparse_csr_tensor(crow, col, val, size=scipy_csr.shape, device=device)

def run_pytorch_tfidf_blocking(s1_df, target_df, s1_name, target_name, country, out_file, K=15):
    print(f"[{country}] Blocking {s1_name} vs {target_name}")
    print(f"  S1 Size: {len(s1_df)}, Target Size: {len(target_df)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    print("  Extracting text...")
    s1_text = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).values
    target_text = (target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')).values
    
    print("  Fitting TF-IDF Vectorizer...")
    # Char n-grams to catch all spelling mistakes
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_df=0.05)
    
    t0 = time.time()
    Target_tfidf = vec.fit_transform(target_text)
    print(f"  Target TF-IDF Shape: {Target_tfidf.shape}, Time: {time.time()-t0:.2f}s")
    
    S1_tfidf = vec.transform(s1_text)
    print(f"  S1 TF-IDF Shape: {S1_tfidf.shape}")
    
    print("  Loading Target Matrix onto GPU...")
    # Transpose and convert to PyTorch CSR on GPU
    Target_T_csr = Target_tfidf.T.tocsr()
    Target_gpu = scipy_to_torch_sparse(Target_T_csr, device)
    
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    # Safe chunk size to avoid VRAM OOM (250 * 3M floats = ~3GB VRAM)
    batch_size = 250
    N = S1_tfidf.shape[0]
    
    print(f"  Starting PyTorch GPU processing in chunks of {batch_size}...")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id," + ",".join([f"candidate_{i+1}" for i in range(K)]) + "\n")
        
        for start_row in range(0, N, batch_size):
            t_chunk = time.time()
            end_row = min(start_row + batch_size, N)
            
            # Get chunk and move to GPU
            chunk_scipy = S1_tfidf[start_row:end_row]
            chunk_gpu = scipy_to_torch_sparse(chunk_scipy, device)
            
            # BLAZING FAST: Sparse Matrix Multiplication on GPU -> Dense Result
            # (250, Vocab) x (Vocab, 3,000,000) = (250, 3,000,000)
            res_dense = torch.sparse.mm(chunk_gpu, Target_gpu).to_dense()
            
            # Fast Top-K on GPU
            vals, inds = torch.topk(res_dense, k=K, dim=1)
            
            # Move indices back to CPU
            inds_cpu = inds.cpu().numpy()
            
            # Write to file
            for i in range(end_row - start_row):
                s1_id = s1_ids[start_row + i]
                cands = target_ids[inds_cpu[i]]
                f.write(f"{s1_id},{','.join(cands)}\n")
                
            if start_row % (batch_size * 20) == 0:
                print(f"    Processed {end_row}/{N} | Last Chunk Time: {time.time()-t_chunk:.3f}s")
                
    # Free GPU memory
    del Target_gpu
    torch.cuda.empty_cache()

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
        
        run_pytorch_tfidf_blocking(s1_c, s2_c, 'S1', 'S2', c, out_s2, K=15)
        run_pytorch_tfidf_blocking(s1_c, s3_c, 'S1', 'S3', c, out_s3, K=15)
        
    print(f"Blocking completed for {stage}!")

if __name__ == '__main__':
    process_blocking('train')
    process_blocking('test')
