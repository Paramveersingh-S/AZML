import pandas as pd
import numpy as np
import re
import os
import glob
from tqdm import tqdm

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation except alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Expand common abbreviations (International)
    abbrev = {
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\bco\b': 'company',
        r'\bltd\b': 'limited',
        r'\bpvt\b': 'private',
        r'\bllc\b': 'limited liability company',
        r'\bgmbh\b': 'gesellschaft mit beschränkter haftung',
        r'\bsarl\b': 'société à responsabilité limitée',
        r'\bsas\b': 'société par actions simplifiée',
        r'\&': 'and',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment',
    }
    for k, v in abbrev.items():
        text = re.sub(k, v, text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(filepath, out_dir):
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath, sep='\t', dtype=str)
    
    # Fill NA addresses with empty string
    if 'business_address' in df.columns:
        df['business_address'] = df['business_address'].fillna('')
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('MISSING')
        
    # Apply text cleaning
    print("  Cleaning business_name...")
    tqdm.pandas()
    df['business_name_clean'] = df['business_name'].progress_apply(clean_text)
    
    print("  Cleaning business_address...")
    if 'business_address' in df.columns:
        df['business_address_clean'] = df['business_address'].progress_apply(clean_text)
    else:
        df['business_address_clean'] = ""
        
    # Combine name and address for semantic blocking
    df['combined_text'] = df['business_name_clean'] + ' ' + df['business_address_clean']
    
    # Save to csv for downstream processing
    basename = os.path.basename(filepath).replace('.tsv', '.csv')
    out_path = os.path.join(out_dir, basename)
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

def main():
    train_dir = '../../dataset/student_resource/dataset/train'
    test_dir = '../../dataset/student_resource/dataset/test'
    
    out_train_dir = '../../dataset/student_resource/dataset/train_clean'
    out_test_dir = '../../dataset/student_resource/dataset/test_clean'
    
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    
    # Process Train files
    for file in glob.glob(os.path.join(train_dir, '*.tsv')):
        if 'ground_truth' not in file:
            process_file(file, out_train_dir)
            
    # Process Test files
    for file in glob.glob(os.path.join(test_dir, '*.tsv')):
        process_file(file, out_test_dir)
        
    # Just copy ground truth as is (CSV format)
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, sep='\t', dtype=str)
        gt.to_csv(os.path.join(out_train_dir, 'train_ground_truth.csv'), index=False)
        print("Processed ground truth.")

if __name__ == '__main__':
    main()
