import pandas as pd
import numpy as np
import re
import os
import glob
import unicodedata
from tqdm import tqdm

# Compile all patterns once at module level (not inside function - 10x faster)
# Order matters: longer/more specific patterns first
ABBREV = [
    # Legal suffixes - name normalisation
    (re.compile(r'\bprivate\s+limited\b'), 'pvt ltd'),
    (re.compile(r'\bpvt\.?\s*ltd\.?\b'), 'pvt ltd'),
    (re.compile(r'\blimited\s+liability\s+company\b'), 'llc'),
    (re.compile(r'\bllp\b'), 'limited liability partnership'),
    (re.compile(r'\bcorporation\b'), 'corp'),
    (re.compile(r'\bincorporated\b'), 'inc'),
    (re.compile(r'\blimited\b'), 'ltd'),
    (re.compile(r'\bcompany\b'), 'co'),
    # French legal suffixes (test set has France!)
    (re.compile(r'\bsociété\s+à\s+responsabilité\s+limitée\b'), 'sarl'),
    (re.compile(r'\bsociété\s+par\s+actions\s+simplifiée\b'), 'sas'),
    (re.compile(r'\bsociété\s+anonyme\b'), 'sa'),
    (re.compile(r'\bsarl\b'), 'sarl'),
    (re.compile(r'\bsas\b'), 'sas'),
    (re.compile(r'\bsa\b'), 'sa'),
    # German suffixes
    (re.compile(r'\bgmbh\b'), 'gmbh'),
    # Indian suffixes
    (re.compile(r'\bpvt\b'), 'pvt'),
    (re.compile(r'\benterprises\b'), 'enterprises'),
    (re.compile(r'\btraders\b'), 'traders'),
    # Address normalisation
    (re.compile(r'\bstreet\b'), 'st'),
    (re.compile(r'\broad\b'), 'rd'),
    (re.compile(r'\bavenue\b'), 'ave'),
    (re.compile(r'\bboulevard\b'), 'blvd'),
    (re.compile(r'\bdrive\b'), 'dr'),
    (re.compile(r'\bsuite\b'), 'ste'),
    (re.compile(r'\bapartment\b'), 'apt'),
    (re.compile(r'\bfloor\b'), 'fl'),
    (re.compile(r'\bnorth\b'), 'n'),
    (re.compile(r'\bsouth\b'), 's'),
    (re.compile(r'\beast\b'), 'e'),
    (re.compile(r'\bwest\b'), 'w'),
    # Common symbols
    (re.compile(r'&'), 'and'),
    (re.compile(r'@'), 'at'),
]

PUNCT_RE = re.compile(r'[^\w\s]')
SPACE_RE = re.compile(r'\s+')


def normalize_unicode(text):
    """
    Normalize unicode: decompose accented chars to their base form.
    Converts é→e, ñ→n, ü→u etc. Critical for French entity names.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def clean_text(text):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""
    # Unicode normalization (removes accents: café -> cafe)
    text = normalize_unicode(text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (replace with space so words don't merge)
    text = PUNCT_RE.sub(' ', text)
    # Apply abbreviation normalization (compiled regex, fast)
    for pattern, replacement in ABBREV:
        text = pattern.sub(replacement, text)
    # Collapse whitespace
    text = SPACE_RE.sub(' ', text).strip()
    return text


def process_file(filepath, out_dir):
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath, sep='\t', dtype=str)

    if 'business_address' in df.columns:
        df['business_address'] = df['business_address'].fillna('')
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('MISSING')

    print("  Cleaning business_name...")
    df['business_name_clean'] = df['business_name'].apply(clean_text)

    print("  Cleaning business_address...")
    if 'business_address' in df.columns:
        df['business_address_clean'] = df['business_address'].apply(clean_text)
    else:
        df['business_address_clean'] = ""

    # Combined text used in TF-IDF blocking (name is more important, repeat it)
    df['combined_text'] = df['business_name_clean'] + ' ' + df['business_name_clean'] + ' ' + df['business_address_clean']

    basename = os.path.basename(filepath).replace('.tsv', '.csv')
    out_path = os.path.join(out_dir, basename)
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path} ({len(df)} rows)")


def main():
    train_dir = '../../dataset/student_resource/dataset/train'
    test_dir = '../../dataset/student_resource/dataset/test'

    out_train_dir = '../../dataset/student_resource/dataset/train_clean'
    out_test_dir = '../../dataset/student_resource/dataset/test_clean'

    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    for file in glob.glob(os.path.join(train_dir, '*.tsv')):
        if 'ground_truth' not in file:
            process_file(file, out_train_dir)

    for file in glob.glob(os.path.join(test_dir, '*.tsv')):
        process_file(file, out_test_dir)

    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, sep='\t', dtype=str)
        gt.to_csv(os.path.join(out_train_dir, 'train_ground_truth.csv'), index=False)
        print("Processed ground truth.")


if __name__ == '__main__':
    main()
