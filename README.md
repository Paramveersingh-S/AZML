# AZML 2026: Business Entity Resolution

## Reproduction Instructions
To reproduce the results (`matching_results.tsv` and `candidate_pairs.tsv`) end-to-end, follow these steps:

1. **Environment Setup:**
   Make sure you have Python 3.8+ installed. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Execute the Pipeline End-to-End:**
   We have provided a convenient batch script to run all 5 steps of the pipeline and validate the output automatically.
   
   Simply run from the root of the project (if using PowerShell, use `.\`):
   ```bash
   .\run_pipeline.bat
   ```

### What `run_pipeline.bat` does:
- **Step 1 (`1_data_prep.py`):** Normalizes business names (lowercase, removes punctuation, expands abbreviations) and saves the raw TSVs as standard CSV files (we bypassed Parquet to speed up downloads).
- **Step 2 (`2_blocking.py`):** Partitions data by country. Uses TF-IDF character 3-grams and sparse matrix exact cosine similarity to retrieve the Top-15 S2 and S3 candidates per S1 entity. Reduces search space drastically.
- **Step 3 (`3_feature_engineering.py`):** Computes pairwise string similarity features (Jaccard, Levenshtein distance, Token Sort Ratio) using `rapidfuzz` for blazing-fast C++ optimized matching.
- **Step 4 (`4_train_model.py`):** Trains a LightGBM Binary Classifier on the extracted features. Uses the validation set to precisely optimize the decision threshold for the $F_{0.5}$ metric (favouring precision).
- **Step 5 (`5_inference.py`):** Applies the pipeline to the test set, filters candidates by the optimal threshold, and formats the output exactly as required, handling singletons perfectly.
- **Step 6 (Validation):** Automatically runs `utils/validate_submission.py` to ensure zero format violations.

### Final Outputs
The final generated files will be located in the `<root>/output/` directory:
- `matching_results.tsv`
- `candidate_pairs.tsv`
