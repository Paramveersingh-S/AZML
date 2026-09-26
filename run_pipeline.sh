#!/bin/bash
set -e

echo "Starting pipeline on Colab/Kaggle..."
export PYTHONPATH="${PYTHONPATH}:."

echo "Step 1: Data Prep"
python src/1_data_prep.py

echo "Step 2: Blocking"
python src/2_blocking.py

echo "Step 3: Feature Engineering"
python src/3_feature_engineering.py

echo "Step 4: Model Training"
python src/4_train_model.py

echo "Step 5: Inference"
python src/5_inference.py

echo "Step 6: Validation"
python ../../dataset/student_resource/utils/validate_submission.py --matching output/final/matching_results.tsv --candidate output/final/candidate_pairs.tsv

echo "Pipeline finished!"
