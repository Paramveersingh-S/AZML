@echo off
setlocal
echo Starting local pipeline execution...
set PYTHONPATH=%cd%

echo STEP 1: Data Prep
python src\1_data_prep.py
if errorlevel 1 exit /b %errorlevel%

echo STEP 2: Blocking
python src\2_blocking.py
if errorlevel 1 exit /b %errorlevel%

echo STEP 3: Feature Engineering
python src\3_feature_engineering.py
if errorlevel 1 exit /b %errorlevel%

echo STEP 4: Model Training
python src\4_train_model.py
if errorlevel 1 exit /b %errorlevel%

echo STEP 5: Inference
python src\5_inference.py
if errorlevel 1 exit /b %errorlevel%

echo STEP 6: Validation
python ..\..\dataset\student_resource\utils\validate_submission.py --matching output\final\matching_results.tsv --candidate output\final\candidate_pairs.tsv
if errorlevel 1 exit /b %errorlevel%

echo Pipeline finished successfully!
