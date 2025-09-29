# Submission v2

## What this does
- Builds test comprehensive features (train-parity) from JSONs
- Builds test temporal features (train-parity temporal subset)
- Merges them into `ultimate_dataset_test.csv`
- Predicts probabilities for class 1 and writes the final submission CSV in the requested format

## Paths
- Inputs (test): final_solution/artifacts/csv/test/{accounts_data_test.json,enquiry_data_test.json,test_flag.csv}
- Intermediates: final_solution/artifacts/csv/test/{comprehensive_test_dataset.csv, temporal_features_test.csv}
- Final test ultimate: final_solution/artifacts/csv/ultimate_dataset_test.csv
- Submission output: senior_ds_test/data/final_submission/final_submission_<First>_<Last>.csv

## How to run
python /home/miso/Documents/WINDOWS/monsoon/final_solution/submission_v2/build_comprehensive_test.py
python /home/miso/Documents/WINDOWS/monsoon/final_solution/submission_v2/build_temporal_test.py
python /home/miso/Documents/WINDOWS/monsoon/final_solution/submission_v2/create_ultimate_dataset_test.py

# Then open the notebook and run to write the final CSV
jupyter notebook /home/miso/Documents/WINDOWS/monsoon/final_solution/submission_v2/final_submission.ipynb

## What is submitted
- The second column in the CSV contains the model's predicted probability for class 1 (default), not 0/1 labels.
- The evaluator uses ROC-AUC on the hidden test labels.

## Update GitHub repo with these changes
cd /home/miso/Documents/WINDOWS/monsoon/final_solution
# If this folder is a git repo with the correct remote already set
git add submission_v2 artifacts/csv/test/*.csv artifacts/csv/ultimate_dataset_test.csv
git commit -m "Add submission_v2 pipeline and generated test datasets"
git push origin main

# If your remote is different, set it first:
# git remote set-url origin https://github.com/<your-username>/<your-repo>.git
