#!/usr/bin/env python3
"""
Create Ultimate Test Dataset (pure merger copy of train step 3)

Reads:
  - final_solution/artifacts/csv/test/comprehensive_test_dataset.csv
  - final_solution/artifacts/csv/test/temporal_features_test.csv
Writes:
  - final_solution/artifacts/csv/ultimate_dataset_test.csv
"""

from pathlib import Path
import pandas as pd


TEST_DIR = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution/artifacts/csv/test')
OUT_ULTI = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution/artifacts/csv/ultimate_dataset_test.csv')

# Same temporal list as train merger
TEMPORAL_FEATURES = [
    'recency_weighted_score', 'recency_weighted_bad_ratio',
    'recent_3m_avg_severity', 'recent_3m_max_severity', 'recent_3m_bad_count',
    'recent_6m_avg_severity', 'recent_6m_max_severity', 'recent_6m_bad_count',
    'recent_12m_avg_severity', 'recent_12m_max_severity', 'recent_12m_bad_count',
    'payment_trend_slope', 'trend_direction', 'deterioration_velocity',
    'max_consecutive_bad_payments', 'death_spiral_risk', 'recent_severe_flag',
    'payment_volatility', 'good_to_bad_transitions', 'bad_to_good_transitions'
]


def main():
    comp = pd.read_csv(TEST_DIR / 'comprehensive_test_dataset.csv')
    temp = pd.read_csv(TEST_DIR / 'temporal_features_test.csv')
    keep = ['uid'] + [c for c in TEMPORAL_FEATURES if c in temp.columns]
    temp = temp[keep]
    df = comp.merge(temp, on='uid', how='left')
    # Fill missing temporal with 0 as in train
    for c in keep:
        if c != 'uid' and c in df.columns:
            df[c] = df[c].fillna(0)
    df.to_csv(OUT_ULTI, index=False)
    print(f'✅ Wrote {OUT_ULTI} shape={df.shape}')


if __name__ == '__main__':
    main()





