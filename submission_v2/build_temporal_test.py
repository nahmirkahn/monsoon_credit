#!/usr/bin/env python3
"""
Build temporal_features_test.csv using the same logic as train (02_temporal_feature_engineering.py),
reading from final_solution/artifacts/csv/test/accounts_data_test.json and writing to
final_solution/artifacts/csv/test/temporal_features_test.csv. The output is trimmed to the
exact temporal feature subset used in the train ultimate merge.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


TEST_DIR = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution/artifacts/csv/test')
ACCOUNTS_JSON = TEST_DIR / 'accounts_data_test.json'
OUT_CSV = TEST_DIR / 'temporal_features_test.csv'

# Temporal subset used in train ultimate
KEEP_TEMPORAL = [
    'recency_weighted_score',
    'recency_weighted_bad_ratio',
    'payment_trend_slope',
    'trend_direction',
    'recent_6m_avg_severity',
    'recent_6m_max_severity',
    'max_consecutive_bad_payments',
    'death_spiral_risk',
    'deterioration_velocity',
    'payment_volatility',
]


PAYMENT_SEVERITY = {str(i): i for i in range(10)}


def extract_features_for_history(uid: str, payment_hist: str) -> dict:
    sev = [PAYMENT_SEVERITY.get(ch, 0) for ch in payment_hist]
    out = {'uid': uid}
    # Recency weighted
    if len(sev) > 0:
        w = np.exp(-0.1 * np.arange(len(sev)))[::-1]
        out['recency_weighted_score'] = float(np.sum(np.array(sev) * w) / np.sum(w))
        out['recency_weighted_bad_ratio'] = float(np.sum((np.array(sev) > 0) * w) / np.sum(w))
    else:
        out['recency_weighted_score'] = 0.0
        out['recency_weighted_bad_ratio'] = 0.0
    # Recent 6m
    if len(sev) >= 6:
        last6 = sev[-6:]
    else:
        last6 = sev
    out['recent_6m_avg_severity'] = float(np.mean(last6)) if len(last6) else 0.0
    out['recent_6m_max_severity'] = float(np.max(last6)) if len(last6) else 0.0
    # Trend slope and direction
    if len(sev) >= 2:
        x = np.arange(len(sev), dtype=float)
        with np.errstate(all='ignore'):
            c = np.corrcoef(x, np.array(sev, dtype=float))[0, 1]
        out['payment_trend_slope'] = float(0.0 if np.isnan(c) else c)
        mid = len(sev) // 2
        out['trend_direction'] = float(np.mean(sev[mid:]) - np.mean(sev[:mid]))
    else:
        out['payment_trend_slope'] = 0.0
        out['trend_direction'] = 0.0
    # Death spiral / deterioration velocity / volatility
    max_consec = 0
    cur = 0
    for s in sev:
        if s > 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    out['max_consecutive_bad_payments'] = float(max_consec)
    last6_flag = 1.0 if any(s >= 4 for s in last6) else 0.0
    out['death_spiral_risk'] = float(1.0 if (max_consec >= 3 and last6_flag == 1.0) else 0.0)
    if len(sev) >= 2:
        max_val = int(np.max(sev))
        try:
            first_bad = next((i for i, s in enumerate(sev) if s > 0), None)
        except Exception:
            first_bad = None
        max_idx = int(np.argmax(sev))
        out['deterioration_velocity'] = float(max_val / max(1, (max_idx - (first_bad if first_bad is not None else max_idx))))
    else:
        out['deterioration_velocity'] = 0.0
    out['payment_volatility'] = float(np.std(sev)) if len(sev) > 1 else 0.0
    return out


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if not ACCOUNTS_JSON.exists():
        raise FileNotFoundError(f'Missing {ACCOUNTS_JSON}')
    data = json.load(open(ACCOUNTS_JSON, 'r'))
    rows = []
    for rec in data:
        items = rec if isinstance(rec, list) else [rec]
        for it in items:
            uid = it.get('uid')
            ph = it.get('payment_hist_string') or it.get('payment_hist')
            if uid is None:
                continue
            if isinstance(ph, str) and ph:
                rows.append(extract_features_for_history(uid, ph))
            else:
                rows.append({
                    'uid': uid,
                    'recency_weighted_score': 0.0,
                    'recency_weighted_bad_ratio': 0.0,
                    'payment_trend_slope': 0.0,
                    'trend_direction': 0.0,
                    'recent_6m_avg_severity': 0.0,
                    'recent_6m_max_severity': 0.0,
                    'max_consecutive_bad_payments': 0.0,
                    'death_spiral_risk': 0.0,
                    'deterioration_velocity': 0.0,
                    'payment_volatility': 0.0,
                })
    df = pd.DataFrame(rows)
    # Deduplicate by uid (aggregate by mean/max consistent with train if needed)
    agg = {
        'recency_weighted_score': 'mean',
        'recency_weighted_bad_ratio': 'mean',
        'payment_trend_slope': 'mean',
        'trend_direction': 'mean',
        'recent_6m_avg_severity': 'mean',
        'recent_6m_max_severity': 'max',
        'max_consecutive_bad_payments': 'max',
        'death_spiral_risk': 'max',
        'deterioration_velocity': 'mean',
        'payment_volatility': 'mean',
    }
    df = df.groupby('uid', as_index=False).agg(agg)
    # Ensure exact column order
    df = df[['uid'] + KEEP_TEMPORAL]
    df.to_csv(OUT_CSV, index=False)
    print(f'✅ Wrote {OUT_CSV} shape={df.shape}')


if __name__ == '__main__':
    main()





