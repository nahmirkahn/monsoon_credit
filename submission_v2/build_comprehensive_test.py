#!/usr/bin/env python3
"""
Build comprehensive_test_dataset.csv with the exact train-time feature set
from test JSONs located in final_solution/artifacts/csv/test.

Inputs:
  - final_solution/artifacts/csv/test/test_flag.csv
  - final_solution/artifacts/csv/test/accounts_data_test.json
  - final_solution/artifacts/csv/test/enquiry_data_test.json

Output:
  - final_solution/artifacts/csv/test/comprehensive_test_dataset.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


TEST_DIR = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution/artifacts/csv/test')
TEST_FLAG = TEST_DIR / 'test_flag.csv'
ACCOUNTS_JSON = TEST_DIR / 'accounts_data_test.json'
ENQUIRY_JSON = TEST_DIR / 'enquiry_data_test.json'
OUT_CSV = TEST_DIR / 'comprehensive_test_dataset.csv'


def _safe_number(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def per_account_metrics(ph: str) -> dict:
    if not isinstance(ph, str) or len(ph) == 0:
        return {
            'length': 0, 'zero_cnt': 0, 'bad_cnt': 0, 'bad_ratio': 0.0,
            'severe_cnt': 0, 'severe_ratio': 0.0,
            'recent_bad_cnt': 0, 'recent_bad_ratio': 0.0,
            'recent_severe_cnt': 0,
        }
    length = len(ph)
    zero_cnt = ph.count('0')
    bad_cnt = sum(1 for c in ph if c != '0')
    severe_cnt = sum(1 for c in ph if c.isdigit() and int(c) >= 4)
    last6 = ph[-6:] if length >= 6 else ph
    recent_bad_cnt = sum(1 for c in last6 if c != '0')
    recent_severe_cnt = sum(1 for c in last6 if c.isdigit() and int(c) >= 4)
    return {
        'length': length,
        'zero_cnt': zero_cnt,
        'bad_cnt': bad_cnt,
        'bad_ratio': bad_cnt / max(1, length),
        'severe_cnt': severe_cnt,
        'severe_ratio': severe_cnt / max(1, length),
        'recent_bad_cnt': recent_bad_cnt,
        'recent_bad_ratio': recent_bad_cnt / max(1, len(last6)),
        'recent_severe_cnt': recent_severe_cnt,
    }


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_FLAG.exists():
        raise FileNotFoundError(f'Missing {TEST_FLAG}')
    tf = pd.read_csv(TEST_FLAG)
    if 'uid' not in tf.columns:
        raise ValueError('test_flag.csv must contain uid')
    uid_set = set(tf['uid'].tolist())

    # Enquiry aggregations
    enq_map = {}
    if ENQUIRY_JSON.exists():
        data = json.load(open(ENQUIRY_JSON, 'r'))
        for rec in data:
            items = rec if isinstance(rec, list) else [rec]
            for it in items:
                uid = it.get('uid')
                if uid in uid_set:
                    e = enq_map.setdefault(uid, {'vals': [], 'types': set(), 'cnt': 0})
                    amt = _safe_number(it.get('enquiry_amt'))
                    if amt is not None:
                        e['vals'].append(amt)
                    t = it.get('enquiry_type')
                    if t is not None:
                        e['types'].add(str(t))
                    e['cnt'] += 1

    # Accounts per-account metrics and aggregates
    acc_map = {}
    if ACCOUNTS_JSON.exists():
        data = json.load(open(ACCOUNTS_JSON, 'r'))
        for rec in data:
            items = rec if isinstance(rec, list) else [rec]
            for it in items:
                uid = it.get('uid')
                if uid in uid_set:
                    a = acc_map.setdefault(uid, {'loan': [], 'ov': [], 'ph': []})
                    la = _safe_number(it.get('loan_amount'))
                    if la is not None:
                        a['loan'].append(la)
                    ov = _safe_number(it.get('amount_overdue'))
                    if ov is not None:
                        a['ov'].append(ov)
                    ph = it.get('payment_hist_string') or it.get('payment_hist')
                    if isinstance(ph, str):
                        a['ph'].append(ph)

    def agg_sum(x): return float(np.sum(x)) if x else 0.0
    def agg_mean(x): return float(np.mean(x)) if x else 0.0
    def agg_max(x): return float(np.max(x)) if x else 0.0
    def agg_std(x): return float(np.std(x)) if x else 0.0

    rows = []
    for uid in uid_set:
        e = enq_map.get(uid, {'vals': [], 'types': set(), 'cnt': 0})
        a = acc_map.get(uid, {'loan': [], 'ov': [], 'ph': []})
        vals = e['vals']
        loan = a['loan']
        ov = a['ov']
        met = [per_account_metrics(ph) for ph in a['ph']]
        lengths = [m['length'] for m in met]
        zero_cnts = [m['zero_cnt'] for m in met]
        bad_cnts = [m['bad_cnt'] for m in met]
        bad_ratios = [m['bad_ratio'] for m in met]
        severe_cnts = [m['severe_cnt'] for m in met]
        severe_ratios = [m['severe_ratio'] for m in met]
        recent_bad_cnts = [m['recent_bad_cnt'] for m in met]
        recent_bad_ratios = [m['recent_bad_ratio'] for m in met]
        recent_severe_cnts = [m['recent_severe_cnt'] for m in met]

        row = {
            'uid': uid,
            'acc_has_account_data': 1 if (loan or ov or met) else 0,
            'acc_loan_amount_count': len(loan),
            'acc_loan_amount_sum': agg_sum(loan),
            'acc_loan_amount_mean': agg_mean(loan),
            'acc_loan_amount_max': agg_max(loan),
            'acc_loan_amount_std': agg_std(loan),
            'acc_amount_overdue_sum': agg_sum(ov),
            'acc_amount_overdue_mean': agg_mean(ov),
            'acc_amount_overdue_max': agg_max(ov),
            'acc_payment_hist_length_sum': agg_sum(lengths),
            'acc_payment_hist_length_mean': agg_mean(lengths),
            'acc_payment_hist_length_max': agg_max(lengths),
            'acc_payment_0_count_sum': agg_sum(zero_cnts),
            'acc_payment_0_count_mean': agg_mean(zero_cnts),
            'acc_bad_payment_count_sum': agg_sum(bad_cnts),
            'acc_bad_payment_count_mean': agg_mean(bad_cnts),
            'acc_bad_payment_count_max': agg_max(bad_cnts),
            'acc_bad_payment_ratio_mean': agg_mean(bad_ratios),
            'acc_bad_payment_ratio_max': agg_max(bad_ratios),
            'acc_bad_payment_ratio_std': agg_std(bad_ratios),
            'acc_severe_delinq_count_sum': agg_sum(severe_cnts),
            'acc_severe_delinq_count_mean': agg_mean(severe_cnts),
            'acc_severe_delinq_ratio_mean': agg_mean(severe_ratios),
            'acc_severe_delinq_ratio_max': agg_max(severe_ratios),
            'acc_recent_bad_count_sum': agg_sum(recent_bad_cnts),
            'acc_recent_bad_count_mean': agg_mean(recent_bad_cnts),
            'acc_recent_bad_ratio_mean': agg_mean(recent_bad_ratios),
            'acc_recent_bad_ratio_max': agg_max(recent_bad_ratios),
            'acc_recent_severe_count_sum': agg_sum(recent_severe_cnts),
            'acc_recent_severe_count_max': agg_max(recent_severe_cnts),
            'enq_has_enquiry_data': 1 if e['cnt'] > 0 else 0,
            'enq_enquiry_amt_count': len(vals),
            'enq_enquiry_amt_sum': agg_sum(vals),
            'enq_enquiry_amt_mean': agg_mean(vals),
            'enq_enquiry_amt_max': agg_max(vals),
            'enq_enquiry_amt_std': agg_std(vals),
            'enq_enquiry_type_nunique': len(e['types']),
            # Derived risks and intensities to match train
            'missing_account_risk': 0.0 if (loan or ov or met) else 1.0,
            'missing_enquiry_risk': 0.0 if e['cnt'] > 0 else 1.0,
            'total_missing_risk': (0.0 if (loan or ov or met) else 1.0) + (0.0 if e['cnt'] > 0 else 1.0),
            'credit_utilization': (agg_sum(ov) / (agg_sum(loan) + 1.0)) if (loan or ov) else 0.0,
            'enquiry_intensity': (len(vals) * agg_mean(vals)) if vals else 0.0,
        }
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    base_cols = ['uid'] + (['NAME_CONTRACT_TYPE'] if 'NAME_CONTRACT_TYPE' in tf.columns else [])
    base_df = tf[base_cols]
    comp_df = base_df.merge(comp_df, on='uid', how='left')
    comp_df.to_csv(OUT_CSV, index=False)
    print(f'✅ Wrote {OUT_CSV} shape={comp_df.shape}')


if __name__ == '__main__':
    main()


