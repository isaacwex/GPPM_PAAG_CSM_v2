"""Supplementary analyses for the PAAG/CSM multi-seed manuscript.

Reproduces, from the saved predictions in results_PAAG_CSM_Multiseed/:
  * per-seed DeLong tests and the DerSimonian-Laird random-effects synthesis (+HKSJ)  -> Table 8
  * validation-threshold sensitivity analysis (seeds 101, 2024)
  * Platt recalibration Brier scores (seeds 101, 2024)
  * Wilcoxon tests on per-image IoU in the primary (TestViz) and replication (BestModel) samples -> Table 4
Usage: python cross_seed_analysis.py path/to/results_PAAG_CSM_Multiseed
"""
import sys, os, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_curve, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'results_PAAG_CSM_Multiseed'
V = ['base', 'paag', 'csm', 'paag_csm']
SEEDS = {42: '', 101: 'seed_101/test_rescored/', 2024: 'seed_2024/test_rescored/'}
PAIRS = [('base', 'paag'), ('base', 'csm'), ('base', 'paag_csm'), ('paag', 'csm'), ('paag', 'paag_csm'), ('csm', 'paag_csm')]

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out

def delong(y, p1, p2):
    """Fast DeLong test for two correlated ROC curves. Returns dAUC (p1-p2), SE, p."""
    y = np.asarray(y).astype(bool); m, n = y.sum(), (~y).sum(); V10, V01, aucs = [], [], []
    for p in (p1, p2):
        p = np.asarray(p, float); pos, neg = p[y], p[~y]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
        aucs.append(tz[:m].sum() / m / n - (m + 1) / (2 * n))
        V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    d = aucs[0] - aucs[1]; se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return d, se, 2 * stats.norm.sf(abs(d / se))

def random_effects(d, se):
    d, se = np.asarray(d), np.asarray(se); w = 1 / se**2; k = len(d)
    fe = (w * d).sum() / w.sum(); Q = (w * (d - fe)**2).sum()
    C = w.sum() - (w**2).sum() / w.sum(); tau2 = max(0.0, (Q - (k - 1)) / C)
    wr = 1 / (se**2 + tau2); mu = (wr * d).sum() / wr.sum(); se_mu = np.sqrt(1 / wr.sum())
    p = 2 * stats.norm.sf(abs(mu / se_mu)); I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    q = (wr * (d - mu)**2).sum() / (k - 1); se_h = np.sqrt(q / wr.sum())
    p_h = 2 * stats.t.sf(abs(mu / se_h), k - 1); t = stats.t.ppf(0.975, k - 1)
    return dict(pooled=mu, ci=(mu - 1.96 * se_mu, mu + 1.96 * se_mu), p=p, tau=np.sqrt(tau2), I2=I2,
                ci_hksj=(mu - t * se_h, mu + t * se_h), p_hksj=p_h)

def load(v, sub):
    return np.load(os.path.join(ROOT, 'BestModel', v, sub, 'predictions.npz'))

def f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p); f = 2 * pr * rc / (pr + rc + 1e-12)
    return th[np.nanargmax(f[:-1])]

y = load('base', '')['y_true']
P = {(v, s): load(v, sub)['y_scores'] for v in V for s, sub in SEEDS.items()}

print('=== Test ROC-AUC by seed')
for s in SEEDS:
    from sklearn.metrics import roc_auc_score
    print(s, {v: round(roc_auc_score(y, P[(v, s)]), 4) for v in V})

print('\n=== Pairwise dAUC per seed and random-effects synthesis (Table 8)')
rows = []
for a, b in PAIRS:
    res = [delong(y, P[(a, s)], P[(b, s)]) for s in SEEDS]
    re_ = random_effects([r[0] for r in res], [r[1] for r in res])
    rows.append({'comparison': f'{a} vs {b}', **{f'seed{s}_dAUC': r[0] for s, r in zip(SEEDS, res)},
                 **{f'seed{s}_p': r[2] for s, r in zip(SEEDS, res)},
                 'n_sig': sum(r[2] < 0.05 for r in res), 'pooled': re_['pooled'], 'ci_low': re_['ci'][0],
                 'ci_high': re_['ci'][1], 'p': re_['p'], 'I2': re_['I2'], 'p_hksj': re_['p_hksj']})
tab = pd.DataFrame(rows); print(tab.round(4).to_string(index=False)); tab.to_csv('cross_seed_synthesis.csv', index=False)

print('\n=== Validation-threshold sensitivity and Platt recalibration (seeds 101, 2024)')
logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))).reshape(-1, 1)
out = []
for v in V:
    for s in ('seed_101', 'seed_2024'):
        va = np.load(os.path.join(ROOT, 'BestModel', v, s, 'predictions.npz'))
        te = P[(v, int(s.split('_')[1]))]
        f_test = f1_score(y, te >= f1_threshold(y, te)); f_val = f1_score(y, te >= f1_threshold(va['y_true'], va['y_scores']))
        lr = LogisticRegression().fit(logit(va['y_scores']), va['y_true'])
        out.append(dict(variant=v, seed=s, F1_test_threshold=f_test, F1_val_threshold=f_val,
                        brier_raw=brier_score_loss(y, te), brier_platt=brier_score_loss(y, lr.predict_proba(logit(te))[:, 1])))
cal = pd.DataFrame(out); print(cal.round(4).to_string(index=False)); print(cal.groupby('variant').mean(numeric_only=True).round(4))
print('Prevalence-only Brier reference:', round(float(((y - y.mean())**2).mean()), 4))

print('\n=== Wilcoxon signed-rank on per-image IoU (Table 4)')
for name, src in (('primary', 'TestViz'), ('replication', 'BestModel')):
    D = {v: pd.read_csv(os.path.join(ROOT, src, v, 'LocalizationIoU', 'per_image_iou.csv')).set_index('Image Index')['IoU'] for v in V}
    print(name, {v: f'{D[v].mean():.4f} ± {D[v].std(ddof=0):.4f}' for v in V})
    for a, b in itertools.combinations(V, 2):
        idx = D[a].index.intersection(D[b].index)
        print(f'  {a} vs {b}: n={len(idx)} p={stats.wilcoxon(D[a][idx], D[b][idx]).pvalue:.3g}')
