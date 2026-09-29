"""No-patch vs. oracle accuracy from self-patching grids written by multi_fact_experiment_nsamples.py.

Each grid has shape (N, 2, L, L): channel 0 = first-answer-token MRR, channel 1 = greedy exact match,
indexed [question, channel, src layer, tgt layer]. The diagonal is a no-op patch, so cell [0, 0] is the
unpatched model; oracle = fraction of questions where any cell is correct.

Usage:
    python patch/compute_oracle.py <run>/patching_results_entity_offset0.npy [--topk 10] [--json out.json]
"""
import argparse
import json

import numpy as np


def summarize(path, topk):
    arr = np.load(path)
    assert arr.ndim == 4 and arr.shape[1] == 2 and arr.shape[2] == arr.shape[3], f"{path}: expected (N, 2, L, L), got {arr.shape}"
    n, L = arr.shape[0], arr.shape[-1]
    res = {'N': n, 'L': L}
    for name, ok in [('em', arr[:, 1] >= 1.0), ('first_token', arr[:, 0] >= 1.0)]:
        no_patch = ok[:, 0, 0].mean()
        oracle = ok.reshape(n, -1).any(1).mean()
        res[name] = {'no_patch': float(no_patch), 'oracle': float(oracle), 'gain': float(oracle - no_patch),
                     'ratio': float(oracle / no_patch) if no_patch > 0 else None}
    em = arr[:, 1] >= 1.0
    # Sanity check: every diagonal cell is a no-op patch, so it must match [0, 0]
    res['diag_mismatch'] = [i for i in range(L) if not np.array_equal(em[:, i, i], em[:, 0, 0])]
    wrong = ~em[:, 0, 0]
    res['n_wrong'] = int(wrong.sum())
    if topk and wrong.any():
        rescue = em[wrong].mean(0)
        cells = np.argsort(-rescue, axis=None, kind='stable')[:topk]
        res['topk'] = [{'src': int(s), 'tgt': int(t), 'rescue': float(rescue[s, t])}
                       for s, t in zip(*np.unravel_index(cells, rescue.shape))]
    return res


def main():
    parser = argparse.ArgumentParser(description="No-patch and oracle accuracy of (N, 2, L, L) self-patching grids")
    parser.add_argument("grids", nargs="+", help=".npy grids (*_counts.npy files are skipped)")
    parser.add_argument("--topk", type=int, default=0, help="Print the K (src, tgt) cells that rescue most unpatched-wrong questions")
    parser.add_argument("--json", type=str, default=None, help="Write all results to this JSON file")
    args = parser.parse_args()

    results = {}
    for path in args.grids:
        if path.endswith("_counts.npy"):
            continue
        res = results[path] = summarize(path, args.topk)
        print(f"{path}\n  N={res['N']} L={res['L']}")
        if res['diag_mismatch']:
            print(f"  WARNING: diagonal EM differs from [0, 0] at layers {res['diag_mismatch']}")
        for name in ['em', 'first_token']:
            r = res[name]
            ratio = f"{r['ratio']:.2f}x" if r['ratio'] is not None else "n/a"
            print(f"  {name:12s} no_patch={r['no_patch']:.3f} oracle={r['oracle']:.3f} gain={r['gain']:+.3f} ratio={ratio}")
        for c in res.get('topk', []):
            print(f"  src={c['src']:2d} -> tgt={c['tgt']:2d}  rescue rate {c['rescue']:.3f} (of {res['n_wrong']} unpatched-wrong)")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
