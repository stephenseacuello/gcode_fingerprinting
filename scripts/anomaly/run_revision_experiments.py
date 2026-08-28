#!/usr/bin/env python3
"""Revision experiments (R1 response, 2026-06-17).

CPU-only experiments scored from cached V7 logits, addressing referee findings:
  - A9 compound attack (A1 command swap + A2 coordinate shift)  [M14]
  - A5 feed-rate manipulation feasibility check                 [M14]
  - DeLong test: rank vs NLL on A2 small-delta (real p-values)  [M5]
  - Holm-Bonferroni correction across the pairwise family       [M6]
  - Adaptive adversary: per-window evasive delta selection      [M15]
  - Two-population per-token AUROC (confident vs uncertain)      [B3-twopop]

All numbers are computed from outputs/anomaly20260319/cached_inference (V7,
best_config_5fold @ max_token_len=6). Validation anchor: A1 V7 S_mean = 0.803.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.anomaly.anomaly_scoring_utils import (
    BASE_DIR, FOLDS, PAD_ID, EOS_ID,
    load_vocab, load_cached_logits, load_cached_targets,
    get_axis_values, build_name_to_id,
    compute_per_token_nll, compute_window_scores, compute_rank_scores,
    compute_auroc, bootstrap_auroc_ci, cohens_d,
    inject_command_swap, inject_coordinate_shift,
    holm_bonferroni, delong_test, save_results, setup_logging,
)
import logging

logger = logging.getLogger(__name__)
OUT = BASE_DIR / "revision_2026_06"

SCORINGS = ["S_mean", "S_max"]


def _auroc_pack(normal, attack):
    info = bootstrap_auroc_ci(normal, attack, n_bootstrap=5000)
    return {"auroc": info["auroc"], "ci": [info["ci_lower"], info["ci_upper"]],
            "cohens_d": cohens_d(attack, normal), "n_normal": len(normal), "n_attack": len(attack)}


# ----------------------------------------------------------------------
# A9: compound attack (A1 command swap + A2 X-shift delta=0.1mm) on same window
# ----------------------------------------------------------------------
def inject_compound(tokens, vocab, delta_mm=0.1, axis="X"):
    """A9: apply BOTH a command swap and a coordinate shift in one window."""
    name2id = build_name_to_id(vocab)
    swap = {name2id["G0"]: name2id["G1"], name2id["G1"]: name2id["G0"]}
    axis_vals = get_axis_values(axis, vocab)
    values = np.array([v for v, _ in axis_vals]); ids = np.array([i for _, i in axis_vals])
    tid_to_val = {int(i): v for v, i in axis_vals}

    attacked, orig = [], []
    N, L = tokens.shape
    for i in range(N):
        seq = tokens[i].clone()
        did_cmd = did_coord = False
        for j in range(L):
            t = int(seq[j].item())
            if not did_cmd and t in swap:
                seq[j] = swap[t]; did_cmd = True
            elif not did_coord and t in tid_to_val:
                target = tid_to_val[t] + delta_mm
                nidx = int(np.argmin(np.abs(values - target)))
                if int(ids[nidx]) != t:
                    seq[j] = int(ids[nidx]); did_coord = True
        if did_cmd and did_coord:               # require a genuine compound
            attacked.append(seq); orig.append(i)
    if not attacked:
        return torch.empty(0, L, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.stack(attacked), torch.tensor(orig, dtype=torch.long)


def exp_a9(folds):
    per_fold = {s: [] for s in SCORINGS}
    n_attacks = []
    for fold in folds:
        lg = load_cached_logits(fold)["legacy_logits"]
        td = load_cached_targets(fold)
        tt, pm, ln = td["target_tokens"], td["padding_mask"], td["lengths"]
        nll_n = compute_per_token_nll(lg, tt, pm)
        ns = compute_window_scores(nll_n, ln)
        att, oi = inject_compound(tt, load_vocab())
        n_attacks.append(int(len(att)))
        if len(att) == 0:
            continue
        nll_a = compute_per_token_nll(lg[oi], att, pm[oi])
        as_ = compute_window_scores(nll_a, ln[oi])
        for s in SCORINGS:
            per_fold[s].append(compute_auroc(ns[s][oi].numpy(), as_[s].numpy()))
    out = {"attack": "A9_compound_A1+A2(X,0.1mm)", "n_attacks_per_fold": n_attacks}
    for s in SCORINGS:
        v = per_fold[s]
        out[s] = {"mean_auroc": float(np.mean(v)), "std_auroc": float(np.std(v)), "per_fold": v}
    return out


# ----------------------------------------------------------------------
# A5: feed-rate manipulation feasibility
# ----------------------------------------------------------------------
def exp_a5_feasibility():
    vocab = load_vocab()
    f_vals = get_axis_values("F", vocab)
    s_vals = get_axis_values("S", vocab)
    # count how many F tokens actually appear in the test targets
    appear = 0; total_F = 0
    f_ids = {i for _, i in f_vals}
    for fold in FOLDS:
        tt = load_cached_targets(fold)["target_tokens"]
        m = torch.zeros_like(tt, dtype=torch.bool)
        for fid in f_ids:
            m |= (tt == fid)
        total_F += int(m.sum())
    return {
        "n_F_tokens_in_vocab": len(f_vals),
        "n_S_tokens_in_vocab": len(s_vals),
        "F_values": [round(v, 4) for v, _ in f_vals],
        "total_F_token_occurrences_in_test": total_F,
        "evaluable": len(f_vals) >= 2,
        "reason": ("Feed rate is constant in this corpus (single F vocabulary token); "
                   "any A5 feed-scaling maps back to the same token, so A5 is a no-op and "
                   "cannot be evaluated on this dataset. This is a dataset limitation, not a "
                   "method limitation."),
    }


# ----------------------------------------------------------------------
# Significance: DeLong (rank vs NLL) + Holm-Bonferroni across family
# ----------------------------------------------------------------------
def _scores_for_a2(fold, axis, delta):
    """Return (y_true, nll_scores, rank_scores) pooling normal+attack windows."""
    vocab = load_vocab()
    lg = load_cached_logits(fold)["legacy_logits"]
    td = load_cached_targets(fold)
    tt, pm, ln = td["target_tokens"], td["padding_mask"], td["lengths"]
    att, oi, _ = inject_coordinate_shift(tt, axis, delta, vocab)
    if len(att) == 0:
        return None
    # NLL (S_mean)
    nll_n = compute_per_token_nll(lg, tt, pm); ns_nll = compute_window_scores(nll_n, ln)["S_mean"].numpy()[oi.numpy()]
    nll_a = compute_per_token_nll(lg[oi], att, pm[oi]); as_nll = compute_window_scores(nll_a, ln[oi])["S_mean"].numpy()
    # Rank (S_rank_mean): normal targets vs attacked targets under same logits
    rk_n = compute_rank_scores(lg[oi], tt[oi], vocab["vocab"])["S_rank_mean"].numpy()
    rk_a = compute_rank_scores(lg[oi], att, vocab["vocab"])["S_rank_mean"].numpy()
    y = np.concatenate([np.zeros(len(oi)), np.ones(len(oi))])
    nll = np.concatenate([ns_nll, as_nll])
    rank = np.concatenate([rk_n, rk_a])
    return y, nll, rank


def exp_significance(folds):
    family = []   # (label, axis, delta)
    for d in [0.001, 0.01, 0.1]:
        family.append((f"A2_X_delta{d}", "X", d))
    results = {}
    pooled_p = {}
    for label, axis, delta in family:
        per_fold_p, auroc_rank, auroc_nll = [], [], []
        ys, nlls, ranks = [], [], []
        for fold in folds:
            r = _scores_for_a2(fold, axis, delta)
            if r is None:
                continue
            y, nll, rank = r
            ys.append(y); nlls.append(nll); ranks.append(rank)
            auroc_nll.append(roc_auc_score(y, nll)); auroc_rank.append(roc_auc_score(y, rank))
            per_fold_p.append(delong_test(y, rank, nll))
        Y = np.concatenate(ys); NLL = np.concatenate(nlls); RANK = np.concatenate(ranks)
        p_pooled = delong_test(Y, RANK, NLL)
        pooled_p[label] = p_pooled
        results[label] = {
            "auroc_rank_mean": float(np.mean(auroc_rank)),
            "auroc_nll_mean": float(np.mean(auroc_nll)),
            "delta_auroc": float(np.mean(auroc_rank) - np.mean(auroc_nll)),
            "delong_p_pooled": p_pooled,
            "delong_p_per_fold": per_fold_p,
        }
    # Holm-Bonferroni across the pooled family
    labels = list(pooled_p.keys()); ps = [pooled_p[l] for l in labels]
    adj = holm_bonferroni(ps)
    for l, a in zip(labels, adj):
        results[l]["delong_p_holm_adjusted"] = a
    return {"comparisons": results, "family_size": len(labels),
            "note": "DeLong (bootstrap, 10k) for rank-vs-NLL AUROC; Holm-Bonferroni across the family of pooled comparisons."}


# ----------------------------------------------------------------------
# Adaptive adversary: per-window evasive delta selection
# ----------------------------------------------------------------------
def exp_adaptive(folds, scoring="S_max", min_meaningful=0.1):
    """Non-adaptive: fixed delta. Adaptive: attacker picks, per window, the delta
    (>= min_meaningful mm, a sabotage-relevant magnitude) that MINIMIZES the detector
    score, i.e. best evasion. Compares detectability."""
    vocab = load_vocab()
    deltas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    fixed_auroc, adaptive_auroc = [], []
    for fold in folds:
        lg = load_cached_logits(fold)["legacy_logits"]
        td = load_cached_targets(fold)
        tt, pm, ln = td["target_tokens"], td["padding_mask"], td["lengths"]
        nll_n = compute_per_token_nll(lg, tt, pm); ns = compute_window_scores(nll_n, ln)[scoring].numpy()
        # For each delta, score attacks; track per-window attack score
        per_delta_scores = {}
        ref_oi = None
        for d in deltas:
            att, oi, _ = inject_coordinate_shift(tt, "X", d, vocab)
            if len(att) == 0:
                continue
            nll_a = compute_per_token_nll(lg[oi], att, pm[oi]); a = compute_window_scores(nll_a, ln[oi])[scoring].numpy()
            # map to a common window index space using oi
            per_delta_scores[d] = (oi.numpy(), a)
            if ref_oi is None or len(oi) < len(ref_oi):
                ref_oi = oi.numpy()
        # restrict to windows attackable at ALL deltas (intersection)
        common = None
        for d, (oi, a) in per_delta_scores.items():
            s = set(oi.tolist())
            common = s if common is None else (common & s)
        common = sorted(common)
        if not common:
            continue
        idx_of = {w: k for k, w in enumerate(common)}
        # build per-window score matrix [n_common, n_delta]
        M = np.full((len(common), len(per_delta_scores)), np.nan)
        dlist = list(per_delta_scores.keys())
        for di, d in enumerate(dlist):
            oi, a = per_delta_scores[d]
            for w, sc in zip(oi.tolist(), a):
                if w in idx_of:
                    M[idx_of[w], di] = sc
        normal_common = ns[np.array(common)]
        # fixed: use delta=0.1 (first meaningful)
        fixed_idx = dlist.index(min_meaningful)
        fixed_attack = M[:, fixed_idx]
        fixed_auroc.append(compute_auroc(normal_common, fixed_attack))
        # adaptive: attacker minimizes detector score per window across deltas
        adaptive_attack = np.nanmin(M, axis=1)
        adaptive_auroc.append(compute_auroc(normal_common, adaptive_attack))
    return {"scoring": scoring, "fixed_delta_mm": min_meaningful, "candidate_deltas": deltas,
            "fixed_auroc_mean": float(np.mean(fixed_auroc)), "fixed_auroc_per_fold": fixed_auroc,
            "adaptive_auroc_mean": float(np.mean(adaptive_auroc)), "adaptive_auroc_per_fold": adaptive_auroc,
            "evasion_drop": float(np.mean(fixed_auroc) - np.mean(adaptive_auroc)),
            "note": "Adaptive attacker selects, per window, the delta>=0.1mm minimizing the detector score (best evasion)."}


# ----------------------------------------------------------------------
# Two-population per-token AUROC (confident vs uncertain)
# ----------------------------------------------------------------------
def exp_two_population(folds, delta=0.001, axis="X", conf_hi=0.5, conf_lo=0.01):
    vocab = load_vocab(); V = vocab["vocab"]
    id2tok = {v: k for k, v in V.items()}
    axis_token_ids = sorted([i for n, i in V.items() if n.startswith(f"NUM_{axis}_")])
    conf_n = unc_n = 0
    conf_pairs = []   # (normal_nll, attack_nll) at confident positions
    unc_pairs = []
    for fold in folds:
        lg = load_cached_logits(fold)["legacy_logits"]
        td = load_cached_targets(fold)
        tt, pm = td["target_tokens"], td["padding_mask"]
        att, oi, _ = inject_coordinate_shift(tt, axis, delta, vocab)
        if len(att) == 0:
            continue
        probs = torch.softmax(lg, dim=-1)         # [N,L,V]
        N, L = tt.shape
        # find, for each attacked sequence, the single changed position
        for k in range(len(oi)):
            i = int(oi[k].item())
            diff = (att[k] != tt[i]).nonzero(as_tuple=True)[0]
            if len(diff) == 0:
                continue
            j = int(diff[0].item())
            maxp = float(probs[i, j].max().item())
            # NLL of original vs attacked token at this position
            nll_orig = float(-torch.log(probs[i, j, tt[i, j]].clamp_min(1e-12)).item())
            nll_att = float(-torch.log(probs[i, j, att[k, j]].clamp_min(1e-12)).item())
            if maxp > conf_hi:
                conf_n += 1; conf_pairs.append((nll_orig, nll_att))
            elif maxp < conf_lo:
                unc_n += 1; unc_pairs.append((nll_orig, nll_att))

    def pop_auroc(pairs):
        if len(pairs) < 5:
            return None, len(pairs)
        normal = np.array([p[0] for p in pairs]); attack = np.array([p[1] for p in pairs])
        y = np.concatenate([np.zeros(len(normal)), np.ones(len(attack))])
        s = np.concatenate([normal, attack])
        try:
            return float(roc_auc_score(y, s)), len(pairs)
        except ValueError:
            return None, len(pairs)

    conf_auroc, _ = pop_auroc(conf_pairs)
    unc_auroc, _ = pop_auroc(unc_pairs)
    tot = conf_n + unc_n
    return {"axis": axis, "delta": delta, "conf_threshold": conf_hi, "uncertain_threshold": conf_lo,
            "n_confident": conf_n, "n_uncertain": unc_n,
            "frac_confident": (conf_n / tot if tot else None), "frac_uncertain": (unc_n / tot if tot else None),
            "confident_per_token_auroc": conf_auroc, "uncertain_per_token_auroc": unc_auroc,
            "mean_delta_nll_uncertain": float(np.mean([b - a for a, b in unc_pairs])) if unc_pairs else None}


def main():
    setup_logging("revision")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    logger.info("A9 compound ...");          results["A9_compound"] = exp_a9(FOLDS)
    logger.info("A5 feasibility ...");        results["A5_feasibility"] = exp_a5_feasibility()
    logger.info("Significance (DeLong+Holm) ..."); results["significance"] = exp_significance(FOLDS)
    logger.info("Adaptive adversary ...");    results["adaptive_adversary"] = exp_adaptive(FOLDS)
    logger.info("Two-population ...");        results["two_population"] = exp_two_population(FOLDS)
    save_results(results, OUT, "revision_experiments.json")
    print("\n==== REVISION EXPERIMENT RESULTS ====")
    import json
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
