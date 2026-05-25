#!/usr/bin/env bash
# T3.1 ingestion: runs once the K=2418/SS=0/dw=0 control finishes (training
# + Phase-2 eval producing beam_0/beam_1 predictions on all 5 folds).
# Re-scores the K-spectrum with the matched-methodology control included
# as a 5th variant, re-runs inferential stats (now exposing the
# "K=335 vs K=2418_designB" within-methodology contrast that the paper
# currently cannot make), and prints a comparison summary the paper update
# can be written from.
#
# Usage: bash scripts/analysis/k_spectrum_t3_1_ingest.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

CK="outputs/decoder20260511/checkpoints/full_window_5fold_designB_K2418"

# ---- gate: require all 5 folds + Phase-2 eval (beam_0 + beam_1) ------------
missing=()
for F in 1 2 3 4 5; do
  for B in beam_0 beam_1; do
    p="$CK/fold_$F/results/${B}_all_predictions.json"
    if [ ! -f "$p" ]; then missing+=("$p"); fi
  done
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "T3.1 ingest: NOT READY. Missing prediction files:"
  for p in "${missing[@]}"; do echo "  $p"; done
  echo
  echo "If training is done but Phase-2 eval has not run, the Phase-2 block"
  echo "of scripts/experiments/train_v8_full_window_5fold_nonum.sh handles it"
  echo "(loops 5 folds, --eval_only --beam_widths 0,1)."
  exit 1
fi
echo "T3.1 ingest: all 5 folds x {beam_0, beam_1} predictions present."

# ---- score + stats ---------------------------------------------------------
echo
echo "=== k_spectrum_compare.py (will detect T3.1 control automatically) ==="
python3 scripts/analysis/k_spectrum_compare.py

echo
echo "=== k_spectrum_stats.py (extends to 5 variants + matched-methodology contrasts) ==="
python3 scripts/analysis/k_spectrum_stats.py

# ---- print T3.1-specific summary -------------------------------------------
echo
echo "=== T3.1 verdict summary ==="
python3 - <<'PY'
import json, statistics as st
from pathlib import Path
d = json.loads(Path("outputs/decoder20260511/audit/k_spectrum_stats.json").read_text())
if not d.get("T3.1_control_present"):
    print("NOTE: T3.1 control not flagged present in stats JSON.")
    raise SystemExit(0)

per_K = d["per_K_summary"]["struct_token_acc"]
hl = per_K["K2418"]
ctrl = per_K["K2418_designB"]
k335 = per_K["K335"]

print(f"\nAR struct token accuracy (5-fold mean +/- sd, [bootstrap 95% CI]):")
print(f"  K2418 headline (SS=0.5, dw=1.0)        : {hl['mean']:.4f} +/- {hl['sd']:.4f}  CI={hl['bootstrap_95ci_mean']}")
print(f"  K2418 T3.1 control (SS=0, dw=0)        : {ctrl['mean']:.4f} +/- {ctrl['sd']:.4f}  CI={ctrl['bootstrap_95ci_mean']}")
print(f"  K=335 (2-digit, SS=0, dw=0)            : {k335['mean']:.4f} +/- {k335['sd']:.4f}  CI={k335['bootstrap_95ci_mean']}")

# the key comparison: K=335 vs the matched-methodology K=2418
matched = d["pairwise_vs_K2418_designB_matched"]["struct_token_acc"]["K335_vs_K2418_designB"]
print(f"\nMATCHED-METHODOLOGY pairwise: K=335 vs K=2418_designB (both SS=0, dw=0):")
print(f"  paired t({matched['n']-1}) = {matched['paired_t']:+.3f}, p = {matched['paired_t_p']:.4f}")
print(f"  Holm-adjusted p = {matched['paired_t_p_holm']:.4f}")
print(f"  Wilcoxon p = {matched['wilcoxon_p']}")
print(f"  mean diff = {matched['mean_diff']:+.4f} ({matched['n_positive']}/{matched['n']} positive)")
print(f"  Cohen's d_z = {matched['cohens_dz']:+.3f}  Hedges' g_z = {matched['hedges_gz']:+.3f}")
print(f"  bootstrap 95% CI on paired diff = {matched['bootstrap_95ci_diff']}")

# verdict logic
p_matched = matched["paired_t_p"]
mean_diff = matched["mean_diff"]
ci = matched["bootstrap_95ci_diff"]
print()
if mean_diff > 0 and ci[0] > 0:
    print("VERDICT (matched-methodology): K=335 EXCEEDS the matched K=2418 control;")
    print("the original 'K=335 structural sweet spot' framing can be re-strengthened")
    print("on the matched-methodology basis. Paper update: lift the 'cannot be cleanly")
    print("attributed' caveat and replace with the matched-methodology statistic.")
elif mean_diff <= 0 or (ci[0] < 0 < ci[1] and abs(mean_diff) < 0.02):
    print("VERDICT (matched-methodology): K=335 does NOT exceed the matched K=2418 control;")
    print("the +9-11 pp K=335 advantage in the original spectrum is attributable to the")
    print("SS/dw schedule change, not vocabulary cardinality. Paper update: replace 'K=335")
    print("sweet spot' with 'SS=0/dw=0 schedule is the cleaner training configuration for")
    print("AR per-token recovery; K cardinality among {24, 69, 335, 2418} has a much smaller effect.'")
else:
    print("VERDICT (matched-methodology): ambiguous; the matched contrast is small in magnitude")
    print("and not clearly distinguishable from zero. Paper should retain the methodology caveat.")

PY

echo
echo "T3.1 ingestion complete. Next step: review the verdict above and apply"
echo "the corresponding paper update (sec:abl-nonumeric, Abstract, Conclusion 2a,")
echo "sec:lim-eval paragraph, sec:abl-vocab, fig:k_spectrum, tab:k_spectrum)."
