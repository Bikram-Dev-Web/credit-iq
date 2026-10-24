# Final presentation outline (11 slides, ~12 minutes + live demo)

Use your own exported numbers. If you present synthetic results, put "synthetic data" on every results slide.

**1. The lending problem**
Lenders must decide under uncertainty. A missed default costs LGD × EAD; rejecting a good borrower costs margin and
customers. Question: can AI estimate default risk well enough to support — not replace — that decision?

**2. What CreditIQ does**
Workflow strip: data → target → features → model → decisions. Ten pages; everything runs locally.

**3. Data and default definition**
Dataset, size, observed default rate, the exact target definition (Home Credit: early payment difficulty), and the
target-mapping safeguard (never assume `1 = default`; German Credit uses 2 = bad).

**4. Guarding against leakage and bias**
Excluded columns and why; the leakage screen catching a post-outcome column; sensitive attributes kept out of the
model but used for the audit. Train / validation / test roles.

**5. Models and performance**
Three algorithms; comparison table (test ROC-AUC, PR-AUC vs base rate, Brier). Why accuracy is not used.

**6. From score to probability**
Calibration curve before and after calibration; why expected loss needs a calibrated PD, not a ranking score.

**7. Threshold trade-offs**
Confusion matrix and the recall/precision/flag-rate curves; what flagging X% of applicants buys.

**8. Why the model says what it says**
Global importance and one local explanation; association ≠ causation; fairness audit headline (one or two group gaps
and their interpretation).

**9. Financial impact**
Portfolio EL by risk band; contribution waterfall; break-even PD and risk-based pricing floor; stress scenarios.

**10. Choosing a lending policy**
Approve-all vs fixed threshold vs cost-minimising vs expected-contribution rule: approvals, declared cost, back-tested
contribution — and why the answer depends on assumptions.

**11. Limitations and recommendations**
Simplified finance (not IFRS 9/capital), target horizon, random-split caveat, fairness scope; next steps: real
LGD/EAD data, out-of-time validation, monitoring, human review.

**Live demo (3–4 minutes)** — follow the 12-step script in the README: load sample → map target → leakage screen →
train → threshold explorer → score an applicant → portfolio EL → stress test → policy comparison → download report.
