# [Project title] — AI-Driven Credit Risk and Lending Decisions

*Financial Management — [course code] · [your name(s)] · [date]*

> How to use this template: replace every bracketed instruction with your own text and with numbers taken from the
> CreditIQ exports (Reports & Documentation → Project report HTML/Markdown, metrics JSON, policy and scenario CSVs).
> Label each number as **measured** (held-out model result), **observed** (historical data), **assumed** (your input)
> or **simulated** (projection). If you used the synthetic sample, say so in every results section.

## 1. Executive summary (½ page)
- [The lending question in one sentence.]
- [Data used and the default definition.]
- [Best model and its test ROC-AUC / PR-AUC / Brier — measured.]
- [Expected loss of the test portfolio and how it changes under stress — simulated, with key assumptions.]
- [Recommended lending policy and why; its main risk.]

## 2. Problem statement
[Why lenders need PD estimates; the cost asymmetry between missed defaults and rejected good borrowers; what decision
the analysis supports and what it does not.]

## 3. Dataset and target definition
- Source, access date, licence/terms, number of rows and columns loaded, sampling applied.
- Target column, label(s) mapped to default, observed default rate, unmapped rows excluded.
- Exact target definition (quote the dataset documentation) and the observation horizon.
- Columns excluded and why (identifiers, leakage, sensitive attributes, dates). Include the leakage-screen table.

## 4. Financial theory
- PD, LGD, EAD and expected loss = PD × LGD × EAD; unexpected loss (conceptually) vs expected loss.
- Risk-based pricing: break-even rate covers funding, expected loss and operating cost.
- Contribution vs accounting profit; NPV of a risky loan; why principal loss must not be counted twice.
- IFRS 9 / Basel context: what this simplified model does **not** do (staging, lifetime PD, capital).

## 5. Exploratory data analysis
[3–5 findings from the EDA page with charts: default rate by key segments, missing-value patterns, strongest
single-feature signals. State that these are associations.]

## 6. Methodology
- Split strategy and sizes; why random vs chronological.
- Preprocessing (imputation, encoding, domain ratios) fitted on training rows only.
- Models compared; class-imbalance handling; calibration method; threshold rule; tuning (if any).
- What the validation set was used for and why test metrics are the ones reported.

## 7. Model comparison
[Table from the Model Lab export: validation and test ROC-AUC, PR-AUC, Brier, KS, recall/precision at threshold,
overfitting gap. Which model was selected and why.]

## 8. Performance and calibration
[ROC and PR curves; calibration curve and ECE; confusion matrix at the chosen threshold; lift/decile table. Interpret
for a lender: what share of defaulters is caught when X% of applicants are flagged?]

## 9. Explainability and fairness
- Global importance (permutation or SHAP) and two local explanations (one high-risk, one low-risk applicant).
- Partial-dependence curve for one key variable.
- Group-level audit for at least one sensitive attribute: flag rate, TPR, FPR, calibration gaps; proxy variables.
- Limits of these analyses.

## 10. Expected loss
[Portfolio KPIs: total EAD, observed default rate, average PD, expected loss and EL/EAD. Breakdown by risk band and one
segment; loss concentration. LGD/CCF assumptions and their justification.]

## 11. Profitability analysis
[Lending-economics waterfall; break-even PD and break-even rate; sensitivity tornado; risk-based pricing floor by
band. State everything the simplified contribution omits.]

## 12. Threshold and policy comparison
[Policy table: approve-all, fixed threshold, cost-minimising threshold, expected-contribution rule — confusion counts,
approval rate, total declared cost, back-tested contribution. Cost assumptions and their rationale. Why no threshold
is universally optimal.]

## 13. Stress testing
[Scenario table (improved, base, adverse, severe): PD multipliers, LGD shifts, EAD multipliers and their source or
rationale; resulting EL and contribution. Label as illustrative unless calibrated to evidence.]

## 14. Loan-level NPV (optional)
[Cash flows of a representative loan; contractual vs risk-adjusted NPV; NPV vs PD curve.]

## 15. Results
[Bring the measured, observed, assumed and simulated results together in one table.]

## 16. Limitations
[Data, target definition, sampling, split, model, calibration, financial simplifications, fairness scope, synthetic
data if used.]

## 17. Recommendations
[For a lender considering such a model: data improvements, validation steps, policy choice, governance, monitoring.]

## 18. References
[Dataset citation(s); textbook(s) used in the course; methodological references from docs/MODEL_CARD.md and
docs/METHODOLOGY.md. Cite only sources you have actually read.]

## Appendix
- Experiment ID, configuration JSON and model card (export from the app).
- Data-quality report.
- Screenshots of the dashboard pages used.
