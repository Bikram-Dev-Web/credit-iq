# Model card — CreditIQ probability-of-default models

This is the standing model card for the CreditIQ modelling approach. An **experiment-specific card**, filled with the
actual dataset, settings and held-out metrics of the model you trained, is generated in
**Reports & Documentation → Model card** and can be downloaded as Markdown. No performance figures are stated here,
because they depend entirely on the data you train on.

## Model details

- **Developed for**: a university Financial Management course and an AI/finance portfolio project.
- **Model types**: logistic regression (interpretable baseline), random forest, and gradient boosting (LightGBM if
  installed; otherwise XGBoost; otherwise scikit-learn HistGradientBoosting). Each is one scikit-learn pipeline plus a
  probability calibrator.
- **Output**: a raw model score and a calibrated probability that the dataset's defined default event occurs; a risk
  band; an illustrative expected loss given LGD and EAD assumptions.
- **Version**: CreditIQ 1.0.0. Experiment IDs identify individual trained models.

## Intended use

- Learning how credit-risk models are built, validated, explained and connected to expected loss, pricing and
  lending-policy economics.
- Exploring trade-offs between missed defaults, rejected good applicants, review workload and contribution.
- Demonstrating responsible-AI practices: leakage control, calibration, fairness checks, monitoring and documentation.

## Out-of-scope use

- Approving, declining, pricing or limiting credit for real people. Outcomes are labelled "within the selected
  threshold" or "flagged for review" — never approval or denial.
- Regulatory capital, IFRS 9 or CECL provisioning, or any compliance decision.
- Applying a model to a population, product, country or period different from its training data without
  re-validation.
- Any use of the synthetic demonstration results as evidence about real borrowers.

## Training data

Whatever dataset the user loads; the experiment card records its name, fingerprint, size, synthetic flag, target
definition and exclusions. The documented sources (Home Credit, UCI German Credit, LendingClub) and the bundled
synthetic sample are described in `DATA_SOURCES.md`.

## Factors and sensitive attributes

Potentially protected attributes (e.g. gender, age/date of birth, marital status, nationality, foreign-worker status)
are excluded from model inputs by default and kept for a separate group-level audit. Remaining inputs may act as
proxies; the app measures this association. Re-including a sensitive attribute requires an explicit justification.

## Evaluation

- Held-out test split, never used for fitting, calibration, tuning, model selection or threshold choice.
- Discrimination: ROC-AUC, Gini, PR-AUC, KS, lift, decile tables.
- Calibration: reliability curve with confidence intervals, expected calibration error, Brier score, log loss.
- Decisions: confusion matrix, precision, recall, specificity, F1, balanced accuracy at the chosen threshold.
- Stability: metrics by quarter when real dates exist; chronological split option.
- Fairness: per-group flag rates (share above the review threshold), error rates, calibration and AUC.

## Ethical considerations and limitations

- Historical labels reflect past lending decisions and data collection, not only borrower behaviour.
- The Home Credit target is a payment-difficulty event on early instalments, not a full-term default.
- Explanations show learned associations, not causes; correlated inputs make attributions unstable.
- Small groups give noisy fairness estimates; meeting one fairness criterion can violate another.
- Expected-loss and contribution figures inherit the uncertainty of the LGD, EAD, rate and cost assumptions.

## Governance requirements before any real use

1. Independent model validation, including out-of-time testing and benchmarking against simpler models.
2. Legal review of every input and of the reasons given to applicants.
3. Documented human review of flagged applications, with an override and appeal process.
4. Monitoring of input drift, score drift, calibration and group outcomes, with thresholds that trigger review.
5. Change control: every retraining logged with data lineage, configuration and sign-off.

## References

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* Proceedings of FAT* 2019.
- Platt, J. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.*
- Zadrozny, B. & Elkan, C. (2002). *Transforming classifier scores into accurate multiclass probability estimates.* KDD.
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Kleinberg, J., Mullainathan, S. & Raghavan, M. (2016). *Inherent Trade-Offs in the Fair Determination of Risk Scores.*
- Chouldechova, A. (2017). *Fair prediction with disparate impact.* Big Data, 5(2).
