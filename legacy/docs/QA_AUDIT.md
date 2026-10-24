# QA audit — CreditIQ

Independent audit of the CreditIQ build against the master prompt's requirements (§5–§21) and acceptance criteria
(§24). Every result below comes from a command or browser session that was actually run, on Linux, with the bundled
**synthetic** data and with a synthetic user CSV created for the audit. No real borrower data was used.

## Environments and how the checks were run

| Environment | Versions | Result |
|---|---|---|
| Main | Python 3.13, Streamlit 1.66.1, pandas 3.0.5, scikit-learn 1.9.1, LightGBM 4.7, SHAP 0.54 (`requirements-lock.txt`) | **124 passed** |
| Minimum versions, no optional extras | Python 3.11, Streamlit 1.46.1, pandas 2.2.3, scikit-learn 1.4.2, NumPy 2.0.2 (no LightGBM/SHAP) | **124 passed** |
| Fresh venv from `requirements.txt` | Python 3.12, Streamlit 1.66.1, pandas 3.1.0 | **124 passed** (FutureWarnings treated as errors in an earlier pass) |

- `pyflakes` over `src/`, `app_pages/`, `scripts/`, `tests/`, `app.py`: no findings.
- App launched with `streamlit run app.py`; health endpoint OK; driven in headless Chromium with Playwright.
- **Browser run 1 (synthetic demo):** load → confirm target → leakage screen → save features → train three models →
  every page → assess applicant → monitoring check → 22 downloads. No exceptions, no console errors, 22/22 downloads.
- **Browser run 2 (user CSV):** a 2,500-row semicolon-separated, Latin-1 CSV with text labels
  (`Charged Off` / `Fully Paid` / `Current`) was uploaded, mapped, trained, scored, saved as a bundle, **re-loaded
  from `models/` as the active model**, scored again, and the report saved. No exceptions or error messages. The
  mapping counts shown in the UI (227 defaults, 2,092 non-defaults, 181 unmapped) equal the counts in the file; the
  sequential `loan_id` was excluded as an identifier and the accented `Süd` level survived.
- **Dashboard numbers vs exported data:** 20 cross-checks between tiles and the downloaded CSV/JSON files all
  agreed (portfolio EL = Σ PD × LGD × EAD over the loan-level export; Overview EL = Portfolio EL = stress *Base* EL =
  sum of risk-band EL; ROC-AUC, PR-AUC and Brier recomputed with scikit-learn from the exported test predictions; the
  policy table's TP/FP recomputed from the exported predictions and threshold; NPV = Σ discounted expected cash
  flows; simulator EL = PD × LGD × EAD from the assessment JSON).
- **Export/import:** the bundle downloaded through the browser was loaded in a separate Python process and
  re-scored the 4,000 test applicants; maximum absolute PD difference from the exported predictions: 1e-16.
- **README results table:** re-running the documented CLI command reproduced every figure to the published
  precision (e.g. Logistic Regression validation ROC-AUC 0.798, test 0.813).
- **Regression evidence:** the first 22 regression tests written in this audit (17 unit, 5 UI) were run against the
  pre-audit ZIP; 19 failed there and all pass now, confirming they exercise real defects.

## Defects found and fixed

| # | Area | Defect reproduced | Fix | Test |
|---|---|---|---|---|
| 1 | Target mapping | Float labels (`1.0/0.0`) never matched integer choices, so every row became unmapped; `"1"` vs `1` raised "values not found". | Canonical label keys (`label_key`) for mapping and negative-class derivation. | `test_target_mapping_matches_float_and_string_labels` |
| 2 | Feature roles | Near-unique **amount** columns were flagged as identifiers and silently excluded; `OWN_CAR_AGE` was flagged as a sensitive "age". | Identifier rule needs an ID-like name or a packed integer sequence; age pattern excludes `car_age`. | `test_sequential_integer_id_is_flagged_but_amounts_are_not` |
| 3 | CSV loading | Semicolon files loaded as one column; header-only files loaded as an empty table; binary files parsed as garbage. | Separator/encoding sniffing (`,` `;` tab `|`, UTF-8/Latin-1); clear errors for header-only and binary files. | `test_semicolon_and_latin1_csv_are_parsed`, `test_header_only_and_binary_csv_are_rejected_clearly` |
| 4 | Provenance | An uploaded copy of the bundled sample was not labelled synthetic; monitoring provenance relied on a file-name prefix. | Signature-based detection; monitoring uses the loader's flag. | `test_bundled_sample_is_flagged_synthetic_when_uploaded` |
| 5 | Data leakage | The binding leakage screen ran on **all** labelled rows (including test rows) in the UI's suggested set and the CLI; a leaky column was kept when the screen was skipped. | Leakage guard inside training, on the **training split only**; Data Center screen labelled advisory. | `test_leakage_guard_screens_training_rows_only` |
| 6 | Probability interpretation | With calibration off, outputs were still labelled "calibrated PD". | `is_calibrated` flag; warnings and relabelled tiles on Overview, Simulator, Portfolio, Finance and Explainability. | `test_uncalibrated_model_is_reported_as_such`, `test_uncalibrated_model_is_labelled_as_a_score_in_the_simulator` |
| 7 | Thresholds | The threshold could be set while viewing **test** results. | Explorer defaults to validation; test view is read-only. | `test_threshold_explorer_defaults_to_validation_and_test_view_is_read_only` |
| 8 | Stale results | Changing the threshold, switching the active model or loading a bundle left old policy/fairness/scenario tables in the session (and in the report). | Model-level invalidation in `state.set_active_model` / `state.set_threshold`. | `test_threshold_override_invalidates_results_and_reaches_policy_page`, `test_switching_active_model_clears_downstream_results` |
| 9 | Model mixing | With a bundle loaded from disk, its threshold was applied to the session model's PDs (policy page, report); the export tab saved the loaded bundle under the session experiment's name; Overview used the *selected* rather than the *active* model. | `state.experiment_threshold()`, `state.experiment_bundle()`, active-model key on Overview. | browser run 2 (loaded-bundle path) |
| 10 | Uncertainty | The simulator's reliability statement used the validation split, which had fitted the calibrator. | Uses held-out test applicants. | — |
| 11 | Financial horizons | Interest and funding scaled with the horizon but PD/expected loss and operating cost did not (2-year interest vs 1-year loss); break-even, stress and policy maths shared the problem. | PD treated as annual, `PD_h = 1 − (1 − PD)^h`, opex per loan per year × h, everywhere. Example: 1 loan of 100, PD 10%, LGD 50%, h = 2 → EL 9.5. | `test_two_year_contribution_uses_one_horizon_everywhere`, `test_break_even_pd_and_rate_at_two_years_zero_the_contribution`, `test_stress_base_matches_contribution_at_horizon`, `test_financial_horizon_input_drives_the_calculation` |
| 12 | Loan NPV | EAD at default omitted the missed month's interest, so a default cost money even with LGD = 0 (−282.89 on a 100,000 / 24-month example). | EAD = opening balance + accrued interest. | `test_npv_default_is_costless_with_zero_lgd_no_lag_and_loan_rate_discounting` |
| 13 | Currency formatting | Letter codes ran into numbers (`INR78.84M`, `Rs.78.84M`); chart axes said "currency units" when a currency was set. | Space after codes/abbreviations; axis titles use the chosen currency. | `test_money_formatting_with_currency` |
| 14 | Small datasets | Fewer than 1,000 labelled rows made the Model Lab sample-size input invalid (minimum > maximum). | Floor = min(1,000, rows). | `test_small_dataset_trains_in_model_lab` |
| 15 | Downloads | Each download re-ran the page; in the browser one of 22 downloads was cancelled mid-transfer. | `on_click="ignore"` on download buttons; re-run: 22/22. | browser run 1 |
| 16 | Wording | The fairness table used "approval rate". | `within_threshold_rate` ("share below the PD threshold"). | `test_fairness_reports_within_threshold_not_approval` |
| 17 | Robustness | An all-missing predictor produced imputer warnings; `load_bundle` crashed on a string path; Economics PD input rounding left its EL 0.05% off the Portfolio EL. | Empty predictors dropped with a warning; `Path()` coercion; PD input to 3 decimals (now identical). | `test_all_missing_predictor_is_dropped_with_a_warning` |
| 18 | Explainability | Global contributions were recomputed on every rerun. | Cached per experiment and model. | — |

## Requirement checklist

**PASS** = verified by an executed test or browser run. **FAIL** = verified not to meet the requirement.
**NOT VERIFIED** = could not be checked here.

### Acceptance criteria (§24)

| Requirement | Status | Evidence |
|---|---|---|
| All required project files exist | PASS | Structure matches §18; pages live in `app_pages/` (not `pages/`) so Streamlit's legacy auto-navigation cannot bypass `app.py` — documented deviation. |
| App launches with the documented commands | PASS (Linux) | `streamlit run app.py`; `run_app.bat` on Windows NOT VERIFIED. |
| Synthetic demo dataset works | PASS | Browser run 1; AppTest. |
| Users can upload their own CSV | PASS | Browser run 2. |
| Users can select and validate the target label | PASS | Browser run 2 (text labels; counts match the file); mapping tests. |
| At least two genuinely distinct algorithms | PASS | Logistic regression, random forest, LightGBM (HistGradientBoosting fallback in the 3.11 env). |
| Valid held-out evaluation | PASS | Disjoint splits; calibration and threshold from validation only; leakage guard on training rows. |
| Meaningful evaluation diagnostics | PASS | Model Lab tiles and charts; values recomputed from exports. |
| Score a compatible hypothetical borrower | PASS | Session model and a bundle re-loaded from disk. |
| Expected loss calculated correctly | PASS | Hand examples; portfolio cross-checks. |
| Financial scenario engine with user-controlled inputs | PASS | Unit tests; horizon input AppTest. |
| Portfolio graphs use the actual selected dataset | PASS | Uploaded dataset's own exposure column and counts; filter AppTest. |
| Explanations or a documented fallback | PASS | Exact linear, native LightGBM, SHAP, occlusion fallback (tests with SHAP/LightGBM absent). |
| Model artifacts and reports can be exported | PASS | 22 browser downloads; cross-process bundle reload. |
| Usable when optional features are unavailable | PASS | Full suite incl. all-page AppTests without LightGBM/SHAP. |
| Tests cover major financial and modelling functions | PASS | 124 tests; §21 list covered. |
| Documentation matches the implementation | PASS for the sections checked | README results reproduced; finance, leakage and troubleshooting sections updated. |
| No decorative buttons or menu items | PASS for those exercised | All navigation links, load/confirm/screen/save/train/assess/monitor/save-bundle/load-bundle/save-reports/delete/clear and every download button. Not clicked: *Local file in data/* loader with a real file, *Tune on validation*. |
| No fabricated metrics or findings | PASS | Tiles recomputed from exports; README table reproduced. |
| Limitations and assumptions visible | PASS | Callouts, provenance badges, footer, report sections. |

### Other key requirements

| Requirement | Status | Note |
|---|---|---|
| §5 Real datasets (Home Credit, German Credit, LendingClub) | NOT VERIFIED | Not downloadable in this environment; conversion scripts tested only on mock files. |
| §6 Identifiers / post-outcome fields excluded, preprocessing fit on training rows, test isolated from selection and tuning, chronological split | PASS | Tests; defect 5 fixed. |
| §7 Calibration never uses test outcomes; score vs PD distinguished; boosting fallback | PASS | Defect 6 fixed. |
| §8 Metric set, threshold explorer, validation-based selection and threshold | PASS | Defect 7 fixed. |
| §9 Baseline / fixed / cost-optimised / contribution policies with TP/FP/TN/FN | PASS | Counts recomputed from exports. |
| §10 PD × LGD × EAD, horizon stated, contribution without double counting, NPV, stress | PASS | Defects 11–12 fixed. |
| §11 Simulator adapts to the schema, bands editable, no approve/deny wording | PASS | Assessment outcome "within the selected threshold". |
| §12 Global/local explanations, SHAP when installed, permutation importance, causality caveat | PASS | |
| §13 Dashboard KPIs, filters that change results, Plotly charts | PASS | Filter AppTest. |
| §15 Fairness audit with sensitive attributes excluded from the model by default | PASS | |
| §16 UI quality | PASS (visual review) | Screenshots reviewed; subjective. |
| §17 Exports; report separates measured / observed / assumed / simulated | PASS | PDF export not implemented (optional in the prompt). |
| §19 Windows and macOS | NOT VERIFIED | Only Linux was available. |
| §20 Local-only storage, size limits, sanitised file names, no raw inputs in the audit log, deletion controls | PASS | Tests; deletion AppTest. |
| §21 Edge cases (empty data, missing/one-class/unknown target, inf, all-missing, small data, missing dates, incompatible inputs, absent optional deps) | PASS | Tests. |

## Open limitations

- Windows, macOS and `run_app.bat` were not executed.
- No real dataset was available; all performance figures describe synthetic data.
- The EDA page describes all labelled rows (descriptive only; no modelling decision uses it), so it includes the rows
  later used for testing.
- The model PD is treated as a 12-month PD; for Home Credit's early-instalment target that is an assumption, stated
  in the UI.
- PDF export is not implemented; Markdown and HTML reports are.
- Home Credit's bureau and instalment tables are only used through the optional enrichment script, not in the app.
- Bundles are Python pickles and load only from the app's own `models/` folder; a downloaded bundle must be copied
  there to be re-imported.
