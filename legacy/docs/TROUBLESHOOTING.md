# Requirements and troubleshooting

## Python and packages

| | |
|---|---|
| Python | 3.11 or 3.12 recommended. Tested on 3.11 and 3.13. 3.10 is untested (pip would install pandas 2.x there, which the code supports). |
| Core (`requirements.txt`) | streamlit ≥ 1.46, pandas ≥ 2.2, numpy ≥ 2.0, scikit-learn ≥ 1.4, plotly ≥ 5.20, joblib, jinja2, pytest |
| Optional (`requirements-optional.txt`) | lightgbm (boosting backend), shap (TreeSHAP for the random forest) |
| Tested combinations | (a) `requirements-lock.txt`: streamlit 1.66.1, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, plotly 7.1.0, lightgbm 4.7.0, shap 0.54.1 on Linux / Python 3.13. (b) Minimum versions, no optional extras: streamlit 1.46.1, pandas 2.2.3, numpy 2.0.2, scikit-learn 1.4.2, plotly 5.24.1 on Linux / Python 3.11. The full test suite passed on both. Windows and macOS were not tested directly. |
| Hardware | Any recent laptop; 8 GB RAM is comfortable for a 100k-row sample of Home Credit. |

Internet is needed only to install packages and download datasets. The app itself runs offline: fonts are bundled
and Streamlit usage statistics are disabled.

## Installation problems

| Symptom | Fix |
|---|---|
| `'py' is not recognized` (Windows) | Install Python from python.org with the *py launcher* option, or use `python -m venv .venv`. |
| `running scripts is disabled on this system` when activating in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and `.venv\Scripts\activate.bat`. |
| `error: Microsoft Visual C++ 14.0 is required` | A package is building from source because no wheel exists for your Python. Use Python 3.11/3.12 (wheels exist), then reinstall. |
| LightGBM fails to install (especially on Apple Silicon or unusual platforms) | Skip it — the app uses scikit-learn's HistGradientBoosting and labels it as the fallback. On macOS you can also `brew install libomp`, then retry. |
| `OSError: libgomp.so.1` / `libomp.dylib` when importing LightGBM | Install the OpenMP runtime (`sudo apt install libgomp1` on Debian/Ubuntu, `brew install libomp` on macOS) or uninstall lightgbm to use the fallback. |
| SHAP fails to install or import | Uninstall it (`pip uninstall shap`). Logistic-regression and LightGBM explanations don't need it; the random forest falls back to occlusion. |
| Version conflicts after a partial install | Delete `.venv`, recreate it, and `pip install -r requirements-lock.txt`. |

## Running the app

| Symptom | Fix |
|---|---|
| The terminal shows `Email:` on first launch | Older Streamlit versions ask once; press Enter to skip. |
| `streamlit: command not found` | Activate the virtual environment, or run `python -m streamlit run app.py`. |
| Browser does not open | Open http://localhost:8501 manually. If the port is busy: `streamlit run app.py --server.port 8502`. |
| Fonts look different from the screenshots | The bundled fonts load from `static/`; keep `server.enableStaticServing = true` in `.streamlit/config.toml`. System fonts are used otherwise — nothing breaks. |
| `ModuleNotFoundError: No module named 'src'` | Run Streamlit from the project folder (the one containing `app.py`). |
| Upload rejected as too large | Copy the CSV to `data/raw/` and use *Local file in data/*, or raise `server.maxUploadSize` and `CREDITIQ_MAX_UPLOAD_MB`. |
| Training is slow or memory runs out | Lower "Maximum labelled rows used" in the Model Lab, sample rows when loading, untick the random forest, or turn off tuning. |
| "The validation split contains only one class" | Too few defaults for the split. Use a larger sample or the stratified split. |
| A page says it needs a training session | Model bundles loaded from disk contain no borrower rows. Pages that analyse held-out applicants need a model trained in the current session. |
| Results disappeared | Session state lives in the browser tab. Refreshing or opening a new tab starts a new session; experiment metadata and saved bundles remain on disk. |
| Could not load a saved bundle | Bundles are tied to the package versions that created them. Retrain with your current environment. |
| A downloaded bundle does not appear in the Model Lab | For safety the app only loads bundles from its own `models/` folder (they are Python pickles). Copy the `.joblib` file into `models/` and reopen the Model Lab. |
| A CSV loads as a single column | The separator is detected from the first lines (comma, semicolon, tab or pipe). If detection fails, re-export as comma-separated UTF-8. |
| "This does not look like a text CSV file" | The file contains binary data — usually an Excel workbook renamed to `.csv`, or a UTF-16 export. Save it as *CSV UTF-8*. |
| "The active model is not calibrated" | Calibration was set to *none*, so outputs are scores, not probabilities; money figures built on them are distorted. Retrain with Platt or isotonic calibration. |
| A predictor I selected is missing from the trained model | The training-split leakage guard removed it (single-feature AUC ≥ 0.90 on training rows), or it had no values in the training rows. See *Training details* in the Model Lab; untick *Leakage guard* only with a documented reason. |

## Tests

```bash
python -m pytest            # whole suite, about 40 seconds
python -m pytest -k finance # a subset
```

Tests write to a temporary folder, never to your `storage/`, `models/` or `reports/`.
