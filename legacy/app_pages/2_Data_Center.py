"""Page 2 — Data Center: load, profile, map the target, choose features, screen for leakage."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import data_loader as dl
from src import data_validation as dv
from src import state, ui
from src.persistence import to_json

ss = st.session_state
ui.page_header("Data Center", "Load data, confirm what “default” means for this dataset, and decide which columns the model may use.")

tab_load, tab_profile, tab_target, tab_features, tab_report = ui.tabs(
    ["Load data", "Profile & quality", "Target mapping", "Features & leakage", "Quality report"], key="tabs_data_center")


def _profile():
    """Profile cached per dataset fingerprint."""
    cache = ss.setdefault("_profile_cache", {})
    fp = ss.ds_info.fingerprint
    if fp not in cache:
        cache.clear()
        summary, prof = dv.profile_dataframe(ss.df)
        cache[fp] = (summary, prof, dv.find_issues(ss.df, prof), dv.outlier_review(ss.df))
    return cache[fp]


# ---------------------------------------------------------------- Load
with tab_load:
    source = st.radio("Data source", ["Synthetic demo dataset", "Upload a CSV", "Local file in data/"], horizontal=True)
    if source == "Synthetic demo dataset":
        ui.note("Fabricated applications whose columns mimic Home Credit <span class='ciq-code'>application_train.csv</span>. Two synthetic extras: "
                "<span class='ciq-code'>APPLICATION_DATE</span> (enables time-based splits) and <span class='ciq-code'>POST_LOAN_COLLECTION_CALLS</span> "
                "(deliberately leaky, to demonstrate the leakage screen).")
        c1, c2, c3 = st.columns([1, 1, 2])
        n_rows = c1.number_input("Rows", min_value=1_000, max_value=300_000, value=20_000, step=5_000)
        seed = c2.number_input("Random seed", min_value=0, max_value=1_000_000, value=config.DEFAULT_SEED)
        if st.button("Load synthetic dataset", type="primary"):
            with st.spinner("Generating synthetic applications"):
                df, info = dl.load_synthetic_dataset(int(n_rows), int(seed))
            state.set_dataset(df, info)
            ui.flash(f"Loaded {len(df):,} synthetic rows. Next: confirm the target in the Target mapping tab.")
            st.rerun()
    elif source == "Upload a CSV":
        ui.note(f"Files are read into memory for this session only and are not written to disk. Limit: {config.MAX_UPLOAD_MB} MB. "
                "For the full Home Credit file, the Local file option is faster.")
        up = st.file_uploader("CSV file", type=["csv"])
        sample = st.number_input("Random sample of rows (0 = load all)", min_value=0, value=0, step=10_000)
        if up is not None and st.button("Load uploaded file", type="primary"):
            if up.size > config.MAX_UPLOAD_MB * 1e6:
                st.error(f"File is {up.size / 1e6:.0f} MB — above the {config.MAX_UPLOAD_MB} MB limit.")
            else:
                try:
                    with st.spinner("Reading CSV"):
                        df, info = dl.read_csv_file(up.getvalue(), name=up.name, sample_rows=int(sample) or None,
                                                    source_kind="upload")
                    state.set_dataset(df, info)
                    ui.flash(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {info.name}.")
                    st.rerun()
                except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
                    st.error(f"Could not read the file: {exc}")
    else:
        files = dl.list_local_csv_files()
        if not files:
            ui.empty_state("No CSV files found", f"Copy a CSV (for example Home Credit <span class='ciq-code'>application_train.csv</span>) into "
                           f"<span class='ciq-code'>{config.DATA_DIR / 'raw'}</span> and refresh this page.")
        else:
            choice = ui.pick(st.selectbox, "File", {p: f"{p.parent.name}/{p.name}  ({p.stat().st_size / 1e6:,.1f} MB)" for p in files})
            default_sample = 100_000 if choice.stat().st_size > 50e6 else 0
            sample = st.number_input("Random sample of rows (0 = load all)", min_value=0, value=default_sample, step=10_000,
                                     help="Sampling streams the file in chunks, so the full table is never held in memory.")
            if st.button("Load local file", type="primary"):
                try:
                    with st.spinner(f"Reading {choice.name}"):
                        df, info = dl.read_csv_file(choice, sample_rows=int(sample) or None, source_kind="local")
                    state.set_dataset(df, info)
                    ui.flash(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {info.name}.")
                    st.rerun()
                except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                    st.error(f"Could not read the file: {exc}")

    if state.has_data():
        info = ss.ds_info
        st.divider()
        st.subheader("Current dataset")
        ui.tiles([
            {"label": "Rows loaded", "value": ui.num(info.n_rows_loaded), "prov": "synthetic" if info.is_synthetic else "observed",
             "sub": f"of {info.n_rows_file:,} in file" if info.n_rows_file and info.sampled else None},
            {"label": "Columns", "value": ui.num(info.n_cols), "prov": "observed"},
            {"label": "Source", "value": info.source.capitalize(), "sub": info.name},
        ])
        for n in info.notes:
            ui.note(n)
        st.dataframe(ss.df.head(50), height=280)
        if st.button("Remove dataset from this session", icon=":material/delete:"):
            state.set_dataset(None, None)
            st.rerun()

# ---------------------------------------------------------------- Profile
with tab_profile:
    if not state.has_data():
        ui.empty_state("Nothing to profile", "Load a dataset in the first tab.")
    else:
        summary, prof, issues, outliers = _profile()
        ui.tiles([
            {"label": "Rows", "value": ui.num(summary["rows"]), "prov": "observed"},
            {"label": "Numeric / categorical", "value": f"{summary['numeric_columns']} / {summary['categorical_columns']}", "prov": "observed"},
            {"label": "Missing cells", "value": ui.pct(summary["missing_cells_pct"]), "prov": "observed"},
            {"label": "Duplicate rows", "value": ui.num(summary["duplicate_rows"]), "prov": "observed",
             "help": "Exact duplicates can appear in both training and test splits and inflate measured performance."},
            {"label": "Memory in session", "value": f"{summary['memory_mb']:,.1f} MB"},
        ])
        st.subheader("Data-quality findings")
        if issues:
            sev_order = {"error": 0, "warning": 1, "info": 2}
            idf = pd.DataFrame([i.to_dict() for i in issues]).sort_values("severity", key=lambda s: s.map(sev_order))
            ui.show_table(idf, height=min(38 * len(idf) + 40, 360))
        else:
            ui.callout("No issues detected by the automated checks.", "ok")

        st.subheader("Missing values")
        miss = prof[prof["missing_pct"] > 0].sort_values("missing_pct", ascending=False).head(30)
        if miss.empty:
            ui.note("No missing values.")
        else:
            fig = go.Figure(go.Bar(x=miss["missing_pct"], y=miss["column"], orientation="h", marker_color="#4E7FA8",
                                   hovertemplate="%{y}: %{x:.1%} missing<extra></extra>"))
            fig.update_layout(title="Share of missing values (top 30 columns)", xaxis_tickformat=".0%",
                              yaxis=dict(autorange="reversed"))
            ui.plot(fig, height=max(260, 22 * len(miss) + 80))

        st.subheader("Column profile")
        ui.show_table(prof.drop(columns=["unique_ratio"]), pct_cols=["missing_pct"], dec_cols=["min", "median", "max"], height=380)

        st.subheader("Outliers and placeholder values")
        if outliers.empty:
            ui.note("No extreme values (beyond 3 × IQR) or known placeholder codes found.")
        else:
            ui.note("Extreme values are reported, not removed. Tree models are robust to them; the logistic model sees "
                    "standardised values, so very large placeholders (e.g. 365243) are converted to missing by feature engineering.")
            ui.show_table(outliers, pct_cols=["extreme_pct"], dec_cols=["p1", "p99"])

# ---------------------------------------------------------------- Target
with tab_target:
    if not state.has_data():
        ui.empty_state("No dataset", "Load a dataset first.")
    else:
        df = ss.df
        cols = list(df.columns)
        suggested = dv.suggest_target_column(df)
        current = ss.target.column if state.has_target() else suggested
        target_col = st.selectbox("Outcome (target) column", cols, index=cols.index(current) if current in cols else 0)
        counts = df[target_col].value_counts(dropna=False).head(40).rename_axis("value").reset_index(name="rows")
        counts["share"] = counts["rows"] / len(df)
        counts["value"] = counts["value"].astype(str)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**Values in this column**")
            ui.show_table(counts, pct_cols=["share"], int_cols=["rows"], height=min(36 * len(counts) + 40, 320))
        n_unique = df[target_col].nunique()
        with c2:
            if n_unique > 30 and pd.api.types.is_numeric_dtype(df[target_col]):
                ui.callout(f"<b>{target_col}</b> has {n_unique:,} distinct numeric values — it looks continuous, not an outcome label.", "warn")
            values = list(df[target_col].dropna().unique())[:200]
            sug_pos, why = dv.suggest_positive_values(df[target_col])
            if state.has_target() and ss.target.column == target_col:
                sug_pos = ss.target.positive_values
            ui.note(why)
            pos = st.multiselect("Labels that mean DEFAULT (positive class)", values,
                                 default=[v for v in sug_pos if v in values], format_func=str)
            neg_default = dv.suggest_negative_values(df[target_col], pos)
            neg = st.multiselect("Labels that mean NON-DEFAULT", [v for v in values if v not in pos],
                                 default=[v for v in neg_default if v in values], format_func=str,
                                 help="Labels in neither list (e.g. loans still outstanding) are excluded from modelling.")
            definition = st.text_area("Target definition (documented with the experiment)",
                                      value=(ss.target.definition if state.has_target() and ss.target.column == target_col
                                             else dl.suggest_target_definition(df, target_col, ss.ds_info.is_synthetic)),
                                      height=110)
            confirm = st.checkbox("I have checked the dataset documentation: the labels above correctly identify default.")
            if st.button("Confirm target mapping", type="primary", disabled=not confirm):
                try:
                    mapping = dv.build_target_mapping(df, target_col, pos, neg, definition)
                    state.set_target(mapping)
                    msg = (f"Target confirmed: {mapping.n_positive:,} defaults and {mapping.n_negative:,} non-defaults "
                           f"({ui.pct(mapping.prevalence)} default rate). Next: review features in the Features & leakage tab.")
                    ui.flash(msg, "warning" if mapping.warnings else "success")
                    st.rerun()
                except dv.TargetMappingError as exc:
                    st.error(str(exc))
        if state.has_target():
            t = ss.target
            ui.callout(f"Confirmed: <b>{t.column}</b> ∈ {[str(v) for v in t.positive_values]} = default. "
                       f"{t.n_positive:,} defaults, {t.n_negative:,} non-defaults, {t.n_unmapped:,} rows excluded as unmapped.", "ok")
            for w in t.warnings:
                st.warning(w)
            ui.note("Different datasets describe different populations, products and default definitions. A model trained on one "
                    "should not be assumed to transfer to another.")

# ---------------------------------------------------------------- Features
with tab_features:
    if not state.has_target():
        ui.empty_state("Confirm the target first", "Feature roles depend on which column is the target.")
    else:
        df = ss.df
        date_options = [None] + dv.detect_date_columns(df)
        saved = state.has_features()
        date_default = ss.date_col if saved and ss.date_col in date_options else (date_options[1] if len(date_options) > 1 else None)
        num_cols = [c for c in df.columns if dv.infer_kind(df[c]) == dv.KIND_NUMERIC and c != ss.target.column]
        exp_sugg = dv.suggest_exposure_columns(df)
        exp_options = [None] + num_cols
        exp_default = ss.exposure_col if saved and ss.exposure_col in exp_options else (exp_sugg[0] if exp_sugg else None)
        c1, c2 = st.columns(2)
        date_col = ui.pick(c1.selectbox, "Application date column (optional)",
                           {v: "None" if v is None else v for v in date_options}, date_default,
                           help="Enables chronological train/validation/test splits and time trends.")
        exposure_col = ui.pick(c2.selectbox, "Exposure / loan amount column (used as EAD)",
                               {v: "None — use a fixed exposure" if v is None else v for v in exp_options}, exp_default)

        key = (ss.ds_info.fingerprint, ss.target.column, date_col)
        if ss.get("_roles_key") != key:
            _, prof, _, _ = _profile()
            ss._roles_working = dv.suggest_feature_roles(df, ss.target.column, date_col, prof)
            ss._roles_key = key
            ss._roles_ver = ss.get("_roles_ver", 0) + 1
        roles = ss._roles_working

        st.markdown("**Leakage screen**")
        ui.note("Advisory screen: each candidate is scored alone against the target on all labelled rows. A single column "
                "separating defaulters almost perfectly (AUC ≥ 0.90) usually contains information recorded after the outcome. "
                "The binding check runs again at training time on the <b>training split only</b> (the Model Lab's leakage guard), "
                "so held-out rows never influence which predictors are kept.")
        if st.button("Run leakage screen", icon=":material/search:"):
            y = dv.apply_target_mapping(df[ss.target.column], ss.target.positive_values, ss.target.negative_values)
            cand = roles[roles["role"].isin(["predictor", "review", "leakage", "sensitive"])]["column"].tolist()
            with st.spinner("Scoring each column against the target"):
                ss.leakage_table = dv.leakage_scan(df, y, cand)
        if ss.leakage_table is not None:
            lt = ss.leakage_table
            flagged = lt[lt["assessment"].str.startswith("SUSPICIOUS")]["column"].tolist()
            sus = [c for c in flagged if bool(roles.loc[roles["column"] == c, "use"].any())]
            if flagged and not sus:
                ui.callout(f"Flagged by the screen and already excluded: <b>{', '.join(flagged)}</b>.", "ok")
            if sus:
                ui.callout(f"Possible leakage: <b>{', '.join(sus)}</b>. Confirm these are known at application time, or exclude them.", "risk")
                if st.button("Exclude suspicious columns"):
                    roles.loc[roles["column"].isin(sus), ["use", "role", "reason"]] = [False, "leakage", "Leakage screen: single-feature AUC ≥ 0.90"]
                    ss._roles_ver += 1
                    st.rerun()
            with st.expander("Leakage screen results", expanded=bool(sus)):
                ui.show_table(lt, dec_cols=["single_feature_auc"], height=300)

        st.markdown("**Predictor selection**")
        ui.note("Suggested roles: identifiers, constant/empty columns, raw dates, known post-outcome fields and potentially "
                "protected attributes are unticked. You can change any choice; changes to excluded roles need a justification.")
        view = roles[["use", "column", "role", "reason", "kind", "missing_pct", "unique"]].copy()
        view["missing_pct"] = view["missing_pct"] * 100
        edited = st.data_editor(
            view,
            key=f"roles_editor_{ss._roles_ver}", hide_index=True, height=420,
            disabled=["column", "role", "reason", "kind", "missing_pct", "unique"],
            column_config={"use": st.column_config.CheckboxColumn("Use", help="Include as a model input"),
                           "column": st.column_config.TextColumn("Column"),
                           "role": st.column_config.TextColumn("Role"),
                           "reason": st.column_config.TextColumn("Reason", width="large"),
                           "kind": st.column_config.TextColumn("Type"),
                           "missing_pct": st.column_config.NumberColumn("Missing %", format="%.1f"),
                           "unique": st.column_config.NumberColumn("Distinct values", format="%d")},
        )
        target_col = ss.target.column
        chosen = [c for c in edited.loc[edited["use"], "column"] if c != target_col]
        risky = edited[edited["use"] & edited["role"].isin(["leakage", "identifier", "sensitive", "target"])]
        sens_default = [c for c in roles.loc[roles["role"] == "sensitive", "column"]]
        sensitive = st.multiselect("Attributes kept for the fairness audit (not used for decisions unless ticked above)",
                                   [c for c in df.columns if c != target_col],
                                   default=[c for c in (ss.sensitive_cols or sens_default) if c in df.columns])
        ack = True
        if len(risky):
            ui.callout(f"You included column(s) flagged as {', '.join(sorted(set(risky['role'])))}: <b>{', '.join(risky['column'])}</b>.", "warn")
            ack = st.checkbox("I have a documented analytical and legal justification for including these columns.")
        sens_in = [c for c in sensitive if c in chosen]
        if sens_in:
            st.warning(f"Audit attributes also used as predictors: {', '.join(sens_in)}. The model can then treat applicants differently "
                       "by these attributes directly.")
        st.caption(f"{len(chosen)} predictors selected.")
        if st.button("Save feature configuration", type="primary", disabled=not chosen or not ack):
            excluded = {r.column: (r.role if r.role != "predictor" and r.role != "review" else "user_excluded")
                        for r in edited.itertuples() if not r.use and r.column != target_col}
            new_roles = roles.copy()
            new_roles["use"] = edited["use"].to_numpy()
            state.set_features(chosen, excluded, date_col, exposure_col, sensitive, new_roles)
            ss._roles_working = new_roles
            ui.flash(f"Saved {len(chosen)} predictors. Next: train models in the AI Model Lab.")
            st.rerun()
        if state.has_features():
            st.page_link("app_pages/4_Model_Lab.py", label="Continue to the AI Model Lab", icon=":material/arrow_forward:")

# ---------------------------------------------------------------- Report
with tab_report:
    if not state.has_data():
        ui.empty_state("No dataset", "Load a dataset to generate its quality report.")
    else:
        summary, prof, issues, outliers = _profile()
        roles = ss.roles if ss.roles is not None else ss.get("_roles_working")
        md = dv.quality_report(ss.ds_info.to_dict(), summary, prof, issues,
                               ss.target.to_dict() if state.has_target() else None, roles)
        ui.note("Generated only from computed checks on the loaded data.")
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.download("Quality report (Markdown)", md, "data_quality_report.md", "text/markdown")
        with c2:
            payload = {"dataset": ss.ds_info.to_dict(), "summary": summary, "issues": [i.to_dict() for i in issues],
                       "target": ss.target.to_dict() if state.has_target() else None,
                       "excluded": ss.excluded, "predictors": ss.predictors,
                       "leakage_screen": ss.leakage_table.to_dict("records") if ss.leakage_table is not None else None}
            ui.download("Findings (JSON)", to_json(payload, indent=2), "data_quality.json", "application/json")
        with c3:
            ui.download("Column profile (CSV)", prof.to_csv(index=False), "column_profile.csv", "text/csv")
        with st.expander("Preview"):
            st.markdown(md)

