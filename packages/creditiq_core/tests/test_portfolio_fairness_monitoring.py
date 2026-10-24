import numpy as np
import pandas as pd
import pytest

from creditiq_core import fairness as fair
from creditiq_core import monitoring as mon
from creditiq_core.data import synthetic as synth
from creditiq_core.finance import portfolio as pa


def test_risk_bands_and_invalid_cutoffs():
    bands = pa.assign_risk_band([0.01, 0.05, 0.12, 0.5], (0.05, 0.10, 0.20))
    assert bands.astype(str).tolist() == ["Low", "Moderate", "Elevated", "High"]
    with pytest.raises(ValueError):
        pa.assign_risk_band([0.1], (0.2, 0.1, 0.3))


def test_portfolio_keeps_observed_predicted_and_el_separate():
    frame = pd.DataFrame({"AMT_CREDIT": [100.0, 200.0, np.nan, 400.0], "seg": list("aabb")})
    port = pa.build_portfolio(frame, [0, 1, 0, 0], [0.1, 0.3, 0.05, 0.2], "AMT_CREDIT", lgd=0.5)
    k = pa.portfolio_kpis(port)
    assert k["observed_default_rate"] == pytest.approx(0.25)
    assert k["avg_pd"] == pytest.approx(0.1625)
    assert port["ead"].iloc[2] == pytest.approx(200.0)  # missing exposure → median, flagged
    assert bool(port["exposure_imputed"].iloc[2])
    assert k["total_el"] == pytest.approx(sum(p * 0.5 * e for p, e in zip(port["pd"], port["ead"])))
    seg = pa.segment_summary(port, "seg")
    assert seg["expected_loss"].sum() == pytest.approx(k["total_el"])
    conc = pa.concentration_metrics(seg)
    assert 0.5 <= conc["hhi"] <= 1


def test_loss_concentration_curve_bounds():
    port = pa.build_portfolio(pd.DataFrame({"x": range(100)}), np.zeros(100), np.linspace(0.01, 0.5, 100), None, 0.45, default_exposure=1000)
    c = pa.loss_concentration_curve(port)
    assert c["share_of_expected_loss"].iloc[0] == 0 and c["share_of_expected_loss"].iloc[-1] == pytest.approx(1)
    assert (np.diff(c["share_of_expected_loss"]) >= -1e-12).all()


def test_psi_stable_vs_shifted():
    base = synth.generate_synthetic_credit_data(4000, seed=11)
    same = synth.generate_synthetic_credit_data(4000, seed=12)
    shifted = synth.generate_synthetic_credit_data(4000, seed=11, shifted=True)
    cols = ["EXT_SOURCE_3", "AMT_INCOME_TOTAL", "NAME_CONTRACT_TYPE"]
    ref = mon.build_reference(base[cols])
    stable = mon.feature_drift(ref, same).set_index("column")
    moved = mon.feature_drift(ref, shifted).set_index("column")
    assert stable.loc["EXT_SOURCE_3", "psi"] < 0.1
    assert moved.loc["EXT_SOURCE_3", "psi"] > stable.loc["EXT_SOURCE_3", "psi"]
    assert moved.loc["EXT_SOURCE_3", "missing_change"] > 0.1
    assert mon.psi_from_props([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    schema = mon.schema_check([*cols, "MISSING_COL"], same, ref)
    assert schema["missing_columns"] == ["MISSING_COL"]


def test_group_metrics_and_disparities():
    rng = np.random.default_rng(0)
    g = np.array(["A"] * 500 + ["B"] * 500 + ["C"] * 10)
    p = np.r_[rng.uniform(0, 0.3, 500), rng.uniform(0.1, 0.5, 500), rng.uniform(0, 1, 10)]
    y = (rng.uniform(0, 1, 1010) < p).astype(int)
    t = fair.group_metrics(y, p, 0.2, pd.Series(g)).set_index("group")
    assert bool(t.loc["C", "small_sample"]) and not bool(t.loc["A", "small_sample"])
    assert t.loc["B", "flag_rate"] > t.loc["A", "flag_rate"]
    d = fair.disparity_summary(t.reset_index())
    assert "C" not in set(d["highest_group"]) | set(d["lowest_group"])


def test_age_groups_and_proxy_scan(synthetic_df):
    groups = fair.make_groups(synthetic_df["DAYS_BIRTH"], "DAYS_BIRTH")
    assert set(groups) <= {"<25", "25–34", "35–44", "45–54", "55–64", "65+", "Missing"}
    gender = fair.make_groups(synthetic_df["CODE_GENDER"], "CODE_GENDER")
    scan = fair.proxy_scan(synthetic_df, gender, ["OCCUPATION_TYPE", "AMT_INCOME_TOTAL", "CNT_CHILDREN"])
    assert scan.iloc[0]["feature"] in ("OCCUPATION_TYPE", "AMT_INCOME_TOTAL")
