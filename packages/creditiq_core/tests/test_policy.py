"""Policy documents: validation, single decisions and simulation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from creditiq_core.policy.engine import FLAGGED, WITHIN, apply_policy, decide
from creditiq_core.policy.schema import PolicyValidationError, parse_policy
from creditiq_core.policy.simulate import compare, simulate

POLICIES = Path(__file__).resolve().parents[3] / "policies"


def _base(**over):
    doc = {"version": "t1", "name": "Test policy", "pd_threshold": {"default": 0.2},
           "hard_rules": [{"id": "affordability", "type": "ratio_limit", "numerator": "AMT_ANNUITY",
                           "denominator": "AMT_INCOME_TOTAL", "max": 0.4, "reason": "Instalment too high"}]}
    doc.update(over)
    return doc


@pytest.mark.parametrize("path", sorted(POLICIES.glob("*.yaml")), ids=lambda p: p.name)
def test_bundled_policies_validate(path):
    policy = parse_policy(yaml.safe_load(path.read_text()))
    assert policy.version and policy.hard_rules


def test_invalid_policies_name_the_problem():
    cases = [
        (_base(band_cutoffs=[0.2, 0.1]), "band_cutoffs"),
        (_base(pd_threshold={"default": 1.5}), "pd_threshold.default"),
        (_base(hard_rules=[{"id": "x1", "type": "ratio_limit", "numerator": "a", "denominator": "b", "max": -1,
                            "reason": "bad"}]), "hard_rules.0.max"),
        (_base(hard_rules=[{"id": "x1", "type": "teleport", "reason": "bad"}]), "hard_rules.0"),
        (_base(unexpected=True), "unexpected"),
        (_base(band_labels=["a", "b"]), "band_labels"),
    ]
    for doc, where in cases:
        with pytest.raises(PolicyValidationError) as exc:
            parse_policy(doc)
        assert any(where in e for e in exc.value.errors), (where, exc.value.errors)


def test_rules_must_use_model_inputs_and_never_sensitive_attributes():
    with pytest.raises(PolicyValidationError, match="not an input"):
        parse_policy(_base(), model_columns={"AMT_ANNUITY"})
    bad = _base(hard_rules=[{"id": "age_rule", "type": "range", "column": "DAYS_BIRTH", "min": -20000,
                             "reason": "uses age"}])
    with pytest.raises(PolicyValidationError, match="sensitive"):
        parse_policy(bad, model_columns={"DAYS_BIRTH"}, sensitive_columns={"DAYS_BIRTH"})


def test_single_decision_wording_and_rules():
    policy = parse_policy(_base(pd_threshold={"default": "model", "segments": [
        {"column": "NAME_CONTRACT_TYPE", "value": "Revolving loans", "threshold": 0.05}]}))
    ok = {"AMT_ANNUITY": 1000, "AMT_INCOME_TOTAL": 10000, "NAME_CONTRACT_TYPE": "Cash loans"}
    d = decide(policy, ok, 0.08, model_threshold=0.15)
    assert d.outcome == WITHIN and d.threshold == 0.15 and d.threshold_source == "model" and not d.rules_fired
    d = decide(policy, {**ok, "NAME_CONTRACT_TYPE": "Revolving loans"}, 0.08, 0.15)
    assert d.outcome == FLAGGED and d.threshold == 0.05 and "segment" in d.threshold_source
    d = decide(policy, {**ok, "AMT_ANNUITY": 5000}, 0.01, 0.15)
    assert d.outcome == FLAGGED and [r.id for r in d.rules_fired] == ["affordability"]
    assert d.rules_fired[0].value == pytest.approx(0.5) and d.rules_fired[0].limit == 0.4
    assert any("affordability" in r for r in d.reasons())
    assert {d.outcome for d in [d]} <= {WITHIN, FLAGGED}  # never approve/deny


def test_note_rules_never_flag():
    policy = parse_policy(_base(hard_rules=[{"id": "no_bureau", "type": "missing_value", "column": "B",
                                             "action": "note", "reason": "missing bureau"}]))
    d = decide(policy, {"B": None}, 0.01, 0.5)
    assert d.outcome == WITHIN and d.rules_fired[0].id == "no_bureau"


def test_simulation_matches_the_single_decision_path_and_compares():
    rng = np.random.default_rng(0)
    n = 500
    frame = pd.DataFrame({"pd": rng.uniform(0, 0.4, n), "AMT_ANNUITY": rng.uniform(500, 6000, n),
                          "AMT_INCOME_TOTAL": rng.uniform(5000, 20000, n), "AMT_CREDIT": rng.uniform(1e4, 1e6, n)})
    frame["y"] = (rng.uniform(0, 1, n) < frame["pd"]).astype(int)
    loose, strict = parse_policy(_base()), parse_policy(_base(version="t2", pd_threshold={"default": 0.1}))
    vec = apply_policy(loose, frame, frame["pd"].to_numpy(), 0.2)["flagged"].to_numpy()
    single = [decide(loose, r._asdict(), r.pd, 0.2).flagged for r in frame.drop(columns="y").itertuples(index=False)]
    assert vec.tolist() == single
    a = simulate(loose, frame, 0.2, "AMT_CREDIT")
    b = simulate(strict, frame, 0.2, "AMT_CREDIT")
    assert b["flag_rate"] > a["flag_rate"] and sum(a["confusion"].values()) == n
    assert a["expected_loss_12m_within_threshold"] < a["expected_loss_12m_all"]
    delta = compare(b, a)
    assert delta["flag_rate"] == pytest.approx(b["flag_rate"] - a["flag_rate"])
    assert delta["ead_within_threshold"] < 0  # stricter policy keeps less exposure within threshold
