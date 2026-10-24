import json

from src import persistence as ps
from src import reporting as rp
from src.portfolio_analysis import band_summary, build_portfolio, portfolio_kpis


def _ctx(experiment):
    port = build_portfolio(experiment.frame_split("test"), experiment.y_split("test"), experiment.pd_split("test"),
                           "AMT_CREDIT", 0.45)
    return {"dataset": {"name": "t.csv", "is_synthetic": True, "source": "synthetic", "notes": []},
            "target": experiment.target, "experiment": experiment, "assumptions": {"LGD": 0.45},
            "portfolio_kpis": portfolio_kpis(port), "band_table": band_summary(port)}


def test_report_without_results_has_no_fabricated_sections():
    md = rp.to_markdown(rp.build_report({}))
    assert "Model comparison" not in md and "Portfolio expected loss" not in md
    assert "Limitations" in md


def test_report_with_results(experiment):
    sections = rp.build_report(_ctx(experiment))
    kinds = {s.kind for s in sections}
    assert {"MEASURED", "OBSERVED", "ASSUMPTION", "SIMULATED"} <= kinds
    md = rp.to_markdown(sections)
    auc = experiment.metrics[experiment.selected]["test"]["roc_auc"]
    assert f"{auc:.4g}" in md
    assert "SYNTHETIC" in md and "placeholder" not in md.lower() and "TODO" not in md
    html = rp.to_html(sections)
    assert html.startswith("<!doctype html>") and "<script" not in html


def test_model_card_uses_real_metadata(experiment):
    card = rp.model_card_markdown(experiment.metadata(), experiment.comparison_table())
    assert experiment.experiment_id in card and "Out-of-scope" in card
    assert f"{experiment.metrics[experiment.selected]['test']['roc_auc']:.4g}" in card


def test_experiment_and_audit_records(experiment, tmp_path):
    db = tmp_path / "t.db"
    ps.save_experiment(experiment, db_path=db)
    hist = ps.list_experiments(db)
    assert hist["id"].tolist() == [experiment.experiment_id]
    json.loads(hist["test_metrics"].iloc[0])
    inputs = {"AMT_INCOME_TOTAL": 123456.0, "NAME_CONTRACT_TYPE": "Cash loans"}
    ps.log_assessment({"experiment_id": "x", "model": "logreg", "pd": 0.1, "input_hash": ps.hash_inputs(inputs)}, db)
    rows = ps.list_assessments(db_path=db)
    assert len(rows) == 1 and "123456" not in rows.to_csv()  # raw inputs are never stored
    ps.log_monitoring_run({"experiment_id": "x", "model": "logreg", "alerts": ["a"]}, db)
    assert len(ps.list_monitoring_runs("x", db)) == 1
    removed = ps.clear_local_records(experiments=True, assessments=True, monitoring=True, db_path=db)
    assert removed == {"experiments": 1, "assessments": 1, "monitoring_runs": 1}


def test_json_is_always_valid():
    text = ps.to_json({"a": float("nan"), "b": [float("inf"), 1.0]})
    assert json.loads(text) == {"a": None, "b": [None, 1.0]}


def test_save_report_sanitises_name(tmp_path):
    p = ps.save_report("# hi", "../../evil name.md", tmp_path)
    assert p.parent == tmp_path and p.name == "evil_name.md"
