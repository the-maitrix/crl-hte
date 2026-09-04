import itertools

import pandas as pd

from scripts import export_synthetic_paper_panels as panels
from scripts import export_synthetic_paper_tables as tables


def frame(learners, axis=None, axis_values=(0.0,)):
    rows = []
    for method, n, learner, y_type, seed, axis_value in itertools.product(
        panels.METHODS,
        panels.EXPECTED_N,
        learners,
        panels.Y_TYPES,
        range(1000, 1010),
        axis_values,
    ):
        row = {
            "method": method,
            "n": n,
            "learner": learner,
            "y_type": y_type,
            "trial_seed": seed,
            "phi_dim": 10,
            "pehe_norm": 1 + n / 10000 + axis_value / 10,
            "policy_norm_20": 0.5 - axis_value / 10,
        }
        if axis:
            row[axis] = axis_value
        rows.append(row)
    return pd.DataFrame(rows)


def test_synthetic_table_export(tmp_path):
    data = frame(tables.LEARNERS)
    assert tables.validate(data, 10) == panels.EXPECTED_N
    raw_comparison = data[
        (data.method == "raw_x") & (data.y_type == "h_y")
    ].copy()
    raw_comparison["method"] = "surrogate_index"
    data = tables.add_raw_x_s(data, raw_comparison, 10)
    for learner, (_, learner_slug) in tables.LEARNERS.items():
        for metric, (_, metric_slug, _) in tables.METRICS.items():
            output = tmp_path / f"papertable_{metric_slug}_phi10_{learner_slug}.tex"
            tables.render(data, learner, metric, output, 10)
            rendered = output.read_text()
            assert "10 paired data resamples" in rendered
            assert r"Raw $X$, $h(X,S)$" in rendered
            assert "---" in rendered
    assert len(list(tmp_path.glob("*.tex"))) == 4


def test_synthetic_panel_export(tmp_path):
    main = frame(panels.LEARNERS)
    alpha = frame(panels.LEARNERS, "alpha", (0.0, 0.25, 0.5, 0.75, 1.0))
    delta = frame(panels.LEARNERS, "delta", (0.0, 0.5, 1.0, 2.0))
    panels.validate(main, 10)
    panels.validate(alpha, 10, "alpha")
    panels.validate(delta, 10, "delta")
    for name in ("main", "alpha-sweep", "delta-sweep"):
        (tmp_path / name).mkdir()
    panels.plot_main_panels(main, tmp_path / "main")
    for learner in panels.LEARNERS:
        panels.plot_violation_rows(
            alpha, "alpha", tmp_path / "alpha-sweep", [50, 250, 750], learner
        )
        panels.plot_violation_rows(
            delta, "delta", tmp_path / "delta-sweep", [50, 250, 750], learner
        )
    assert len(list(tmp_path.rglob("*.pdf"))) == 24
