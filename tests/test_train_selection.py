"""Tests for selective tuning and the merge into the selections file.

Tuning is split across machines: the LSTM is a long sequential recurrence that
belongs on a GPU, the other two finish locally in minutes. That only works if a
run tuning one model leaves the others' selections intact, so these tests pin
the merge rather than the search.
"""

from __future__ import annotations

import json

import pytest

from src.train import ALL_MODELS, build_parser, train_all


def _args(argv):
    return build_parser().parse_args(argv)


def test_models_flag_defaults_to_every_model() -> None:
    parsed = _args([])
    assert tuple(parsed.models.split(",")) == ALL_MODELS


def test_models_flag_accepts_a_subset() -> None:
    assert _args(["--models", "lstm"]).models == "lstm"


def test_unknown_model_name_is_rejected(monkeypatch, tmp_path) -> None:
    """A typo must fail loudly rather than silently tuning nothing."""

    class FakeConfig:
        pass

    with pytest.raises(ValueError, match="unknown models"):
        train_all(FakeConfig(), models=("lstm", "transfomer"))


def _write_selections(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tuning_one_model_keeps_the_others(monkeypatch, tmp_path) -> None:
    """The property the split across machines depends on."""
    import src.train as train

    selections = tmp_path / "selected_hyperparameters.json"
    _write_selections(
        selections,
        {
            "tuning_area": 5161,
            "harmonic_arima": {"k_daily": 6, "k_weekly": 2, "order": [3, 0, 1]},
            "lightgbm": {"num_leaves": 63},
        },
    )

    _stub_environment(monkeypatch, train, tmp_path, area=5161)
    monkeypatch.setattr(train, "tune_lstm", lambda *a, **k: {"hidden_size": 128, "best_epoch": 11})

    result = train_all(_FakeStudyConfig(tmp_path), models=("lstm",))

    assert result["harmonic_arima"]["order"] == [3, 0, 1]
    assert result["lightgbm"]["num_leaves"] == 63
    assert result["lstm"] == {"hidden_size": 128, "best_epoch": 11}

    on_disk = json.loads(selections.read_text(encoding="utf-8"))
    assert on_disk == result


def test_selections_from_a_different_area_are_refused(monkeypatch, tmp_path) -> None:
    """Mixing areas would silently pair one area's harmonics with another's LSTM."""
    import src.train as train

    selections = tmp_path / "selected_hyperparameters.json"
    _write_selections(selections, {"tuning_area": 9999, "lightgbm": {"num_leaves": 63}})

    _stub_environment(monkeypatch, train, tmp_path, area=5161)

    with pytest.raises(ValueError, match="different areas must not be mixed"):
        train_all(_FakeStudyConfig(tmp_path), models=("lstm",))


# --------------------------------------------------------------------------
# Minimal stand-ins, so the merge is tested without running a search
# --------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload

    def describe(self):
        return self.payload


class _FakeLog:
    def summary(self):
        return "0 experiments logged"


class _FakePaths:
    def __init__(self, root):
        self.tables = root
        self.experiments_csv = root / "experiments.csv"


class _FakeStudyConfig:
    def __init__(self, root):
        self.paths = _FakePaths(root)


def _stub_environment(monkeypatch, train, tmp_path, *, area: int) -> None:
    class FakeAreas:
        top_traffic = area

    class FakeSplits:
        def describe(self):
            return "(splits)"

    monkeypatch.setattr(train.SelectedAreas, "load", staticmethod(lambda path: FakeAreas()))
    monkeypatch.setattr(train, "load_area_series", lambda c, a: (None, None))
    monkeypatch.setattr(train, "make_splits", lambda v, t, c: FakeSplits())
    monkeypatch.setattr(train, "ExperimentLog", lambda path: _FakeLog())


# --------------------------------------------------------------------------
# The stopping point must survive the trip through the selections file
# --------------------------------------------------------------------------


def test_lstm_selection_carries_the_epoch_count(monkeypatch, tmp_path) -> None:
    """``best_epoch`` must reach the file, not just exist during the sweep.

    ``_fit_final`` reads this key to decide how long to train when there is no
    validation set left. If the selection block omits it the final fit silently
    runs the full sweep budget, which is a different model from the one that
    was selected -- and nothing downstream would report the difference.
    """
    import src.train as train

    _stub_environment(monkeypatch, train, tmp_path, area=5161)
    monkeypatch.setattr(train, "tune_lstm", lambda *a, **k: {"hidden_size": 64, "best_epoch": 16})

    result = train_all(_FakeStudyConfig(tmp_path), models=("lstm",))
    assert result["lstm"]["best_epoch"] == 16

    on_disk = json.loads((tmp_path / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    assert on_disk["lstm"]["best_epoch"] == 16


def test_gbm_selection_carries_the_tree_count(monkeypatch, tmp_path) -> None:
    """Same contract for LightGBM: without it the refit uses all 2000 trees."""
    import src.train as train

    _stub_environment(monkeypatch, train, tmp_path, area=5161)
    monkeypatch.setattr(
        train, "tune_gbm", lambda *a, **k: {"num_leaves": 68, "best_iteration": 412}
    )

    result = train_all(_FakeStudyConfig(tmp_path), models=("lightgbm",))
    assert result["lightgbm"]["best_iteration"] == 412


def test_committed_selections_carry_their_stopping_points() -> None:
    """Guards the real artefact, not a stand-in for it."""
    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "results" / "tables" / "selected_hyperparameters.json"
    if not path.exists():
        pytest.skip("selected_hyperparameters.json not present")
    selected = json.loads(path.read_text(encoding="utf-8"))

    if "lightgbm" in selected:
        assert selected["lightgbm"].get("best_iteration", 0) > 0, (
            "LightGBM selection has no best_iteration; the final fit would train "
            "the full n_estimators ceiling. Re-run `run.py train --models lightgbm`."
        )
    if "lstm" in selected:
        assert selected["lstm"].get("best_epoch", -1) >= 0, (
            "LSTM selection has no best_epoch; the final fit would run the full "
            "sweep budget. Re-run the Kaggle LSTM notebook."
        )
