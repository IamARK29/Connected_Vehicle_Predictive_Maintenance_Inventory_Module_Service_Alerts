"""Tests for ModelRegistry and individual model predict_single functions.

These tests gracefully skip if model files are not yet trained (FileNotFoundError).
"""
from __future__ import annotations

import pytest

from models.model_registry import ModelRegistry, _MODEL_SPECS, MODEL_DIR

VALID_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "critical", "warning", "ok", "unknown", "error"}
SAMPLE_VINS = [
    "MH01MZ7X0001",
    "MH01MZ7X0002",
    "MH01MZ7X0003",
    "MH01MZ7X0004",
    "MH01MZ7X0005",
]


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry()


# ── load_all_models ────────────────────────────────────────────────────────────

def test_load_all_models_returns_dict(registry):
    status = registry.load_all_models()
    assert isinstance(status, dict)
    assert set(status.keys()) == set(_MODEL_SPECS.keys())


def test_load_all_models_values_are_bool(registry):
    for name, ok in registry.load_all_models().items():
        assert isinstance(ok, bool), f"Status for {name!r} is not bool: {ok!r}"


def test_model_specs_have_required_keys():
    for name, spec in _MODEL_SPECS.items():
        assert "files"  in spec, f"{name} missing 'files'"
        assert "module" in spec, f"{name} missing 'module'"
        assert isinstance(spec["files"], list)
        assert len(spec["files"]) > 0


# ── get_model_metadata ─────────────────────────────────────────────────────────

def test_get_model_metadata_returns_list(registry):
    meta = registry.get_model_metadata()
    assert isinstance(meta, list)
    assert len(meta) == len(_MODEL_SPECS)


def test_get_model_metadata_has_model_key(registry):
    for row in registry.get_model_metadata():
        assert "model"   in row
        assert "trained" in row
        assert isinstance(row["trained"], bool)


# ── predict_all ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("vin", SAMPLE_VINS)
def test_predict_all_returns_dict(registry, vin):
    results = registry.predict_all(vin)
    assert isinstance(results, dict)
    assert len(results) == len(_MODEL_SPECS)


@pytest.mark.parametrize("vin", SAMPLE_VINS)
def test_predict_all_severity_valid(registry, vin):
    results = registry.predict_all(vin)
    for model_name, output in results.items():
        severity = output.get("severity", "")
        assert severity in VALID_SEVERITIES, (
            f"VIN {vin}, model {model_name}: invalid severity {severity!r}"
        )


@pytest.mark.parametrize("vin", SAMPLE_VINS)
def test_predict_all_has_message(registry, vin):
    results = registry.predict_all(vin)
    for model_name, output in results.items():
        assert "message" in output or "severity" in output, (
            f"VIN {vin}, model {model_name}: output missing both 'message' and 'severity'"
        )


def test_predict_all_unknown_vin_does_not_raise(registry):
    results = registry.predict_all("UNKNOWN_VIN_999")
    assert isinstance(results, dict)
    for output in results.values():
        assert "severity" in output


# ── Individual model predict_single (skipped if not trained) ──────────────────

def _import_model_predict(module_path: str):
    import importlib
    try:
        mod = importlib.import_module(module_path)
        return mod.predict_single
    except FileNotFoundError:
        pytest.skip(f"Model files not trained yet: {module_path}")
    except ImportError as exc:
        pytest.skip(f"Cannot import {module_path}: {exc}")


@pytest.mark.parametrize("model_name,spec", _MODEL_SPECS.items())
def test_individual_predict_single_output_schema(model_name, spec):
    predict_single = _import_model_predict(spec["module"])
    result = predict_single(SAMPLE_VINS[0])
    assert isinstance(result, dict), f"{model_name}.predict_single must return dict"
    # Must have at least one of these keys
    has_known_key = any(k in result for k in (
        "urgency", "severity", "battery_health", "score", "predicted_days",
        "risk_category", "anomaly_score",
    ))
    assert has_known_key, f"{model_name} output missing all known schema keys: {list(result.keys())}"


@pytest.mark.parametrize("model_name,spec", _MODEL_SPECS.items())
def test_individual_predict_single_for_multiple_vins(model_name, spec):
    predict_single = _import_model_predict(spec["module"])
    for vin in SAMPLE_VINS[:3]:
        result = predict_single(vin)
        assert isinstance(result, dict), f"{model_name}.predict_single({vin!r}) returned {type(result)}"


# ── MODEL_DIR ─────────────────────────────────────────────────────────────────

def test_model_dir_constant():
    from pathlib import Path
    assert isinstance(MODEL_DIR, Path)
    assert MODEL_DIR.parts[-2:] == ("models", "saved") or MODEL_DIR.name == "saved"
