"""Тесты конфигурации: словарь, YAML, валидация."""
import pytest

from drift_guardian import DriftConfig, Thresholds


def test_dict_roundtrip_restores_nested_thresholds():
    config = DriftConfig(thresholds=Thresholds(psi_warning=0.05), bonferroni=False)
    restored = DriftConfig.from_dict(config.to_dict())
    assert restored == config


def test_yaml_roundtrip(tmp_path):
    config = DriftConfig(
        thresholds=Thresholds(psi_critical=0.3),
        target_column="y",
        exclude_columns=["id"],
        value_bounds={"age": (18, 90)},
        allowed_categories={"region": ["a", "b"]},
    )
    path = config.to_yaml(tmp_path / "config.yaml")
    loaded = DriftConfig.from_yaml(path)
    assert loaded.target_column == "y"
    assert loaded.value_bounds == {"age": (18.0, 90.0)}
    assert loaded.allowed_categories == {"region": ["a", "b"]}
    assert loaded.thresholds.psi_critical == 0.3


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Неизвестные параметры"):
        DriftConfig.from_dict({"psi": 0.1})


def test_column_thresholds_override_and_validation(tmp_path):
    config = DriftConfig(column_thresholds={"income": {"psi_warning": 0.3, "psi_critical": 0.6}})
    assert config.thresholds_for("income").psi_warning == 0.3
    assert config.thresholds_for("income").js_warning == config.thresholds.js_warning
    assert config.thresholds_for("age") is config.thresholds
    loaded = DriftConfig.from_yaml(config.to_yaml(tmp_path / "c.yaml"))
    assert loaded.column_thresholds == {"income": {"psi_warning": 0.3, "psi_critical": 0.6}}
    with pytest.raises(ValueError, match="Неизвестные пороги"):
        DriftConfig(column_thresholds={"income": {"psi_warn": 0.3}})
