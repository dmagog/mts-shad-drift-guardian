"""Тесты Schema Validation."""
import pandas as pd

from drift_guardian.schema import validate_schema


def test_identical_schema_is_clean():
    ref = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    cur = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})
    assert validate_schema(ref, cur) == []


def test_missing_column_is_critical():
    ref = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    cur = pd.DataFrame({"a": [3, 4]})
    issues = validate_schema(ref, cur)
    assert any(i.check == "missing_column" and i.severity == "critical" for i in issues)


def test_extra_column_is_warning():
    ref = pd.DataFrame({"a": [1, 2]})
    cur = pd.DataFrame({"a": [3, 4], "b": ["x", "y"]})
    issues = validate_schema(ref, cur)
    assert any(i.check == "extra_column" and i.severity == "warning" for i in issues)


def test_dtype_change_is_critical():
    ref = pd.DataFrame({"a": [1.0, 2.0]})
    cur = pd.DataFrame({"a": ["1", "2"]})
    issues = validate_schema(ref, cur)
    assert any(i.check == "dtype_mismatch" and i.severity == "critical" for i in issues)


def test_empty_current_is_critical():
    ref = pd.DataFrame({"a": [1, 2]})
    cur = pd.DataFrame({"a": []})
    issues = validate_schema(ref, cur)
    assert any(i.check == "empty_current" for i in issues)
