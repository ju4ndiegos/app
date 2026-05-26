import pytest
from pipeline.core.tabular import extract_tabular


def test_tabular_correct_columns(apk_path, schema):
    row = extract_tabular(apk_path, schema)
    expected_keys = {"apk_name"} | set(schema) | {"Class"}
    assert set(row.keys()) == expected_keys


def test_tabular_binary_values(apk_path, schema):
    row = extract_tabular(apk_path, schema)
    bad = {col: row[col] for col in schema if row[col] not in (0, 1)}
    assert len(bad) == 0, f"Non-binary values: {bad}"


def test_tabular_internet_permission(apk_path, schema):
    row = extract_tabular(apk_path, schema, label="benign")
    assert row.get("permission.INTERNET") == 1, "INTERNET permission should be present in this app"


def test_tabular_label(apk_path, schema):
    row = extract_tabular(apk_path, schema, label="benign")
    assert row["Class"] == "benign"
