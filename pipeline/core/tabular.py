"""
Tabular modality extractor — DMLDroid paper (arXiv:2509.11187)

Extracts binary features (0/1) from the AndroidManifest.xml:
  - permissions   → permission.* columns
  - intent actions → action.* columns
  - intent categories → category.* columns
  - service/activity/receiver names → bare-name columns

Schema is locked to the 400 columns in the CICMalDroid 2020 training CSV.
Requires androguard for binary AXML parsing.
"""

import csv
import hashlib
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.csv"

try:
    from androguard.core.apk import APK as AndroAPK
    _ANDROGUARD = True
except ImportError:
    try:
        from androguard.core.bytecodes.apk import APK as AndroAPK  # androguard <4
        _ANDROGUARD = True
    except ImportError:
        _ANDROGUARD = False


def load_schema(schema_path: Path = _SCHEMA_PATH) -> list[str]:
    """Return the 400 binary feature column names (no apk_name, no Class)."""
    with open(schema_path, newline="") as f:
        headers = next(csv.reader(f))
    return headers[1:-1]


def _parse_manifest(apk_path: str) -> tuple[set, set, set, set]:
    """
    Returns (permissions, actions, categories, component_names).
    All values are base names (last component after the final dot).
    """
    if not _ANDROGUARD:
        return set(), set(), set(), set()

    a = AndroAPK(apk_path)

    permissions = {p.split(".")[-1] for p in (a.get_permissions() or [])}

    component_names: set[str] = set()
    for comp in (
        list(a.get_activities() or [])
        + list(a.get_services() or [])
        + list(a.get_receivers() or [])
    ):
        component_names.add(comp.split(".")[-1])

    actions:    set[str] = set()
    categories: set[str] = set()

    # get_intent_filters returns {comp_name: {"action": [...], "category": [...]}}
    for comp_type in ("activity", "service", "receiver"):
        filters = a.get_intent_filters(comp_type, None) or {}
        for _, details in filters.items():
            for act in details.get("action", []):
                actions.add(act.split(".")[-1])
            for cat in details.get("category", []):
                categories.add(cat.split(".")[-1])

    return permissions, actions, categories, component_names


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_tabular(apk_path: str, schema: list[str], label: str = "") -> dict:
    """
    Return a dict with apk_name, all 400 schema columns (0/1), and Class.
    """
    permissions, actions, categories, component_names = _parse_manifest(apk_path)

    row: dict = {"apk_name": _sha256(apk_path)}

    for col in schema:
        if col.startswith("permission."):
            feat = col[len("permission."):]
            row[col] = 1 if feat in permissions else 0
        elif col.startswith("action."):
            feat = col[len("action."):]
            row[col] = 1 if feat in actions else 0
        elif col.startswith("category."):
            feat = col[len("category."):]
            row[col] = 1 if feat in categories else 0
        else:
            # Column IS the component base name (e.g. "UpdateService")
            row[col] = 1 if col in component_names else 0

    row["Class"] = label
    return row
