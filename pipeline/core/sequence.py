"""
Sequence modality extractor — DMLDroid paper (arXiv:2509.11187)

Extracts method names directly from the DEX binary (method_ids section),
without androguard's full call-graph analysis. This avoids the memory and
time cost of AnalyzeAPK() on large APKs while producing the same vocabulary
(actual declared method names from the DEX).

DEX method_ids section layout (each entry is 8 bytes):
  +0  class_idx  u16  → index into type_ids
  +2  proto_idx  u16  → index into proto_ids
  +4  name_idx   u32  → index into string_ids  ← we read this
"""

import re
import struct
import zipfile

MAX_TOKENS = 512


def _read_uleb128(raw: bytes, offset: int) -> tuple[int, int]:
    """Return (value, bytes_consumed)."""
    result, shift, consumed = 0, 0, 0
    while offset + consumed < len(raw):
        b = raw[offset + consumed]
        result |= (b & 0x7F) << shift
        consumed += 1
        if not (b & 0x80) or consumed > 5:
            break
        shift += 7
    return result, consumed


def _method_names_from_dex(raw: bytes) -> list[str]:
    """
    Extract unique method names from one DEX file's method_ids section.
    Returns names in definition order, skipping constructors and synthetic names.
    """
    if len(raw) < 112 or not raw.startswith(b"dex\n"):
        return []

    n_strings      = struct.unpack_from("<I", raw, 56)[0]
    string_ids_off = struct.unpack_from("<I", raw, 60)[0]
    n_methods      = struct.unpack_from("<I", raw, 88)[0]
    method_ids_off = struct.unpack_from("<I", raw, 92)[0]

    # Sanity checks
    if (string_ids_off + n_strings * 4 > len(raw) or
            method_ids_off + n_methods * 8 > len(raw)):
        return []

    names: list[str] = []
    seen:  set[str]  = set()

    for i in range(n_methods):
        item_off = method_ids_off + i * 8
        name_idx = struct.unpack_from("<I", raw, item_off + 4)[0]

        ptr = string_ids_off + name_idx * 4
        if ptr + 4 > len(raw):
            continue
        data_off = struct.unpack_from("<I", raw, ptr)[0]
        if data_off >= len(raw):
            continue

        length, consumed = _read_uleb128(raw, data_off)
        s_start = data_off + consumed
        s = raw[s_start: s_start + length].decode("utf-8", errors="ignore")

        if (s and s not in ("<init>", "<clinit>")
                and "$" not in s
                and s.isidentifier()
                and s not in seen):
            seen.add(s)
            names.append(s)

    return names


def extract_sequence(apk_path: str) -> str:
    """
    Return space-separated method-name tokens (max 512) extracted from all
    classes*.dex files inside the APK without running androguard analysis.
    """
    tokens: list[str] = []
    seen:   set[str]  = set()

    with zipfile.ZipFile(apk_path) as z:
        dex_names = sorted(
            n for n in z.namelist()
            if re.fullmatch(r"classes\d*\.dex", n)
        )
        for dex_name in dex_names:
            if len(tokens) >= MAX_TOKENS:
                break
            try:
                raw = z.read(dex_name)
            except Exception:
                continue
            for name in _method_names_from_dex(raw):
                if name not in seen:
                    seen.add(name)
                    tokens.append(name)
                if len(tokens) >= MAX_TOKENS:
                    break

    return " ".join(tokens[:MAX_TOKENS])
