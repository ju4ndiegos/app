"""
Sequence modality extractor — DMLDroid paper (arXiv:2509.11187)

Builds an API Call Graph (ACG) via androguard and traverses it with DFS
to produce a space-separated token sequence of method names (max 512 tokens).

Fallback (no androguard): scans the DEX string pool for lowercase identifiers.
"""

import re
import struct
import zipfile

MAX_TOKENS = 512

try:
    from androguard.misc import AnalyzeAPK
    _ANDROGUARD = True
except ImportError:
    _ANDROGUARD = False


# ---------------------------------------------------------------------------
# Androguard-based extraction (ACG + DFS)
# ---------------------------------------------------------------------------

def _extract_androguard(apk_path: str) -> list[str]:
    _, _, dx = AnalyzeAPK(apk_path)
    visited: set[str] = set()
    tokens: list[str] = []

    def _dfs(m_analysis):
        key = m_analysis.get_method().get_class_name() + m_analysis.get_method().get_name()
        if key in visited or len(tokens) >= MAX_TOKENS:
            return
        visited.add(key)
        name = m_analysis.get_method().get_name()
        # Skip constructors, static initializers, and synthetic Kotlin/R8 names (contain $)
        if name not in ("<init>", "<clinit>") and "$" not in name and name.isidentifier():
            tokens.append(name)
        for _, callee, _ in m_analysis.get_xref_to():
            if len(tokens) >= MAX_TOKENS:
                break
            _dfs(callee)

    for m in dx.get_methods():
        if len(tokens) >= MAX_TOKENS:
            break
        _dfs(m)

    return tokens[:MAX_TOKENS]


# ---------------------------------------------------------------------------
# Fallback: DEX string-pool scan (no androguard)
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"^[a-z][a-zA-Z0-9]{1,63}$")


def _read_uleb128(raw: bytes, offset: int) -> tuple[int, int]:
    """Return (value, bytes_consumed)."""
    result, shift = 0, 0
    consumed = 0
    while offset + consumed < len(raw):
        b = raw[offset + consumed]
        result |= (b & 0x7F) << shift
        consumed += 1
        if not (b & 0x80) or consumed > 5:
            break
        shift += 7
    return result, consumed


def _extract_fallback(apk_path: str) -> list[str]:
    tokens: list[str] = []
    DEX_MAGIC = b"dex\n"

    with zipfile.ZipFile(apk_path) as z:
        dex_names = sorted(
            n for n in z.namelist()
            if re.fullmatch(r"classes\d*\.dex", n)
        )
        for dex_name in dex_names:
            if len(tokens) >= MAX_TOKENS:
                break
            raw = z.read(dex_name)
            if len(raw) < 112 or not raw.startswith(DEX_MAGIC):
                continue

            n_strings  = struct.unpack_from("<I", raw, 56)[0]
            ids_offset = struct.unpack_from("<I", raw, 60)[0]

            if ids_offset + n_strings * 4 > len(raw):
                continue

            for i in range(n_strings):
                if len(tokens) >= MAX_TOKENS:
                    break
                data_off = struct.unpack_from("<I", raw, ids_offset + i * 4)[0]
                if data_off >= len(raw):
                    continue
                length, consumed = _read_uleb128(raw, data_off)
                s_start = data_off + consumed
                s_bytes = raw[s_start: s_start + length]
                if len(s_bytes) < length:
                    continue
                try:
                    s = s_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if _IDENT_RE.match(s):
                    tokens.append(s)

    return tokens[:MAX_TOKENS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sequence(apk_path: str) -> str:
    """Return space-separated method-name tokens (max 512)."""
    if _ANDROGUARD:
        tokens = _extract_androguard(apk_path)
    else:
        tokens = _extract_fallback(apk_path)
    return " ".join(tokens)
