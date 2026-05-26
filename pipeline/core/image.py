"""
Image modality extractor — DMLDroid paper (arXiv:2509.11187)

Semantic RGB encoding from DEX sections:
  R channel → header bytes (fixed 112-byte DEX header)
  G channel → data section (code items, string data, annotations, …)
  B channel → identifier bytes (string IDs, type IDs, proto IDs,
               field IDs, method IDs, class defs)

All classes*.dex files in the APK are processed and their sections concatenated
per channel before building the output image.

The 256-wide spatial layout of the DEX file (used by the paper to produce native
256×256 images) maps data_off to row 50 (12800 / 256 = 50), which is why G
starts at row 50 and B occupies rows 0-55 in the dataset images.
"""

import re
import struct
import zipfile

import numpy as np
from PIL import Image

DEFAULT_SIZE = 64  # pixels per side for model input


def _dex_section_offsets(raw: bytes) -> tuple[int, int]:
    """
    Return (ident_start=112, data_start) byte offsets inside a DEX file.

    DEX header layout (all little-endian uint32):
      offset 96  → class_defs_size
      offset 100 → class_defs_off
      offset 104 → data_size
      offset 108 → data_off   ← start of data section
    """
    if len(raw) < 112:
        return 112, 112

    class_defs_size = struct.unpack_from("<I", raw,  96)[0]
    class_defs_off  = struct.unpack_from("<I", raw, 100)[0]
    data_off        = struct.unpack_from("<I", raw, 108)[0]

    if data_off < 112 or data_off > len(raw):
        data_off = class_defs_off + class_defs_size * 32
        data_off = max(112, min(data_off, len(raw)))

    return 112, data_off


def _to_channel(data: bytes, size: int) -> np.ndarray:
    """Lay bytes into a size×size uint8 array (truncate or zero-pad)."""
    total = size * size
    arr = np.frombuffer(data, dtype=np.uint8) if data else np.zeros(0, dtype=np.uint8)
    if len(arr) >= total:
        arr = arr[:total]
    else:
        arr = np.pad(arr, (0, total - len(arr)))
    return arr.reshape(size, size)


def extract_image(apk_path: str, size: int = DEFAULT_SIZE) -> Image.Image:
    """
    Convert all DEX files in the APK to a semantic RGB image.

    Channel assignment:
      R = header bytes  (112 bytes per DEX file, nearly zero in dataset because tiny)
      G = data section  (code, string data — dominant channel)
      B = identifier bytes between header and data
          (string IDs, type IDs, proto IDs, field IDs, method IDs, class defs)

    Args:
        apk_path: path to the APK file.
        size: output image side length in pixels (e.g. 64 or 256).

    Returns a PIL Image ready to be saved as PNG.
    """
    header_bytes = bytearray()
    ident_bytes  = bytearray()
    data_bytes   = bytearray()

    with zipfile.ZipFile(apk_path) as z:
        dex_names = sorted(
            n for n in z.namelist()
            if re.fullmatch(r"classes\d*\.dex", n)
        )
        for name in dex_names:
            raw = z.read(name)
            if len(raw) < 8 or not raw.startswith(b"dex\n"):
                continue
            ident_start, data_start = _dex_section_offsets(raw)
            header_bytes.extend(raw[:ident_start])
            ident_bytes.extend(raw[ident_start:data_start])
            data_bytes.extend(raw[data_start:])

    r = _to_channel(bytes(header_bytes), size)
    g = _to_channel(bytes(data_bytes),   size)   # data  → G (dominant)
    b = _to_channel(bytes(ident_bytes),  size)   # ident → B (medium)

    rgb = np.stack([r, g, b], axis=2)
    return Image.fromarray(rgb, mode="RGB")