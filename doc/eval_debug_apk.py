"""
Evaluate the image extraction pipeline on ../app-debug.apk.
Produces doc/debug_apk_image.png for visual comparison against
../datasets/images/{class}/adv_*.png reference images.

Run from the app/ directory:
    uv run python doc/eval_debug_apk.py
"""

import re
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.core.image import extract_image

APK_PATH = Path(__file__).parent.parent.parent / "app-debug.apk"
OUT_PATH = Path(__file__).parent / "debug_apk_image.png"


def inspect_dex(apk_path: Path) -> None:
    print(f"\nAPK: {apk_path}")
    with zipfile.ZipFile(apk_path) as z:
        dex_names = sorted(
            n for n in z.namelist()
            if re.fullmatch(r"classes\d*\.dex", n)
        )
        if not dex_names:
            print("  No DEX files found.")
            return
        for name in dex_names:
            raw = z.read(name)
            size = len(raw)
            valid = size >= 8 and raw.startswith(b"dex\n")
            if valid and size >= 112:
                data_off = struct.unpack_from("<I", raw, 108)[0]
                ident_size = data_off - 112
                data_size = size - data_off
                print(
                    f"  {name}: {size:,} bytes | "
                    f"header=112 | ident={ident_size:,} | data={data_size:,}"
                )
            else:
                print(f"  {name}: {size:,} bytes (invalid or too small)")


def main() -> None:
    if not APK_PATH.exists():
        print(f"ERROR: APK not found at {APK_PATH}", file=sys.stderr)
        sys.exit(1)

    inspect_dex(APK_PATH)

    print("\nExtracting image...")
    img = extract_image(str(APK_PATH))

    assert img.size == (64, 64), f"Expected 64x64, got {img.size}"
    assert img.mode == "RGB",    f"Expected RGB, got {img.mode}"

    img.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print(f"Size: {img.size}, mode: {img.mode}")
    print("\nCompare against reference images in:")
    print("  ../datasets/images/{adware,banking,benign,riskware,sms}/adv_*.png")


if __name__ == "__main__":
    main()
