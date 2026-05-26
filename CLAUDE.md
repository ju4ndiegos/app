# APK Detector — DMLDroid Pipeline

Django web app that extracts three feature modalities from Android APK files for malware classification.  
Paper: arXiv:2509.11187. Full context: `doc/paper_dmldroid.md`.

## Architecture

Three parallel extractors, all orchestrated by `pipeline/core/pipeline.py`:

| Modality | Output format | Key file |
|----------|--------------|----------|
| Image | 64×64 RGB PNG — DEX semantic encoding | `pipeline/core/image.py` |
| Sequence | Space-separated API method names (max 512 tokens) | `pipeline/core/sequence.py` |
| Tabular | 400 binary manifest features (dict) | `pipeline/core/tabular.py` |

## Image Extraction (focus of this project)

Implements DMLDroid DEX-to-image semantic encoding:

```
DEX file sections  →  RGB channels
──────────────────────────────────
bytes 0–111        →  R  (header, nearly zero for most APKs)
bytes 112–data_off →  B  (identifier: string/type/proto/field/method/class IDs)
bytes data_off–end →  G  (data: bytecode, string data — dominant channel)
```

- All `classes*.dex` files in the APK are concatenated per channel before rendering
- Bytes truncated or zero-padded to fill a 64×64 grid (4096 bytes per channel)
- `data_off` read from DEX header at byte offset 108 (little-endian uint32)

Reference images extracted from CICMalDroid 2020:
```
../datasets/images/
  adware/   adv_<sha256>.png   (2,311 files)
  banking/  adv_<sha256>.png   (3,703 files)
  benign/   adv_<sha256>.png   (6,448 files)
  riskware/ adv_<sha256>.png   (5,942 files)
  sms/      adv_<sha256>.png   (7,674 files)
```

## Dataset

CICMalDroid 2020 — `../datasets/multimodal-cic-andmal2020/`
- `Tabular/` — train/test/adv/obfus CSV, 400 binary features + Class column
- `Sequence/` — JSON files with API call sequences
- `Image/` — PNG splits: train / test / adv / obfus

Schema locked to `train-tabular.csv` column order (400 features).

## Malware Classes

`adware` | `banking` | `benign` | `riskware` | `sms`

## Running the App

```bash
cd app
uv run python manage.py runserver
```

Navigate to http://localhost:8000/, upload an APK file, pick a label → results page shows image, sequence preview, and active tabular features.

## Running Tests

```bash
cd app
uv run pytest
```

Test APK: `pipeline/test_data/sombriyakotlin4.apk`  
Tests cover: pipeline orchestration, image shape/mode, SHA256 hash format.

## Evaluating a Debug APK

```bash
cd app
uv run python doc/eval_debug_apk.py
```

Runs image extraction on `../app-debug.apk` and writes the result to `doc/debug_apk_image.png`.

## Key Files

```
app/
├── CLAUDE.md                        ← this file
├── doc/
│   ├── paper_dmldroid.md            ← paper context (arXiv:2509.11187)
│   ├── eval_debug_apk.py            ← evaluation script for app-debug.apk
│   └── debug_apk_image.png          ← generated: 64×64 RGB of the debug APK
├── pipeline/
│   ├── core/
│   │   ├── image.py                 ← DEX → RGB extraction
│   │   ├── sequence.py              ← API call graph extraction
│   │   ├── tabular.py               ← manifest feature extraction
│   │   └── pipeline.py              ← orchestrator
│   ├── models.py                    ← AnalysisResult Django model
│   ├── views.py                     ← upload + result views
│   └── tests/                       ← pytest test suite
└── pyproject.toml                   ← deps: androguard, django, numpy, pillow
```

## Dependencies

Python >=3.13 | androguard>=4.1.3 | django>=6.0.5 | numpy>=2.4.4 | pillow>=12.2.0
