# DMLDroid — Paper Reference

**Full title**: DMLDroid: Deep Multimodal Fusion Framework for Android Malware Detection with Resilience to Code Obfuscation and Adversarial Perturbations  
**Source**: arXiv:2509.11187  
**Fetched**: 2026-05-22

---

## Abstract

DMLDroid is a multimodal deep learning framework for Android malware detection that combines three complementary feature representations extracted from APK files:

- **Tabular**: permissions and intents from `AndroidManifest.xml`
- **Image**: semantic RGB encoding of DEX binary sections
- **Sequence**: API call graph traversal method names

Key results:
- **97.98% accuracy** on original (unmodified) malware samples
- **98%+ F1-score** sustained under code obfuscation and adversarial perturbations
- Superior resilience vs. single-modality baselines

---

## Dataset: CICMalDroid 2020

Five classes of Android applications:

| Class | Description |
|-------|-------------|
| adware | Apps displaying unwanted advertisements |
| banking | Banking trojans targeting financial credentials |
| benign | Legitimate applications |
| riskware | Potentially harmful but not explicitly malicious |
| sms | SMS-based malware (premium-rate fraud, spyware) |

Test scenarios beyond the baseline:
- **obfus**: Obfuscated variants (variable renaming, code reshuffling)
- **adv**: Adversarially perturbed samples

Dataset splits available in `../datasets/multimodal-cic-andmal2020/`:
- `Tabular/` — CSV files with 400 binary features + Class column
- `Sequence/` — JSON files with API call sequences
- `Image/` — PNG images split into train/test/adv/obfus

Pre-extracted image references (by class) are in `../datasets/images/{class}/adv_<sha256>.png` (~26,078 total).

---

## Image Extraction Pipeline

### Core Concept

APK files contain one or more DEX (Dalvik Executable) bytecode files (`classes.dex`, `classes2.dex`, …). DMLDroid maps the three structural sections of each DEX file to the R, G, B channels of a 64×64 PNG image.

### DEX File Structure

```
┌─────────────────────────────┐ offset 0
│  HEADER (112 bytes)         │         → R channel
├─────────────────────────────┤ offset 112
│  IDENTIFIER SECTION         │         → B channel
│  (string IDs, type IDs,     │
│   proto IDs, field IDs,     │
│   method IDs, class defs)   │
├─────────────────────────────┤ offset data_off (from header @ offset 108)
│  DATA SECTION               │         → G channel (dominant)
│  (code items, string data,  │
│   annotations, debug info)  │
└─────────────────────────────┘
```

### Channel Assignment

| Channel | Source | DEX Bytes | Notes |
|---------|--------|-----------|-------|
| R | Header | `raw[0:112]` | Fixed 112 bytes; nearly zero-valued for most APKs |
| G | Data section | `raw[data_off:]` | Largest section; contains actual bytecode |
| B | Identifier section | `raw[112:data_off]` | String/type/proto/field/method/class IDs |

`data_off` is read from the DEX header at byte offset 108 (little-endian uint32).

### Multi-DEX Handling

All `classes*.dex` files in the APK are processed. Bytes from each file are **concatenated** per channel before building the image:

```python
for each classes*.dex:
    header_bytes  += raw[0:112]
    ident_bytes   += raw[112:data_off]
    data_bytes    += raw[data_off:]

R = to_channel(header_bytes)   # truncate/pad to 64×64
G = to_channel(data_bytes)     # truncate/pad to 64×64
B = to_channel(ident_bytes)    # truncate/pad to 64×64
image = RGB stack
```

### Spatial Layout Note

The paper uses a 256-wide layout producing native 256×256 images. The `data_off` in a typical DEX file is ~12800 bytes, which maps to row 50 in that layout (12800 / 256 = 50). This implementation uses 64×64 for model input efficiency.

### Implementation

`pipeline/core/image.py` → `extract_image(apk_path: str) -> PIL.Image`

---

## Sequence Extraction Pipeline

Extracts an API Call Graph (ACG) via DFS traversal using androguard:

1. Parse the APK with androguard's `APK` + `DalvikVMFormat`
2. Build a directed call graph
3. DFS traversal collecting method names (skip constructors, synthetic names)
4. Output: space-separated string, max 512 tokens

Fallback (no androguard): scan the DEX string pool for method-like strings.

**Implementation**: `pipeline/core/sequence.py` → `extract_sequence(apk_path: str) -> str`

---

## Tabular Extraction Pipeline

Parses `AndroidManifest.xml` (via androguard AXML decoder) and extracts 400 binary features locked to the CICMalDroid 2020 schema:

- `permission.*` — declared permissions (e.g., `permission.INTERNET`)
- `action.*` — intent actions (e.g., `action.BOOT_COMPLETED`)
- `category.*` — intent categories
- Component names — activity/service/receiver/provider names

Schema source: `../datasets/multimodal-cic-andmal2020/Tabular/train-tabular.csv` (column headers).

**Implementation**: `pipeline/core/tabular.py` → `extract_tabular(apk_path, schema, label) -> dict`

---

## Model Architecture

DMLDroid uses four parallel components fused by dynamic weighted fusion:

```
APK ──┬── image.py  ──→ CNN branch       ─┐
      ├── sequence.py ─→ GNN / Transformer ─┼→ Dynamic weighted fusion → Class label
      └── tabular.py ─→ MLP branch       ─┘
```

- **CNN branches**: extract spatial patterns from 64×64 RGB images
- **GNN**: models API call relationships through graph topology
- **Attention mechanisms**: dynamically weight modality importance
- **Fusion layers**: combine per-modality predictions

The multi-view architecture learns obfuscation-invariant features — even if one modality is disrupted by renaming/reshuffling, the others maintain signal.

---

## Key Results

| Test Scenario | Accuracy | F1-Score |
|--------------|----------|----------|
| Original samples | 97.98% | 98.67% |
| Obfuscated (variable renaming, code reshuffling) | >97% | >98% |
| Adversarial perturbations | >97% | >98% |

Evaluation metrics: precision, recall, F1 per class, ROC-AUC curves.
