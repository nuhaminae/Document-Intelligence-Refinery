# Document Intelligence Refinery

[![CI](https://github.com/nuhaminae/Document-Intelligence-Refinery/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/Document-Intelligence-Refinery/actions/workflows/CI.yml)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort Imports](https://img.shields.io/badge/imports-isort-blue.svg)
![Flake8 Lint](https://img.shields.io/badge/lint-flake8-yellow.svg)

## Project Review

The **Document-Intelligence-Refinery** is a modular pipeline for intelligent PDF document processing. It classifies documents into profiles, routes them to the most suitable extraction strategy, and logs provenance for auditability. The system combines lightweight heuristics, layout‑aware parsing, and vision‑augmented extraction to handle diverse document types ranging from simple digital PDFs to scanned, image‑heavy reports.

---

## Key Feature

- **Triage Agent**: Computes metrics (character density, whitespace ratio, bounding box distribution, and image ratio) and classifies documents into profiles.  
- **Extraction Router**: Selects the appropriate strategy (FastText, LayoutAware, VisionAugmented) and escalates if confidence is low.  
- **Strategies**:  
  - *FastTextExtractor*: Uses pdfplumber for native, single‑column PDFs.  
  - *LayoutExtractor*: Uses Docling for multi‑column, table‑heavy, or figure‑heavy PDFs.  
  - *VisionExtractor*: Uses MinerU for scanned/image‑heavy documents.  
- **Provenance Tracking**: Every extracted Logic Document Unit (LDU) is logged with its source, transformations, and confidence.  
- **Ledger & Profiles**: Outputs JSON profiles and an extraction ledger for reproducibility and auditing.
  
---

## Table of Contents

- [Document Intelligence Refinery](#document-intelligence-refinery)
  - [Project Review](#project-review)
  - [Key Feature](#key-feature)
  - [Table of Contents](#table-of-contents)
  - [Project Structure (Snippet)](#project-structure-snippet)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
  - [Extraction Strategy Decision Tree](#extraction-strategy-decision-tree)
  - [Project Status](#project-status)

---

## Project Structure (Snippet)

```bash
Document-Intelligence-Refinery/
├── src/
│   ├── agents/
│   │   ├── extractor.py        # Router orchestrating strategies
│   │   └── triage.py           # Preliminary classification
│   ├── models/
│   │   └── models.py           # Pydantic schemas
│   ├── strategies/
│   │   ├── fasttext_extractor.py
│   │   ├── layout_extractor.py
│   │   └── vision_extractor.py
│   └── utils/
│       └── preprocessor.py     # Deduplication/preprocessing
├── tests/                      # Test suite
├── data/                       # Input PDFs
├── .refinery/                  # Profiles and ledger outputs
├── rubric/                     # Extraction rules
├── DOMAIN_NOTES.md             # On Boarding doc
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.12+  
- Git  

### Setup

```bash
# Clone repo
git clone https://github.com/nuhaminae/Document-Intelligence-Refinery.git
cd Document-Intelligence-Refinery

# Install dependencies
pip install uv
uv pip install -r pyproject.toml
```

---

## Usage

1. Place your PDFs in `data/`.  
2. Run preporcessor script to deduplicate data:

  ```bash
  python -m src.utils.preprocessor
  ```

3. Set document language (Optional):

  ```bash
  # src/strategies/vision_extractor.py
  ...
    data = pytesseract.image_to_data(
                img, lang="amh+eng", output_type=pytesseract.Output.DICT
            )
   ```

4. Run the extractor:  

   ```bash
   python -m src.agents.extractor
   ```

**Outputs**:  

- Profiles → `.refinery/profiles/*.json`  
- Ledger → `.refinery/extraction_ledger.jsonl`  
- Preprocessing logs → `.refinery/preprocessor.log`

---

## Extraction Strategy Decision Tree

```mermaid
flowchart TB
    A(["PDF Input"]) --> B{"Character Density"}
    B --> C["High Density (>=0.0015), Consistent"] & D["Medium Density(0.0005–0.0015), Variable"] & E["Low Density (&lt;0.0005), Sparse"]
    C --> F("pdfplumber (FastText)")
    D --> G{"BBox Distribution"}
    G --> H["Narrow x_range (single column)"] & I["Wide x_range (multi-column/tables)"] & K["Highly irregular layout"]
    H --> F
    I --> J["Docling"]
    K --> L["Vision-Augmented (MinerU/VLM)"]
    E --> M{"Whitespace Ratio"}
    M --> N["Low whitespace (<0.3)"]
    N --> F
    M --> O["Moderate whitespace (0.3 -0.6)"]
    O --> J
    M --> P["High whitespace(>0.6)"]
    P --> L
    F --> Q["Chunking + PageIndex"]
    J --> Q
    L --> Q
    Q --> R(["Query Interface"])
```

---

## Project Status

The project is ongoing. Check the [commit history](https://github.com/nuhaminae/Document-Intelligence-Refinery/).
