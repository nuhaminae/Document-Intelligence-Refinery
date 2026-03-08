# Document Intelligence Refinery

[![CI](https://github.com/nuhaminae/Document-Intelligence-Refinery/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/Document-Intelligence-Refinery/actions/workflows/CI.yml)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort Imports](https://img.shields.io/badge/imports-isort-blue.svg)
![Flake8 Lint](https://img.shields.io/badge/lint-flake8-yellow.svg)

## Project Review

The **Document-Intelligence-Refinery** is a modular, containerized pipeline for intelligent PDF document processing. It classifies documents into corpus classes, routes them to the most suitable extraction strategy, and logs provenance for auditability. The system combines lightweight heuristics, layout‑aware parsing, OCR vision extraction, and fact/vector ingestion to handle diverse document types ranging from simple digital PDFs to scanned, image‑heavy reports and numeric tables.

The pipeline is fully Dockerized, reproducible, and demo‑ready. It supports both **batch runs** (pipeline service) and **interactive querying** (FastAPI query agent exposed at `localhost:8000`).

---

## Key Feature

- **Corpus Coverage**: Representative files selected from each class (least chunk size):
  - Class A (native text): `Company_Profile_2024_25.pdf`
  - Class B (scanned/image-based): `2021_Audited_Financial_Statement_Report.pdf`
  - Class C (mixed text + tables): `CBE Annual Report 2012-13.pdf`
  - Class D (table-heavy numeric): `Consumer Price Index March 2025.pdf`

- **Triage Agent**: Computes metrics (character density, whitespace ratio, bounding box distribution, image ratio) and classifies documents into profiles.

- **Extraction Router**: Selects the appropriate strategy (FastText, LayoutAware, VisionAugmented) and escalates if confidence is low.

- **Strategies**:  
  - *FastTextExtractor*: Uses pdfplumber for native, single‑column PDFs.  
  - *LayoutExtractor*: Uses Docling for multi‑column, table‑heavy, or figure‑heavy PDFs.  
  - *VisionExtractor*: Uses MinerU + pytesseract for scanned/image‑heavy documents.  

- **Fact Extraction & Vectorization**:  
  - Facts stored in SQLite (`facts.db`).  
  - Embeddings stored in ChromaDB (`vector_store`).  

- **Audit Agent**: Verifies extracted claims against source provenance.  

- **Query Agent**: FastAPI interface for semantic search, page index navigation, and fact retrieval.  

- **Provenance Tracking**: Every extracted Logic Document Unit (LDU) is logged with its source, transformations, and confidence.  

- **Ledger & Profiles**: Outputs JSON profiles and an extraction ledger for reproducibility and auditing.  

- **Dockerization**:  
  - Individual services for each agent.  
  - Unified `pipeline` service for sequential runs.  
  - Query agent exposed at `http://localhost:8000` for demo.  

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
    - [Batch Pipeline Run](#batch-pipeline-run)
    - [Individual Agent Runs](#individual-agent-runs)
    - [Query Agent (Interactive)](#query-agent-interactive)
  - [Extraction Strategy Decision Tree](#extraction-strategy-decision-tree)
  - [Dockerization](#dockerization)
  - [Project Status](#project-status)

---

## Project Structure (Snippet)

```bash
Document-Intelligence-Refinery/
├── src/
│   ├── agents/
│   │   ├── extractor_rubric_config.py
│   │   ├── extract_docs.py
│   │   ├── chunkr.py
│   │   ├── indexer.py
│   │   ├── fact_extractor.py
│   │   ├── vector_ingestor.py
│   │   ├── audit_agent.py
│   │   └── query_agent.py
│   ├── strategies/
│   │   ├── fasttext_extractor.py
│   │   ├── layout_extractor.py
│   │   └── vision_extractor.py
│   └── utils/
│       └── preprocessor.py
├── tests/
├── data/
├── .refinery/
├── rubric/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+  
- Git  
- Docker & Docker Compose  

### Setup

```bash
# Clone repo
git clone https://github.com/nuhaminae/Document-Intelligence-Refinery.git
cd Document-Intelligence-Refinery

# Build Docker image
docker-compose build
```

---

## Usage

### Batch Pipeline Run

```bash
docker-compose run --rm pipeline
```

### Individual Agent Runs

```bash
docker-compose run --rm preprocessor
docker-compose run --rm extractor
docker-compose run --rm extract_docs
docker-compose run --rm chunkr
docker-compose run --rm indexer
docker-compose run --rm fact_extractor
docker-compose run --rm vector_ingestor
docker-compose run --rm audit_agent
docker-compose run --rm query_agent
```

### Query Agent (Interactive)

```bash
docker-compose up query_agent
```

Open `http://localhost:8000` [(localhost in Bing)](https://www.bing.com/search?q="http%3A%2F%2Flocalhost%3A8000%2F") to query facts interactively.

**Outputs**:  

- Profiles → `.refinery/profiles/*.json`  
- Ledger → `.refinery/extraction_ledger.jsonl`  
- Facts → `.refinery/facts.db`  
- Vectors → `.refinery/vector_store/`  
- Index → `.refinery/pageindex/`  

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

## Dockerization

- **Dockerfile**: Defines environment (Python 3.11, OCR, dependencies).  
- **docker-compose.yml**: Defines services for each agent and unified `pipeline`.  
- **Exposed Query Agent**: Accessible at `http://localhost:8000`. 

---

## Project Status

The project is completed. Check the [commit history](https://github.com/nuhaminae/Document-Intelligence-Refinery/).
