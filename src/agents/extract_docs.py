# src/agents/extract_docs.py
# Regenerate ExtractedDocument JSONs for chunker

import json
import logging
import os

from src.models.models import (
    DocumentProfile,
    ExtractedDocument,
    LayoutComplexity,
    OriginType,
    StrategyType,
)
from src.strategies.fasttext_extractor import FastTextExtractor
from src.strategies.layout_extractor import LayoutExtractor
from src.strategies.vision_extractor import VisionExtractor

logging.basicConfig(level=logging.INFO)

PROFILES_DIR = ".refinery/profiles"
LEDGER_PATH = ".refinery/extraction_ledger.jsonl"
OUTPUT_DIR = ".refinery/extracted"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_extractor(strategy, file_path):
    if strategy == StrategyType.fasttext:
        return FastTextExtractor(file_path)
    elif strategy == StrategyType.layout_aware:
        return LayoutExtractor(file_path)
    elif strategy == StrategyType.vision_augmented:
        return VisionExtractor(file_path)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def migrate_extracted_documents(input_dir="data"):
    # Load ledger entries into a dict keyed by document_id
    ledger_entries = {}
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                ledger_entries[entry["document_id"]] = entry

    for fname in os.listdir(input_dir):
        if not fname.endswith(".pdf"):
            continue
        doc_id = os.path.splitext(fname)[0]
        file_path = os.path.join(input_dir, fname)

        profile_path = os.path.join(PROFILES_DIR, f"{doc_id}.json")
        if not os.path.exists(profile_path):
            logging.warning(f"No profile found for {doc_id}, skipping.")
            continue

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        profile = DocumentProfile(**profile_data)

        # Use strategy from ledger if available, else re-select
        if doc_id in ledger_entries:
            strategy = StrategyType(ledger_entries[doc_id]["strategy_used"])
            extraction_confidence = ledger_entries[doc_id]["extraction_confidence"]
            cost_estimate = ledger_entries[doc_id]["cost_estimate"]
        else:
            # Fallback: re-select strategy
            density, whitespace = profile.char_density, profile.whitespace_ratio
            image_ratio = getattr(profile, "image_area_ratio", 0)
            has_fonts = getattr(profile, "has_font_metadata", False)

            if density < 0.0005 or whitespace > 0.6:
                strategy = StrategyType.vision_augmented
            elif profile.layout_complexity in [
                LayoutComplexity.multi_column,
                LayoutComplexity.table_heavy,
            ]:
                strategy = StrategyType.layout_aware
            elif (
                has_fonts
                and density > 0.001
                and profile.origin_type == OriginType.digital
            ):
                strategy = StrategyType.layout_aware
            elif density >= 0.001 and whitespace < 0.5 and image_ratio < 0.3:
                strategy = StrategyType.fasttext
            else:
                strategy = StrategyType.fasttext
            extraction_confidence = profile.triage_confidence
            cost_estimate = None

        logging.info(
            f"Regenerating ExtractedDocument for {doc_id} using {strategy.value}"
        )
        extractor = get_extractor(strategy, file_path)
        extracted_doc: ExtractedDocument = extractor.extract(profile)

        # Build unified schema
        unified_doc = {
            "document_id": doc_id,
            "strategy_used": strategy.value,
            "extraction_confidence": extraction_confidence,
            "content_blocks": [
                ldu.model_dump() for ldu in extracted_doc.content_blocks
            ],
            "provenance_chain": [
                prov.model_dump() for prov in extracted_doc.provenance_chain
            ],
            "page_indexes": [
                pi.model_dump() for pi in getattr(extracted_doc, "page_indexes", [])
            ],
            "cost_estimate": cost_estimate,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(unified_doc, f, indent=2)
        logging.info(f"Saved unified ExtractedDocument → {out_path}")


if __name__ == "__main__":
    migrate_extracted_documents()
