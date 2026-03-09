# src/agents/chunker.py
# Script to semantically chunk ExtractedDocument JSONs based on rules
# Rules are found in rubric/extraction_rules.yaml

import json
import logging
import os
import re
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO)


class Chunk(BaseModel):
    type: str
    content: Any
    metadata: Dict[str, Any] = {}
    relationships: List[Dict[str, Any]] = []


class ChunkValidator:
    """
    Validates chunks against the Logical Document Unit (LDU) constitution
    defined in extraction_rules.yaml.
    """

    def __init__(self, rubric_path="rubric/extraction_rules.yaml"):
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.rules = self.config["chunking_rules"]
        self.thresholds = self.config.get("thresholds", {})

    def validate(self, chunks: List[Chunk]) -> List[Chunk]:
        validated = []
        max_ldus_per_page = self.thresholds.get("max_ldus_per_page", 1000)

        for chunk in chunks:
            try:
                chunk = Chunk(**chunk.model_dump())
            except ValidationError as e:
                logging.warning(f"Chunk validation failed: {e}")
                continue

            # Rule checks
            if chunk.type == "table" and "header" not in chunk.metadata:
                logging.warning("Table missing header metadata")
            if chunk.type == "figure" and "caption" not in chunk.metadata:
                logging.warning("Figure missing caption metadata")
            if (
                chunk.type == "list"
                and isinstance(chunk.content, list)
                and len(chunk.content) > self.thresholds.get("max_list_items", 200)
            ):
                logging.warning("List exceeds max_list_items, consider splitting")
            if chunk.type == "section" and "header" not in chunk.metadata:
                logging.warning("Section missing header metadata")

            # Cross-reference detection
            if isinstance(chunk.content, str):
                refs = re.findall(r"see\s+Table\s+\d+", chunk.content)
                if refs:
                    chunk.relationships.append({"cross_references": refs})

            # Page group validation
            if chunk.type == "page_group":
                page_num = chunk.metadata.get("page_number")
                ldu_ids = chunk.content if isinstance(chunk.content, list) else []
                if chunk.metadata.get("synthetic"):
                    logging.info(f"Synthetic page index used for page {page_num}")
                if not ldu_ids:
                    logging.warning(f"Page group {page_num} has no LDUs")
                elif len(ldu_ids) > max_ldus_per_page:
                    logging.warning(
                        f"Page {page_num} has {len(ldu_ids)} LDUs, exceeding threshold {max_ldus_per_page}"
                    )

            validated.append(chunk)

        # --- Post-validation merge pass ---
        merged_validated = []
        buffer = None
        for chunk in validated:
            if (
                buffer
                and buffer.type == "section"
                and chunk.type == "section"
                and len(str(buffer.content)) < 100
                and len(str(chunk.content)) < 100
            ):
                # Merge consecutive short sections
                buffer.content = f"{buffer.content} {chunk.content}"
            else:
                if buffer:
                    merged_validated.append(buffer)
                buffer = chunk
        if buffer:
            merged_validated.append(buffer)

        return merged_validated


class Chunker:
    """
    Applies chunking rules to ExtractedDocument objects and validates them.
    Produces LDUs in memory for downstream FactTable and vector store ingestion.
    """

    def __init__(self, rubric_path="rubric/extraction_rules.yaml"):
        self.validator = ChunkValidator(rubric_path)

    def is_trivial_text(self, text: str, min_length: int = 25) -> bool:
        """
        Decide if a text block is too small to stand alone.
        Language-aware: avoids bias against Amharic by only treating
        digits/punctuation as trivial in non-Latin scripts.
        """
        if not text or not text.strip():
            return True

        stripped = text.strip()

        if stripped.isdigit() or all(ch in ".," for ch in stripped):
            return True

        if any("\u1200" <= ch <= "\u137f" for ch in stripped):
            return False

        if len(stripped) < min_length:
            return True

        return False

    def is_noise_block(self, text: str) -> bool:
        """
        Detect if a text block is just formatting noise.
        Examples: repeated 'c', long digit strings, gibberish.
        Avoids flagging Amharic (Ethiopic Unicode range).
        """
        if not text or not text.strip():
            return True

        stripped = text.strip()

        if any("\u1200" <= ch <= "\u137f" for ch in stripped):
            return False

        if re.fullmatch(r"(.)\1{4,}", stripped):
            return True

        if re.fullmatch(r"\d{5,}", stripped):
            return True

        non_alnum_ratio = sum(1 for ch in stripped if not ch.isalnum()) / len(stripped)
        if non_alnum_ratio > 0.7:
            return True

        return False

    def chunk_document(self, extracted_doc: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk an ExtractedDocument into Logical Document Units (LDUs).
        Standardises all chunks so content is always a dict with 'text' and optional 'ldu_ids'.
        """
        chunks = []
        merged_buffer = []

        for block in extracted_doc.get("content_blocks", []):
            ldu_id = block.get("ldu_id")
            ldu_type = block.get("type")
            text = block.get("text", "")

            def make_chunk(chunk_type, text_value, extra_meta=None):
                return Chunk(
                    type=chunk_type,
                    content={"text": text_value, "ldu_ids": [ldu_id] if ldu_id else []},
                    metadata=extra_meta or {},
                )

            if ldu_type == "table":
                if merged_buffer:
                    merged_text = " ".join(merged_buffer)
                    chunks.append(make_chunk("section", merged_text))
                    merged_buffer = []
                table_text = json.dumps(block.get("table_data"))
                chunks.append(
                    make_chunk(
                        "table",
                        table_text,
                        {"header": block.get("metadata", {}).get("header")},
                    )
                )

            elif ldu_type == "figure":
                if merged_buffer:
                    merged_text = " ".join(merged_buffer)
                    chunks.append(make_chunk("section", merged_text))
                    merged_buffer = []
                caption = block.get("metadata", {}).get("caption")
                chunks.append(
                    make_chunk("figure", block.get("figure_ref"), {"caption": caption})
                )

            elif ldu_type == "list":
                if merged_buffer:
                    merged_text = " ".join(merged_buffer)
                    chunks.append(make_chunk("section", merged_text))
                    merged_buffer = []
                list_items = block.get("text").splitlines()
                list_text = " ".join(list_items)
                rule = (
                    "list_intact"
                    if len(list_items)
                    <= self.validator.thresholds.get("max_list_items", 200)
                    else "list_split"
                )
                chunks.append(make_chunk("list", list_text, {"rule": rule}))

            elif ldu_type == "equation":
                if merged_buffer:
                    merged_text = " ".join(merged_buffer)
                    chunks.append(make_chunk("section", merged_text))
                    merged_buffer = []
                chunks.append(
                    make_chunk(
                        "equation", block.get("text"), {"rule": "equation_atomic"}
                    )
                )

            elif ldu_type == "footnote":
                if merged_buffer:
                    merged_text = " ".join(merged_buffer)
                    chunks.append(make_chunk("section", merged_text))
                    merged_buffer = []
                chunks.append(
                    make_chunk("footnote", block.get("text"), {"reference_id": ldu_id})
                )

            else:  # default: section
                if self.is_noise_block(text):
                    continue
                elif self.is_trivial_text(text) or len(text.strip()) < 50:
                    merged_buffer.append(text.strip())
                else:
                    if merged_buffer:
                        merged_text = " ".join(merged_buffer)
                        chunks.append(make_chunk("section", merged_text))
                        merged_buffer = []
                    chunks.append(
                        make_chunk(
                            "section",
                            text,
                            {"header": block.get("metadata", {}).get("header")},
                        )
                    )

        if merged_buffer:
            merged_text = " ".join(merged_buffer)
            chunks.append(make_chunk("section", merged_text))

        # Page index fallback
        page_indexes = extracted_doc.get("page_indexes", [])
        max_ldus_per_page = self.validator.thresholds.get("max_ldus_per_page", 1000)

        if not page_indexes:
            logging.info(
                "No page_indexes found, falling back to grouping by page_number"
            )
            synthetic_indexes = {}
            for block in extracted_doc.get("content_blocks", []):
                page_num = block.get("page_number")
                if page_num not in synthetic_indexes:
                    synthetic_indexes[page_num] = []
                synthetic_indexes[page_num].append(block.get("ldu_id"))

            ldu_lookup = {
                block.get("ldu_id"): block.get("text", "")
                for block in extracted_doc.get("content_blocks", [])
            }

            for page_num, ldu_ids in synthetic_indexes.items():
                resolved_texts = [ldu_lookup.get(ldu_id, "") for ldu_id in ldu_ids]
                aggregated_text = " ".join(resolved_texts)

                if len(ldu_ids) > max_ldus_per_page:
                    for i in range(0, len(ldu_ids), max_ldus_per_page):
                        batch = ldu_ids[i : i + max_ldus_per_page]
                        batch_texts = [ldu_lookup.get(ldu_id, "") for ldu_id in batch]
                        chunks.append(
                            Chunk(
                                type="page_group",
                                content={
                                    "text": " ".join(batch_texts),
                                    "ldu_ids": batch,
                                },
                                metadata={
                                    "page_number": page_num,
                                    "rule": "synthetic_page_index",
                                    "synthetic": True,
                                    "batch_index": i // max_ldus_per_page,
                                },
                            )
                        )
                else:
                    chunks.append(
                        Chunk(
                            type="page_group",
                            content={"text": aggregated_text, "ldu_ids": ldu_ids},
                            metadata={
                                "page_number": page_num,
                                "rule": "synthetic_page_index",
                                "synthetic": True,
                            },
                        )
                    )

        validated_chunks = self.validator.validate(chunks)
        logging.info(f"Produced {len(validated_chunks)} validated chunks")
        return validated_chunks


class ChunkLoader:
    """
    Utility to load ExtractedDocument JSONs from disk, run chunking,
    persist results to disk, and generate a summary index with warnings.
    """

    def __init__(
        self,
        extracted_dir=".refinery/extracted",
        rubric_path="rubric/extraction_rules.yaml",
        output_dir=".refinery/chunks",
    ):
        self.extracted_dir = extracted_dir
        self.chunker = Chunker(rubric_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_chunk_all(self) -> Dict[str, List[Chunk]]:
        results = {}
        index_summary = []

        for fname in os.listdir(self.extracted_dir):
            if not fname.endswith(".json"):
                continue
            doc_id = os.path.splitext(fname)[0]
            file_path = os.path.join(self.extracted_dir, fname)

            with open(file_path, "r", encoding="utf-8") as f:
                extracted_doc = json.load(f)

            logging.info(f"Chunking {doc_id} from {file_path}")
            chunks = self.chunker.chunk_document(extracted_doc)
            results[doc_id] = chunks

            # Persist chunks to disk
            out_path = os.path.join(self.output_dir, f"{doc_id}_chunks.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([chunk.model_dump() for chunk in chunks], f, indent=2)
            logging.info(f"Saved chunks → {out_path}")

            # Collect warnings for index
            warnings = []
            for chunk in chunks:
                if chunk.type == "page_group" and chunk.metadata.get("synthetic"):
                    warnings.append(
                        f"Synthetic page index used for page {chunk.metadata.get('page_number')}"
                    )
                if chunk.type == "page_group" and "batch_index" in chunk.metadata:
                    warnings.append(
                        f"Page {chunk.metadata.get('page_number')} split into batch {chunk.metadata['batch_index']}"
                    )

            # Add to index summary
            index_summary.append(
                {
                    "document_id": doc_id,
                    "chunk_count": len(chunks),
                    "output_file": out_path,
                    "warnings": warnings,
                }
            )

        # Save index summary
        index_path = os.path.join(self.output_dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_summary, f, indent=2)
        logging.info(f"Saved index summary → {index_path}")

        logging.info(f"Chunked {len(results)} documents")
        return results


if __name__ == "__main__":
    loader = ChunkLoader()
    all_chunks = loader.load_and_chunk_all()
    for doc_id, chunks in all_chunks.items():
        print(f"{doc_id}: {len(chunks)} chunks")
