# src/strategies/fasttext_extractor.py
# script to define extraction strategies with pdfplumber

import hashlib
from typing import List

import pdfplumber

from src.models.models import (
    LDU,
    DocumentProfile,
    ExtractedDocument,
    LDUType,
    ProvenanceChain,
    StrategyType,
)


class FastTextExtractor:
    """
    Strategy A: FastText (pdfplumber)
    - Best for native, single-column PDFs with high character density.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self, profile: DocumentProfile) -> ExtractedDocument:
        ldus: List[LDU] = []
        provenance: List[ProvenanceChain] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                lines = text.splitlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    ldu_id = f"ldu_{page_num}_{i}"
                    bbox = (0, 0, page.width, page.height)  # simplified bbox

                    # --- Detect captions ---
                    if line.lower().startswith(("figure", "fig", "table")):
                        ldu_type = LDUType.caption
                        content = line
                        i += 1

                    # --- Detect lists (group consecutive items) ---
                    elif line.startswith(("-", "•")) or line[0].isdigit():
                        ldu_type = LDUType.paragraph
                        list_items = [line]
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j].strip()
                            if (
                                next_line.startswith(("-", "•"))
                                or next_line[0].isdigit()
                            ):
                                list_items.append(next_line)
                                j += 1
                            else:
                                break
                        content = "\n".join(list_items)
                        i = j  # skip ahead past grouped list

                    else:
                        ldu_type = LDUType.paragraph
                        content = line
                        i += 1

                    # --- Compute content hash ---
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=ldu_type,
                            text=content,
                            table_data=None,
                            figure_ref=None,
                            bbox=bbox,
                            page_number=page_num,
                        )
                    )

                    # --- Provenance ---
                    provenance.append(
                        ProvenanceChain(
                            ldu_id=ldu_id,
                            strategy_used=StrategyType.fasttext,
                            source_bbox=bbox,
                            source_page=page_num,
                            transformations=["raw_text_extraction"],
                            confidence_score=profile.triage_confidence,
                            content_hash=content_hash,
                        )
                    )

        return ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.fasttext,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=profile.triage_confidence,
        )
