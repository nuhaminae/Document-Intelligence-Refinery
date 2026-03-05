# src/strategies/fasttext_extractor.py
# script to define extraction strategies with pdfplumber

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

                # Each line becomes an LDU
                for i, line in enumerate(text.splitlines()):
                    ldu_id = f"ldu_{page_num}_{i}"
                    bbox = (0, 0, page.width, page.height)  # simplified bbox

                    ldus.append(
                        LDU(
                            ldu_id=ldu_id,
                            type=LDUType.paragraph,
                            text=line,
                            table_data=None,
                            figure_ref=None,
                            bbox=bbox,
                            page_number=page_num,
                        )
                    )

                    provenance.append(
                        ProvenanceChain(
                            ldu_id=ldu_id,
                            strategy_used=StrategyType.fasttext,
                            source_bbox=bbox,
                            source_page=page_num,
                            transformations=["raw_text_extraction"],
                            confidence_score=profile.triage_confidence,
                        )
                    )

        return ExtractedDocument(
            document_id=profile.document_id,
            strategy_used=StrategyType.fasttext,
            content_blocks=ldus,
            provenance_chain=provenance,
            extraction_confidence=profile.triage_confidence,
        )
